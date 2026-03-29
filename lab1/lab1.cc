/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * EE597 Spring 2026 - Lab 1
 * NS-3 Simulation of IEEE 802.11 CSMA/CA DCF Saturation Throughput
 *
 * Topology:
 *   One receiver node (index 0) + numNodes transmitter nodes,
 *   all placed within a 10x10 m area so every node is within
 *   radio range of every other node (single collision domain).
 *
 * Usage (from ns-3-dev root):
 *   ./waf --run "lab1 --numNodes=10 --dataRate=10 --caseB=false --simTime=30"
 *
 * Command-line arguments:
 *   --numNodes  : number of transmitter nodes (N)              [default: 10]
 *   --dataRate  : per-node offered data rate in Mbps (R)       [default: 10.0]
 *   --caseB     : false = Case A (CWmin=1,   CWmax=1023)
 *                 true  = Case B (CWmin=63,  CWmax=127)        [default: false]
 *   --simTime   : simulation duration in seconds               [default: 30.0]
 *   --verbose   : enable WiFi / application log components     [default: false]
 *   --csvFile   : output CSV filename                          [default: results.csv]
 *
 * Output:
 *   Appends one row to csvFile:
 *     case, numNodes, dataRate_Mbps, totalThroughput_Mbps, perNodeThroughput_Mbps
 *
 * See Lab1Run.sh to drive all four experiment configurations.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Lab1CsmaCa");

int
main (int argc, char *argv[])
{
  // -----------------------------------------------------------------------
  // Simulation parameters (overridable from the command line)
  // -----------------------------------------------------------------------
  uint32_t    numNodes  = 10;
  double      dataRate  = 10.0;           // Mbps per transmitter node
  bool        caseB     = false;
  double      simTime   = 30.0;           // seconds
  bool        verbose   = false;
  std::string csvFile   = "results.csv";

  CommandLine cmd;
  cmd.AddValue ("numNodes", "Number of transmitter nodes",              numNodes);
  cmd.AddValue ("dataRate", "Per-node offered data rate (Mbps)",        dataRate);
  cmd.AddValue ("caseB",    "Case B CW settings: CWmin=63, CWmax=127",  caseB);
  cmd.AddValue ("simTime",  "Simulation duration (seconds)",            simTime);
  cmd.AddValue ("verbose",  "Enable verbose WiFi/app logging",          verbose);
  cmd.AddValue ("csvFile",  "Output CSV filename",                      csvFile);
  cmd.Parse (argc, argv);

  if (verbose)
    {
      LogComponentEnable ("OnOffApplication", LOG_LEVEL_INFO);
      LogComponentEnable ("PacketSink",       LOG_LEVEL_INFO);
    }

  // -----------------------------------------------------------------------
  // Contention window settings
  //   Case A: CWmin =   1,  CWmax = 1023   (standard 802.11 DCF defaults)
  //   Case B: CWmin =  63,  CWmax =  127   (narrow, fixed window)
  //
  // Setting these as global defaults before any MAC objects are created is
  // the most reliable method.  If this path does not match your ns-3-dev
  // build (e.g. the class is still called DcaTxop), edit txop.cc to hard-
  // code m_minCw / m_maxCw in the constructor as the lab sheet suggests.
  // -----------------------------------------------------------------------
  uint32_t cwMin = caseB ?  63u :    1u;
  uint32_t cwMax = caseB ? 127u : 1023u;

  // Disable fragmentation and RTS/CTS for small packets so 512-byte
  // application packets go through without any additional overhead
  Config::SetDefault ("ns3::WifiRemoteStationManager::FragmentationThreshold",
                      StringValue ("2200"));
  Config::SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold",
                      StringValue ("2200"));

  // -----------------------------------------------------------------------
  // Nodes
  //   allNodes[0]           = receiver
  //   allNodes[1..numNodes] = transmitters
  // -----------------------------------------------------------------------
  NodeContainer allNodes;
  allNodes.Create (numNodes + 1);

  NodeContainer txNodes;
  for (uint32_t i = 1; i <= numNodes; ++i)
    txNodes.Add (allNodes.Get (i));

  Ptr<Node> rxNode = allNodes.Get (0);

  // -----------------------------------------------------------------------
  // WiFi PHY
  //   Log-distance channel with a short reference distance keeps all nodes
  //   well within range at < 10 m separation.
  // -----------------------------------------------------------------------
  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
                                  "Exponent",           DoubleValue (2.0),
                                  "ReferenceDistance",  DoubleValue (1.0),
                                  "ReferenceLoss",      DoubleValue (46.6777));

  YansWifiPhyHelper phy = YansWifiPhyHelper::Default ();
  phy.SetChannel (wifiChannel.Create ());
  // High TX power ensures every node in the 10x10 m area hears every other
  phy.Set ("TxPowerStart", DoubleValue (20.0));
  phy.Set ("TxPowerEnd",   DoubleValue (20.0));

  // -----------------------------------------------------------------------
  // WiFi MAC
  //   802.11a, fixed 6 Mbps data + control rate, ad-hoc (DCF only, no QoS)
  // -----------------------------------------------------------------------
  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211a);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode",    StringValue ("OfdmRate6Mbps"),
                                "ControlMode", StringValue ("OfdmRate6Mbps"));

  WifiMacHelper mac;
  mac.SetType ("ns3::AdhocWifiMac");

  NetDeviceContainer devices = wifi.Install (phy, mac, allNodes);

  // Set proper cwMin and cwMax
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/Txop/MinCw",
             UintegerValue (cwMin));
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/Txop/MaxCw",
             UintegerValue (cwMax));

  // -----------------------------------------------------------------------
  // Mobility
  //   All nodes placed at random positions within a 10x10 m square and kept
  //   stationary for the entire simulation.
  // -----------------------------------------------------------------------
  MobilityHelper mobility;
  mobility.SetPositionAllocator (
    "ns3::RandomRectanglePositionAllocator",
    "X", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=10.0]"),
    "Y", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=10.0]"));
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (allNodes);

  // -----------------------------------------------------------------------
  // Internet stack and IP addressing
  // -----------------------------------------------------------------------
  InternetStackHelper internet;
  internet.Install (allNodes);

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = ipv4.Assign (devices);

  // -----------------------------------------------------------------------
  // Applications
  //
  // Receiver: PacketSink listening on UDP port 9.
  //
  // Each transmitter: OnOffApplication sending 512-byte UDP datagrams to
  // the receiver at the offered rate R Mbps.  OnTime is set to a very large
  // constant so the application never turns off, saturating the channel.
  // -----------------------------------------------------------------------
  uint16_t sinkPort = 9;
  InetSocketAddress sinkAddress (interfaces.GetAddress (0), sinkPort);

  PacketSinkHelper sinkHelper ("ns3::UdpSocketFactory", sinkAddress);
  ApplicationContainer sinkApp = sinkHelper.Install (rxNode);
  sinkApp.Start (Seconds (0.0));
  sinkApp.Stop  (Seconds (simTime));

  std::ostringstream rateStr;
  rateStr << dataRate << "Mbps";

  OnOffHelper onoff ("ns3::UdpSocketFactory", sinkAddress);
  // 512-byte packets at the specified rate
  onoff.SetConstantRate (DataRate (rateStr.str ()), 512);
  // Never switch off - each node always has packets queued (saturation)
  onoff.SetAttribute ("OnTime",
                      StringValue ("ns3::ConstantRandomVariable[Constant=1e9]"));
  onoff.SetAttribute ("OffTime",
                      StringValue ("ns3::ConstantRandomVariable[Constant=0]"));

  ApplicationContainer txApps = onoff.Install (txNodes);

  // Stagger start times by 1 ms each to avoid a simultaneous first
  // transmission from all nodes (which would cause an initial burst of
  // collisions that may skew steady-state statistics)
  for (uint32_t i = 0; i < txApps.GetN (); ++i)
    {
      txApps.Get (i)->SetStartTime (Seconds (1.0 + i * 0.001));
    }
  txApps.Stop (Seconds (simTime));

  // -----------------------------------------------------------------------
  // Flow Monitor - attached to all nodes
  // -----------------------------------------------------------------------
  FlowMonitorHelper flowHelper;
  Ptr<FlowMonitor> flowMonitor = flowHelper.InstallAll ();

  // -----------------------------------------------------------------------
  // Run
  // -----------------------------------------------------------------------
  Simulator::Stop (Seconds (simTime + 0.5));
  Simulator::Run ();

  // -----------------------------------------------------------------------
  // Collect results from FlowMonitor
  //
  // Only flows destined to the receiver are counted.
  // The first second of simulation is excluded from throughput calculations
  // to allow applications to reach steady state.
  // -----------------------------------------------------------------------
  flowMonitor->CheckForLostPackets ();
  Ptr<Ipv4FlowClassifier> classifier =
    DynamicCast<Ipv4FlowClassifier> (flowHelper.GetClassifier ());

  std::map<FlowId, FlowMonitor::FlowStats> stats = flowMonitor->GetFlowStats ();

  // Measurement window excludes the 1-second ramp-up
  double measureDuration = simTime - 1.0;

  double totalRxBytes    = 0.0;
  uint32_t flowsCounted  = 0;

  for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator it = stats.begin ();
       it != stats.end (); ++it)
    {
      Ipv4FlowClassifier::FiveTuple ft = classifier->FindFlow (it->first);

      // Count only uplink flows to the receiver
      if (ft.destinationAddress == interfaces.GetAddress (0))
        {
          totalRxBytes += it->second.rxBytes;
          ++flowsCounted;
        }
    }

  double totalThroughput   = totalRxBytes * 8.0 / measureDuration / 1.0e6; // Mbps
  double perNodeThroughput = (numNodes > 0) ? (totalThroughput / numNodes)  : 0.0;

  // -----------------------------------------------------------------------
  // Print summary to stdout
  // -----------------------------------------------------------------------
  std::string caseLabel = caseB ? "B" : "A";

  NS_LOG_UNCOND ("[Case " << caseLabel << "] "
                 << "N=" << numNodes
                 << "  R=" << dataRate << " Mbps"
                 << "  flows_counted=" << flowsCounted
                 << "  total_tp=" << totalThroughput << " Mbps"
                 << "  per_node_tp=" << perNodeThroughput << " Mbps");

  // -----------------------------------------------------------------------
  // Append results to CSV
  //   Header is written only if the file does not yet exist.
  // -----------------------------------------------------------------------
  bool fileExists = false;
  {
    std::ifstream probe (csvFile.c_str ());
    fileExists = probe.good ();
  }

  std::ofstream csv (csvFile.c_str (), std::ios::app);
  if (!fileExists)
    {
      csv << "case,numNodes,dataRate_Mbps,"
          << "totalThroughput_Mbps,perNodeThroughput_Mbps\n";
    }
  csv << caseLabel      << ","
      << numNodes        << ","
      << dataRate        << ","
      << totalThroughput << ","
      << perNodeThroughput << "\n";
  csv.close ();

  Simulator::Destroy ();
  return 0;
}
