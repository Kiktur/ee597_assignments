#!/usr/bin/env bash
# =============================================================================
# Lab1Run.sh - EE597 Spring 2026, Lab 1
#
# Drives all four simulation configurations:
#   Case A E1: vary N,  fix R = 10 Mbps
#   Case A E2: fix N = 20, vary R
#   Case B E1: vary N,  fix R = 10 Mbps
#   Case B E2: fix N = 20, vary R
#
# Usage (from ns-3-dev root directory):
#   bash Lab1Run.sh
#
# Results are appended to results.csv in the ns-3-dev root directory.
# =============================================================================

set -e

# Where this script lives (should be ns-3-dev root)
NS3_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$NS3_ROOT"

SIM_TIME=15           # seconds per run - increase for lower variance
FIXED_RATE=10         # Mbps used in E1 experiments
FIXED_NODES=20        # node count used in E2 experiments

# N values for E1 (vary number of nodes)
N_VALUES=(1 5 10 15 20 30)
# N_VALUES=(1 5)

# R values for E2 (vary offered data rate, in Mbps)
R_VALUES=(1 5 10 20 30 40 54)
# R_VALUES=(1 5)

# Fresh output file each run
rm -f results.csv

echo "============================================================"
echo " EE597 Lab 1 - Simulation Suite"
echo " SIM_TIME = ${SIM_TIME}s per run"
echo "============================================================"

# -------------------------------------------------------------------------
# Case A, Experiment 1 - Vary N
# -------------------------------------------------------------------------
echo ""
echo "[Case A | E1] Varying number of nodes (R = ${FIXED_RATE} Mbps)"

for N in "${N_VALUES[@]}"; do
  echo "  -> N = $N"
  ./waf --run "lab1 --numNodes=${N} --dataRate=${FIXED_RATE} \
               --caseB=false --simTime=${SIM_TIME}" 2>/dev/null
done
# ./waf --run "lab1 --numNodes=5 --dataRate=10 --caseB=false --simTime=5"
# -------------------------------------------------------------------------
# Case A, Experiment 2 - Vary R
# -------------------------------------------------------------------------
echo ""
echo "[Case A | E2] Varying data rate (N = ${FIXED_NODES} nodes)"
for R in "${R_VALUES[@]}"; do
  echo "  -> R = $R Mbps"
  ./waf --run "lab1 --numNodes=${FIXED_NODES} --dataRate=${R} \
               --caseB=false --simTime=${SIM_TIME}" 2>/dev/null
done

# -------------------------------------------------------------------------
# Case B, Experiment 1 - Vary N
# -------------------------------------------------------------------------

echo ""
echo "[Case B | E1] Varying number of nodes (R = ${FIXED_RATE} Mbps)"
for N in "${N_VALUES[@]}"; do
  echo "  -> N = $N"
  ./waf --run "lab1 --numNodes=${N} --dataRate=${FIXED_RATE} \
               --caseB=true --simTime=${SIM_TIME}" 2>/dev/null
done

# -------------------------------------------------------------------------
# Case B, Experiment 2 - Vary R
# -------------------------------------------------------------------------
echo ""
echo "[Case B | E2] Varying data rate (N = ${FIXED_NODES} nodes)"
for R in "${R_VALUES[@]}"; do
  echo "  -> R = $R Mbps"
  ./waf --run "lab1 --numNodes=${FIXED_NODES} --dataRate=${R} \
               --caseB=true --simTime=${SIM_TIME}" 2>/dev/null
done

echo ""
echo "============================================================"
echo " All runs complete.  Results written to: results.csv"
echo "============================================================"
