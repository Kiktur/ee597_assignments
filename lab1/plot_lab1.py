#!/usr/bin/env python3
"""
EE597 Spring 2026 - Lab 1 Plotting Script
Reads results.csv and generates all required submission plots.

Usage:
    python3 plot_lab1.py                        # uses results.csv in cwd
    python3 plot_lab1.py --csv /path/to/results.csv
    python3 plot_lab1.py --outdir ./figures

Output (saved to --outdir, default: ./figures):
    caseA_E1_total_throughput.png
    caseA_E1_pernode_throughput.png
    caseA_E2_total_throughput.png
    caseA_E2_pernode_throughput.png
    caseB_E1_total_throughput.png
    caseB_E1_pernode_throughput.png
    caseB_E2_total_throughput.png
    caseB_E2_pernode_throughput.png
    combined_E1_total.png          (A vs B overlay, useful for discussion)
    combined_E2_total.png          (A vs B overlay, useful for discussion)
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser (description="Plot EE597 Lab 1 results")
parser.add_argument ("--csv",    default="results.csv",
                     help="Path to results CSV (default: results.csv)")
parser.add_argument ("--outdir", default="figures",
                     help="Directory to write figures (default: ./figures)")
args = parser.parse_args ()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if not os.path.exists (args.csv):
    sys.exit (f"ERROR: CSV file not found: {args.csv}")

df = pd.read_csv (args.csv)
df.columns = df.columns.str.strip ()

required = {"case", "numNodes", "dataRate_Mbps",
            "totalThroughput_Mbps", "perNodeThroughput_Mbps"}
missing  = required - set (df.columns)
if missing:
    sys.exit (f"ERROR: CSV is missing columns: {missing}")

os.makedirs (args.outdir, exist_ok=True)

# Separate by case
A = df [df ["case"] == "A"].copy ()
B = df [df ["case"] == "B"].copy ()

# The fixed values used in each experiment (inferred from data)
# E1: multiple N values at a single R  ->  identify by rows where numNodes varies
# E2: multiple R values at a single N  ->  identify by rows where dataRate varies

def split_E1_E2 (case_df):
    """
    E1 rows: dataRate is (nearly) constant, numNodes varies.
    E2 rows: numNodes is constant (20), dataRate varies.
    We identify them by the fixed_nodes=20 convention used in Lab1Run.sh.
    """
    e2 = case_df [case_df ["numNodes"] == 20].copy ()
    # E1 may also include N=20 at the fixed rate; distinguish by whether
    # multiple dataRate values exist for that N
    fixed_rate = case_df ["dataRate_Mbps"].mode () [0]
    e1 = case_df [case_df ["dataRate_Mbps"] == fixed_rate].copy ()
    return e1.sort_values ("numNodes"), e2.sort_values ("dataRate_Mbps")

A_e1, A_e2 = split_E1_E2 (A)
B_e1, B_e2 = split_E1_E2 (B)

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
STYLE = {
    "A": {"color": "#1f77b4", "marker": "o", "label": "Case A"},
    "B": {"color": "#d62728", "marker": "s", "label": "Case B"},
}

def save_fig (fig, name):
    path = os.path.join (args.outdir, name)
    fig.savefig (path, dpi=150, bbox_inches="tight")
    print (f"  Saved: {path}")
    plt.close (fig)

def plot_single (x, y, xlabel, ylabel, title, case_key, filename):
    fig, ax = plt.subplots (figsize=(7, 4.5))
    s = STYLE [case_key]
    ax.plot (x, y, color=s ["color"], marker=s ["marker"],
             linewidth=2, markersize=6, label=s ["label"])
    ax.set_xlabel (xlabel, fontsize=12)
    ax.set_ylabel (ylabel, fontsize=12)
    ax.set_title  (title,  fontsize=13)
    ax.legend     (fontsize=11)
    ax.grid       (True, linestyle="--", alpha=0.5)
    ax.xaxis.set_minor_locator (ticker.AutoMinorLocator ())
    ax.yaxis.set_minor_locator (ticker.AutoMinorLocator ())
    save_fig (fig, filename)

def plot_overlay (x_a, y_a, x_b, y_b, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots (figsize=(7, 4.5))
    for x, y, key in [(x_a, y_a, "A"), (x_b, y_b, "B")]:
        s = STYLE [key]
        ax.plot (x, y, color=s ["color"], marker=s ["marker"],
                 linewidth=2, markersize=6, label=s ["label"])
    ax.set_xlabel (xlabel, fontsize=12)
    ax.set_ylabel (ylabel, fontsize=12)
    ax.set_title  (title,  fontsize=13)
    ax.legend     (fontsize=11)
    ax.grid       (True, linestyle="--", alpha=0.5)
    ax.xaxis.set_minor_locator (ticker.AutoMinorLocator ())
    ax.yaxis.set_minor_locator (ticker.AutoMinorLocator ())
    save_fig (fig, filename)

# ---------------------------------------------------------------------------
# Individual required plots (8 total, 10 pts each in rubric)
# ---------------------------------------------------------------------------
print ("\nGenerating individual plots...")

# Case A - E1
plot_single (A_e1 ["numNodes"], A_e1 ["totalThroughput_Mbps"],
             "Number of Nodes (N)", "Aggregate Throughput (Mbps)",
             "Case A - E1: Aggregate Throughput vs. N",
             "A", "caseA_E1_total_throughput.png")

plot_single (A_e1 ["numNodes"], A_e1 ["perNodeThroughput_Mbps"],
             "Number of Nodes (N)", "Per-Node Throughput (Mbps)",
             "Case A - E1: Per-Node Throughput vs. N",
             "A", "caseA_E1_pernode_throughput.png")

# Case A - E2
plot_single (A_e2 ["dataRate_Mbps"], A_e2 ["totalThroughput_Mbps"],
             "Offered Data Rate per Node (Mbps)", "Aggregate Throughput (Mbps)",
             "Case A - E2: Aggregate Throughput vs. Offered Rate",
             "A", "caseA_E2_total_throughput.png")

plot_single (A_e2 ["dataRate_Mbps"], A_e2 ["perNodeThroughput_Mbps"],
             "Offered Data Rate per Node (Mbps)", "Per-Node Throughput (Mbps)",
             "Case A - E2: Per-Node Throughput vs. Offered Rate",
             "A", "caseA_E2_pernode_throughput.png")

# Case B - E1
plot_single (B_e1 ["numNodes"], B_e1 ["totalThroughput_Mbps"],
             "Number of Nodes (N)", "Aggregate Throughput (Mbps)",
             "Case B - E1: Aggregate Throughput vs. N",
             "B", "caseB_E1_total_throughput.png")

plot_single (B_e1 ["numNodes"], B_e1 ["perNodeThroughput_Mbps"],
             "Number of Nodes (N)", "Per-Node Throughput (Mbps)",
             "Case B - E1: Per-Node Throughput vs. N",
             "B", "caseB_E1_pernode_throughput.png")

# Case B - E2
plot_single (B_e2 ["dataRate_Mbps"], B_e2 ["totalThroughput_Mbps"],
             "Offered Data Rate per Node (Mbps)", "Aggregate Throughput (Mbps)",
             "Case B - E2: Aggregate Throughput vs. Offered Rate",
             "B", "caseB_E2_total_throughput.png")

plot_single (B_e2 ["dataRate_Mbps"], B_e2 ["perNodeThroughput_Mbps"],
             "Offered Data Rate per Node (Mbps)", "Per-Node Throughput (Mbps)",
             "Case B - E2: Per-Node Throughput vs. Offered Rate",
             "B", "caseB_E2_pernode_throughput.png")

# ---------------------------------------------------------------------------
# Overlay / comparison plots (useful for the discussion section)
# ---------------------------------------------------------------------------
print ("\nGenerating comparison overlay plots...")

plot_overlay (A_e1 ["numNodes"],      A_e1 ["totalThroughput_Mbps"],
              B_e1 ["numNodes"],      B_e1 ["totalThroughput_Mbps"],
              "Number of Nodes (N)", "Aggregate Throughput (Mbps)",
              "E1: Aggregate Throughput vs. N  (Case A vs. B)",
              "combined_E1_total.png")

plot_overlay (A_e2 ["dataRate_Mbps"], A_e2 ["totalThroughput_Mbps"],
              B_e2 ["dataRate_Mbps"], B_e2 ["totalThroughput_Mbps"],
              "Offered Data Rate per Node (Mbps)", "Aggregate Throughput (Mbps)",
              "E2: Aggregate Throughput vs. Offered Rate  (Case A vs. B)",
              "combined_E2_total.png")

# ---------------------------------------------------------------------------
# Print summary table to terminal
# ---------------------------------------------------------------------------
print ("\n" + "=" * 60)
print ("SUMMARY TABLE")
print ("=" * 60)
for label, e1, e2 in [("A", A_e1, A_e2), ("B", B_e1, B_e2)]:
    print (f"\n--- Case {label} ---")
    print ("\nE1 (vary N):")
    print (e1 [["numNodes", "dataRate_Mbps",
                "totalThroughput_Mbps", "perNodeThroughput_Mbps"]].to_string (index=False))
    print ("\nE2 (vary R, N=20):")
    print (e2 [["numNodes", "dataRate_Mbps",
                "totalThroughput_Mbps", "perNodeThroughput_Mbps"]].to_string (index=False))

print (f"\nAll figures saved to: {os.path.abspath (args.outdir)}/")
