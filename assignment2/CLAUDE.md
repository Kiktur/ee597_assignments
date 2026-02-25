# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EE597 Assignment 2: USC Campus Wireless Coverage Simulation. Simulates wireless network coverage on a USC campus map using free-space path loss (Friis), log-normal shadowing, and ray-cast line-of-sight. Includes a Python coverage engine with simulated annealing optimizer, a PyQt5 interactive GUI, and a React/TypeScript web app.

## Commands

### Python
```bash
# Run simulated annealing optimization (default entry point)
python usc_coverage.py

# Launch interactive GUI (requires PyQt5)
python interactive_coverage_tool.py

# Install dependencies
pip install numpy pillow matplotlib pyqt5
pip install numba  # optional, ~10x speedup for coverage computation
```

### Web App (usc-wireless-coverage/)
```bash
cd usc-wireless-coverage
npm install
npm run dev      # Vite dev server
npm run build    # tsc + vite build
npm run lint     # eslint
npm run preview  # preview production build
```

## Architecture

### Python Side
- **`usc_coverage.py`** — Core coverage engine + simulated annealing optimizer. `CoverageCalculator` loads `usc_map_buildings_filled.png`, builds a 1/8 downscaled building mask (300x220), and computes coverage via FSPL + shadowing + 720-ray LOS visibility maps. Hot loops have Numba JIT versions (`_build_visibility_map_numba`, `_compute_coverage_numba`) with pure-Python/NumPy fallbacks. SA optimization via `simulated_annealing()` with spatial-diversity neighbor generation.
- **`interactive_coverage_tool.py`** — PyQt5 GUI wrapping `CoverageCalculator` for click-to-place/drag/right-click-delete BS interaction. Imports from `usc_coverage.py`.

### Web App (`usc-wireless-coverage/src/`)
- **`workers/coverageEngine.ts`** — TypeScript port of the Python coverage engine (should be algorithmically identical). Runs at 1/8 resolution.
- **`workers/coverage.worker.ts`** — Web Worker wrapper; uses request-ID pattern to discard stale results.
- **`workers/prng.ts`** — Mulberry32 + Box-Muller (matches Python `SeededRNG`/`Mulberry32` classes).
- **`workers/bresenham.ts`** — Ray-cast building intersection for LOS.
- **`components/MapCanvas/MapCanvas.tsx`** — Canvas-based map with BS placement interaction and coordinate transforms.
- **`components/Sidebar/`** — Parameter panel, stats panel, instructions panel.
- **`hooks/useCoverageWorker.ts`** — React hook managing worker lifecycle and 150ms debounced recalculation.
- **`App.tsx`** — Top-level state via `useReducer` (ADD_BS, MOVE_BS, DELETE_BS, SET_PARAMS actions).

### Critical Invariant
Python and TypeScript implementations should produce **identical results** for the same inputs. The Mulberry32 PRNG, Box-Muller transform, FSPL calculation, 720-ray visibility maps, and 1/8 downscaling are exact ports. Any algorithm change must be mirrored in both codebases.

**Known discrepancy:** Python `SEED = 40` (usc_coverage.py:56) vs web app hardcoded `SeededRNG(42)` (coverageEngine.ts:135). These must match for cross-implementation parity.

## Key Constants (shared across both codebases)
- Map: 640m x 430m physical, 2400x1760px image, 300x220px compute grid (1/8 scale)
- RF defaults: TX=-10 dBm, Noise=-101 dBm, SNR threshold=10 dB, Shadowing=4 dB, Freq=2.4 GHz
- LOS: 720 rays, COMPUTE_SCALE=8

## Map Asset
`usc_map_buildings_filled.png` — Binary building mask (black=building, white=outdoor). Required by both Python and web app (copied to `usc-wireless-coverage/public/` as `usc_map_buildings_bw.png`).
