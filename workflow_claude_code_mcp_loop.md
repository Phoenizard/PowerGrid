# Self-Correcting Paper Figure Reproduction Workflow
# Claude Code + MCP Visual Feedback Loop

---

## Overview

This document defines a **fully automated, iterative workflow** for reproducing Fig. 1C and 1D from Smith et al., Sci. Adv. 8, eabj6734 (2022). The workflow uses:

- **Claude Code**: writes and executes Python simulation code on the local terminal
- **MCP (Model Context Protocol)**: connects to the local machine, views output images, compares with the reference figure, and provides structured feedback
- **Loop**: repeats until the reproduction is pixel-accurate to the paper

---

## Hardware Specification

**All code must be written and optimized for the following machine. Never exceed these limits.**

| Component | Spec |
|-----------|------|
| **CPU** | Intel Core i7 (specific gen unknown, assume ~8 threads) |
| **RAM** | 16 GB DDR |
| **GPU** | NVIDIA GeForce RTX 3050 Ti, **4 GB VRAM** |
| **OS** | Windows (assume WSL or native Python) |

### Computation Constraints Derived from Hardware

1. **Memory budget**: Keep peak Python process memory **< 10 GB** (leave headroom for OS + browser). Each ODE integration of n=50 nodes is tiny (~1 KB state), but 200 realizations × hundreds of simplex points adds up. **Always process sequentially and release memory between configurations.**

2. **CPU parallelism**: Use `multiprocessing.Pool(workers=6)` maximum (leave 2 threads for OS). For the simplex sweep this is the main bottleneck — parallelize across ensemble realizations, not across simplex points.

3. **GPU**: This simulation is CPU-bound (ODE integration via SciPy). **Do NOT use CUDA/GPU.** Do not install PyTorch/TensorFlow/CuPy for this task. The GPU is irrelevant here.

4. **Time budget per iteration**:
   - ENSEMBLE_SIZE=50, STEP_SIZE=3: target **< 15 min** for full simplex sweep
   - ENSEMBLE_SIZE=50, cross-section only: target **< 5 min**
   - ENSEMBLE_SIZE=200 (final run): allow up to **60 min**
   - If any single run exceeds these, reduce parameters and note it

5. **ODE solver optimization for this CPU**:
   - Use `scipy.integrate.solve_ivp` with `method='RK45'` (fastest for this problem)
   - Set `max_step=1.0`, `rtol=1e-6`, `atol=1e-8`
   - Pre-compile the RHS with a closure (avoid dict lookups in hot loop)
   - Consider using `scipy.integrate.odeint` (Fortran LSODA) if RK45 is too slow — it is often 2-3x faster for stiff-ish problems on Intel CPUs
   - **Numba JIT** (`@numba.njit`) on the ODE RHS function is allowed and recommended — it can give 5-10x speedup on Intel i7. Install with `pip install numba`.

6. **Disk**: Cache all computation results to `.npz` files. Between iterations, **only recompute what changed** — if only plot styling changed, skip the simulation entirely and re-plot from cache.

7. **Estimated total workflow time**: ~2-3 hours including all iterations. Plan accordingly.

### config.py Hardware-Aware Defaults

```python
# === Hardware-aware settings for i7 + 16GB RAM ===
import os
N_WORKERS = min(6, os.cpu_count() - 2)  # leave 2 cores free

# Fast iteration mode (used for loops 1-7)
ENSEMBLE_SIZE_FAST = 50
STEP_SIZE_FAST = 3
T_INTEGRATE_FAST = 80

# Final production mode (used for last run)
ENSEMBLE_SIZE_FINAL = 200
STEP_SIZE_FINAL = 2
T_INTEGRATE_FINAL = 100

# Memory guard: process simplex points in batches
BATCH_SIZE = 20  # number of simplex configs to hold in memory at once
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Claude)                 │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │  PHASE 1  │───▶│  PHASE 2  │───▶│     PHASE 3      │   │
│  │  Write &  │    │  View &   │    │  Diagnose &      │   │
│  │  Execute  │    │  Compare  │    │  Patch            │   │
│  │  (Claude  │    │  (MCP     │    │  (Claude Code)    │   │
│  │   Code)   │    │  Terminal)│    │                    │   │
│  └──────────┘    └──────────┘    └──────────────────┘   │
│       ▲                                    │             │
│       │            LOOP until               │             │
│       │            PASS ✓                   │             │
│       └────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘
```

---

## Complete Prompt (paste this into Claude)

```
You are tasked with reproducing Fig. 1C and Fig. 1D from the paper "The effect of renewable energy incorporation on power grid stability and resilience" (Smith et al., Sci. Adv. 8, eabj6734, 2022).

You have two capabilities:
1. **Claude Code** — write and run Python code on my local machine via terminal
2. **MCP image viewing** — view the output PNG files and the reference figure to compare them

You must follow this iterative workflow until the figures match the paper exactly.

==========================================================
WORKFLOW: ITERATIVE FIGURE REPRODUCTION
==========================================================

## PHASE 0: SETUP (run once)

1. Create a project directory: `~/paper_reproduction/`
2. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib networkx tqdm
   ```
3. Save the reference figure (from the PDF, page 2) as `~/paper_reproduction/reference_fig1.png`
4. Create a `config.py` with all tunable parameters (so patches are easy):
   ```python
   # config.py — tuned for Intel i7 / 16GB RAM / RTX 3050 Ti 4GB
   import os

   # === Physics ===
   N = 50              # network size
   K = 4               # mean degree (Watts-Strogatz k parameter)
   P_MAX = 1.0         # normalized max power
   GAMMA = 1.0         # damping coefficient
   KAPPA_RANGE = (0.001, 2.0)  # bisection search range
   BISECTION_STEPS = 15
   CONV_TOL = 1e-4     # convergence tolerance for steady state
   Q_VALUES = [0.0, 0.1, 0.4, 1.0]  # Watts-Strogatz rewiring params

   # === Hardware-aware computation ===
   N_WORKERS = min(6, os.cpu_count() - 2)  # parallel workers (leave 2 cores for OS)
   USE_NUMBA = True     # JIT-compile ODE RHS for ~5-10x speedup on i7

   # Fast iteration mode (loops 1–7, target <15 min per sweep)
   ENSEMBLE_SIZE = 50
   STEP_SIZE = 3        # simplex sampling granularity
   T_INTEGRATE = 80     # ODE integration time
   ODE_METHOD = 'RK45'  # or 'LSODA' if RK45 is slow
   ODE_MAX_STEP = 1.0

   # Final production mode (last run, allow ~60 min)
   ENSEMBLE_SIZE_FINAL = 200
   STEP_SIZE_FINAL = 2
   T_INTEGRATE_FINAL = 100

   # Memory management (16GB total, keep process <10GB)
   BATCH_SIZE = 20      # simplex configs processed per batch before gc

   # === Output ===
   DPI = 300
   OUTPUT_DIR = 'output'
   ```

## PHASE 1: WRITE & EXECUTE (Claude Code)

Write the complete simulation in modular files:

### File structure:
```
~/paper_reproduction/
├── config.py          # all parameters
├── model.py           # swing equation, kappa_c computation
├── sweep.py           # simplex sweep, cross-section sweep
├── plot_fig1c.py      # ternary heatmap
├── plot_fig1d.py      # cross-section line plot
├── plot_combined.py   # side-by-side layout
├── run_all.sh         # master script
├── reference_fig1.png # extracted from PDF
└── output/
    ├── fig1c.png
    ├── fig1d.png
    ├── fig1cd_combined.png
    ├── data_simplex.npz    # cached computation results
    └── data_crosssec.npz   # cached computation results
```

### model.py requirements:

```python
"""
Core physics:
- swing_equation(t, state, P, kappa, A, gamma): ODE right-hand side
- compute_kappa_c(A, P, config): bisection to find critical coupling
- generate_network(n, k, q): Watts-Strogatz graph
- assign_power(n, n_plus, n_minus, P_max): random role assignment
"""
```

### Execution:
```bash
cd ~/paper_reproduction
python sweep.py        # heavy computation, saves .npz
python plot_fig1c.py   # generates fig1c.png
python plot_fig1d.py   # generates fig1d.png
python plot_combined.py # generates combined figure
```

After each execution, print the file paths and sizes of all output PNGs.

## PHASE 2: VIEW & COMPARE (MCP)

After code execution completes, use MCP to:

1. **View the output figure**: open `~/paper_reproduction/output/fig1c.png` (or fig1d.png)
2. **View the reference figure**: open `~/paper_reproduction/reference_fig1.png`
3. **Perform a structured comparison** using this checklist:

### Comparison Checklist for Fig. 1C:

| Item | Criterion | Status |
|------|-----------|--------|
| C-1 | Triangle orientation: vertex at top, base at bottom | ☐ |
| C-2 | Colormap: viridis, range [0.36, 0.52] | ☐ |
| C-3 | Color pattern: dark purple at bottom-center, yellow at top and edges | ☐ |
| C-4 | Colorbar: vertical, right side, labeled κ̄_c with ticks 0.36 and 0.52 | ☐ |
| C-5 | Axis labels: "Generators →" (left, rotated), "Passive →" (right, rotated), "Consumers →" (bottom) | ☐ |
| C-6 | White dashed line (i) with label, positioned at correct height | ☐ |
| C-7 | Panel label "C" bold, top-left | ☐ |
| C-8 | Serif font throughout | ☐ |
| C-9 | Clean white background, no extra grid/ticks | ☐ |
| C-10 | Data correctness: minimum κ̄_c ≈ 0.36 at bottom-center | ☐ |

### Comparison Checklist for Fig. 1D:

| Item | Criterion | Status |
|------|-----------|--------|
| D-1 | X-axis: "Consumers", range [1, 34], ticks at 1 and 34 only | ☐ |
| D-2 | Y-axis: κ̄_c (overbar), range [0.1, 0.5], ticks at 0.1 and 0.5 | ☐ |
| D-3 | Line colors: red (q=0), blue (q=0.1), green (q=0.4), orange (q=1.0) | ☐ |
| D-4 | Shaded ±1 SD bands, alpha ≈ 0.2 | ☐ |
| D-5 | U-shaped curves: minimum at center, peaks at edges | ☐ |
| D-6 | q=0 highest (~0.36 center, ~0.5 edges), q=1.0 lowest (~0.06 center) | ☐ |
| D-7 | Legend: right side, no frame, format "q = X.X" | ☐ |
| D-8 | "(i)" label inside plot, top-center area | ☐ |
| D-9 | Open frame: only left and bottom spines | ☐ |
| D-10 | Panel label "D" bold, top-left | ☐ |
| D-11 | Serif font throughout | ☐ |

### Scoring:
- **PASS** (all ✓): proceed to final output
- **PARTIAL** (≥7 ✓): proceed to Phase 3 targeted fixes
- **FAIL** (<7 ✓): proceed to Phase 3 major revision

## PHASE 3: DIAGNOSE & PATCH (Claude Code)

Based on the comparison results, generate **targeted patches**:

### Diagnosis format:
```
FAILED ITEMS: [C-3, D-6, D-9]

C-3: Color pattern wrong — minimum appears at top instead of bottom.
     ROOT CAUSE: simplex coordinate mapping is inverted.
     FIX: swap the generator/consumer axes in barycentric transform.

D-6: q=0 curve peaks too low (~0.3 instead of ~0.5).
     ROOT CAUSE: bisection range too narrow, or ensemble too small.
     FIX: widen KAPPA_RANGE to (0.001, 3.0), increase ensemble to 100.

D-9: All four spines visible.
     FIX: add ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
```

### Patching rules:
1. **Never rewrite from scratch** unless >50% items fail
2. Use **targeted edits** to specific files
3. **Cache expensive computations** — only recompute if model/config changes
4. **Only re-plot** if the issue is visual
5. After patching, return to **PHASE 1** (execute) → **PHASE 2** (compare)

## PHASE 4: FINAL VALIDATION (MCP)

When all checklist items pass:

1. Generate final high-resolution figures (DPI=300)
2. Place reference and reproduction **side-by-side** in a single comparison image
3. View this comparison image via MCP for final sign-off
4. If approved, copy final outputs to a designated location

==========================================================
END OF WORKFLOW
==========================================================

## IMPORTANT RULES:

1. **Maximum iterations**: 8. If not converged after 8 loops, output the best version with a report of remaining discrepancies.

2. **Hardware**: The machine is Intel i7 + 16GB RAM + RTX 3050 Ti 4GB. 
   - **CPU only** — do not use GPU/CUDA for this task.
   - **Max 6 parallel workers** (leave 2 cores for OS).
   - **Keep memory < 10 GB** — process in batches, release between configs.
   - **Use Numba JIT** on the ODE RHS for ~5-10x speedup on i7.
   - **On Windows**: always use `if __name__ == '__main__':` guard for multiprocessing.

3. **Computation budget**: Start with ENSEMBLE_SIZE=50 for fast iteration (target <15 min per sweep). Only increase to 200 for the final run after all visual elements pass (allow ~60 min).

4. **Caching**: Always save intermediate computation results to .npz files. Only recompute when the physics model or parameters change, not when only visual styling changes. Before any computation, check if cache exists and load it.

5. **Logging**: Maintain a `iteration_log.md` file that records each iteration:
   ```
   ## Iteration 3
   - Time: 12 min (sweep) + 3 sec (plot)
   - Memory peak: 4.2 GB
   - Changes: Fixed colormap range, added dashed line
   - Checklist: C-1✓ C-2✓ C-3✓ C-4✗ C-5✓ ...
   - Failed: C-4 (colorbar ticks missing)
   - Next action: Fix colorbar ticks in plot_fig1c.py [plot-only, no recompute]
   ```

6. **Error handling**: If code crashes, read the traceback, fix the bug, and re-execute. Do not count bug-fix iterations toward the 8-iteration limit.

7. **Progress reporting**: After each phase, briefly report status to me. I don't need to approve each iteration — run autonomously until done or stuck.

8. **Performance monitoring**: At the start of each computation run, print estimated time and memory. If a run is clearly going to exceed the budget, abort early, reduce parameters, and retry.

Now begin. Start with Phase 0, then Phase 1, then Phase 2, and loop.
```

---

## Physics Model Specification

(Attach this as a separate reference file or include below the main prompt)

```
==========================================================
PHYSICS REFERENCE: Swing Equation Model
==========================================================

Equation 1 (swing equation):
  d²θ_i/dt² + γ·dθ_i/dt = P_i − κ·Σ_j A_ij·sin(θ_i − θ_j)

Parameters:
  γ = 1 (damping)
  κ = coupling strength (variable)
  A_ij = adjacency matrix (Watts-Strogatz, n=50, K̄=4)
  P_i = power at node i

Node types:
  Generator (n+ nodes): P_i = +P_max / n+
  Consumer  (n− nodes): P_i = −P_max / n−
  Passive   (np nodes): P_i = 0
  Constraint: n+ + n− + np = n = 50

Critical coupling κ_c:
  The minimum κ for which a stable fixed point of Eq.1 exists.
  Found via bisection: integrate from small random IC for T=100,
  check if max|dθ/dt| < 1e-4 at end.
  Normalize: κ̄_c = κ_c / P_max

Fig. 1C: Ternary simplex heatmap
  - Sweep (n+, n−, np) with step~3, q=0 (lattice)
  - Each point: mean κ̄_c over 200 realizations
  - Colormap: viridis [0.36, 0.52]

Fig. 1D: Cross-section line plot
  - Cross-section at np≈16 (dashed line (i) in Fig.1C)
  - n− from 1 to 33, n+ = 34 − n−, np = 16
  - X-axis "Consumers" [1, 34]
  - Curves for q = 0.0, 0.1, 0.4, 1.0
  - Each point: mean ± 1 SD over 200 realizations
  - Y-axis κ̄_c [0.1, 0.5]
```

---

## Visual Specification (pixel-level detail)

```
==========================================================
VISUAL REFERENCE: Exact Styling
==========================================================

FONTS:
  - Family: serif (Times New Roman or similar)
  - Axis labels: 10pt
  - Tick labels: 9pt
  - Panel labels (C, D): 12pt bold

FIG 1C — TERNARY HEATMAP:
  - Triangle: equilateral, vertex=top, base=bottom
  - Left edge label: "Generators →" rotated +60°
  - Right edge label: "Passive →" rotated −60°  
  - Bottom edge label: "Consumers →" horizontal
  - Arrows (→) integrated into labels
  - Interior: filled tricontourf, viridis, levels=50+
  - Colorbar: vertical, right side
    - Label: $\overline{\kappa}_c$ at top
    - Ticks: 0.36 (bottom), 0.52 (top)
  - Dashed line: white, linewidth=1.5, dashes=(5,3)
    - Label "(i)" in white, left of line, inside triangle
  - No triangle edge ticks, no gridlines
  - Panel label "C" upper-left, bold

FIG 1D — LINE PLOT:
  - Box: open frame (left+bottom spines only)
  - X-axis: "Consumers", [1, 34], ticks=[1, 34]
  - Y-axis: "$\overline{\kappa}_c$", [0.1, 0.5], ticks=[0.1, 0.5]  
    (Note: paper may show additional subtle gridline at 0.36)
  - Lines (solid, lw=2):
    q=0.0 → red    (#D94040 or tab:red)
    q=0.1 → blue   (#4878A8 or tab:blue)
    q=0.4 → green  (#5AA05A or tab:green)
    q=1.0 → orange (#E8A040 or tab:orange)
  - Shaded bands: same color, alpha=0.25
  - Legend: outside right, no frame, vertical stack
    Format: "$q = 0.0$" with colored line swatch
  - "(i)" text: inside plot, near top-center, dark gray
  - Panel label "D" upper-left, bold

COMBINED LAYOUT:
  - Side by side: C left, D right
  - Figure size: 7" × 3"
  - Tight layout, minimal padding
  - DPI: 300
```

---

## Troubleshooting Guide

Include this at the end of the prompt so Claude Code can self-diagnose common issues:

```
==========================================================
TROUBLESHOOTING: Common Issues & Fixes
==========================================================

ISSUE: κ̄_c values all near 0 or all near max
  → Check power assignment: must have both P>0 and P<0 nodes
  → Check A_ij is correct (symmetric, unweighted)
  → Ensure bisection bounds bracket the transition

ISSUE: Simplex plot is blank or uniform color
  → Check barycentric coordinate mapping
  → Verify data_simplex.npz contains varying values
  → Check tricontourf masking (points outside triangle)

ISSUE: Computation takes >30 min
  → Reduce ENSEMBLE_SIZE to 30 for iteration
  → Reduce T_INTEGRATE to 50
  → Use vectorized ODE if possible
  → Parallelize with multiprocessing.Pool(workers=6) [i7 has ~8 threads]
  → Enable Numba JIT on the ODE RHS function (@numba.njit)
  → Use scipy.integrate.odeint (Fortran LSODA) instead of solve_ivp — often 2-3x faster on Intel

ISSUE: Memory usage exceeds 10 GB or system starts swapping
  → Process simplex points in batches of BATCH_SIZE=20, call gc.collect() between batches
  → Do NOT hold all 200 ODE solutions in memory simultaneously
  → Reduce ENSEMBLE_SIZE temporarily
  → Monitor with: print(f"RSS: {psutil.Process().memory_info().rss / 1e9:.1f} GB")

ISSUE: Numba compilation fails
  → Ensure ODE RHS uses only numpy arrays and basic math (no scipy, no networkx inside)
  → Pre-extract adjacency matrix as dense numpy array before passing to numba function
  → If numba is problematic, fall back to pure numpy — still faster than naive Python

ISSUE: multiprocessing crashes on Windows
  → Wrap main code in if __name__ == '__main__': guard
  → Use 'spawn' start method: mp.set_start_method('spawn')
  → If still fails, fall back to sequential with progress bar

ISSUE: ODE integration diverges
  → Add max_step=0.5 to solve_ivp
  → Try method='RK45' or 'LSODA'
  → Check that kappa is not negative

ISSUE: Ternary labels overlap or mispositioned
  → Adjust label positions manually with transform offsets
  → Use ax.text() instead of ax.set_xlabel()

ISSUE: Colors don't match paper
  → Use exact hex codes from spec
  → Verify colormap range with vmin/vmax
  → Check alpha values for shaded bands

ISSUE: Cross-section shape is wrong (not U-shaped)
  → Verify the cross-section coordinates match the simplex
  → Check that np is held constant (not n+)
  → Ensure n+ = 34 - n_, not n+ = 50 - n_
```
