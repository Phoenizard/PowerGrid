# Prompt: Reproduce Fig. 1C and 1D from Smith et al., Sci. Adv. 8, eabj6734 (2022)

Reproduce **Fig. 1C** and **Fig. 1D** from the paper "The effect of renewable energy incorporation on power grid stability and resilience" (Smith et al., Sci. Adv. 8, eabj6734, 2022). The output must match the paper's figures as closely as possible in data, layout, styling, colors, labels, and typography.

---

## Physics Model

Use the **swing equation** (Eq. 1 in the paper):

```
d²θ_i/dt² + γ·dθ_i/dt = P_i − κ·Σ_j A_ij·sin(θ_i − θ_j)
```

- `θ_i(t)`: phase deviation of node `i`
- `γ = 1`: damping constant
- `κ`: coupling parameter
- `A_ij`: adjacency matrix (unweighted, undirected)
- `P_i`: power at node `i`

### Node power assignment

For a network with `n = 50` nodes, assign `n+` generators, `n−` consumers, `np` passive nodes (`n+ + n− + np = n`):

- Generator: `P_i = +P_max / n+`
- Consumer: `P_i = −P_max / n−`
- Passive: `P_i = 0`

Use `P_max = 1` (all results are normalized by `P_max`).

Node roles are assigned to **random positions** in the network for each realization.

### Computing `κ_c` (critical coupling)

`κ_c` is the minimum `κ` for which Eq. 1 has a stable fixed point (the network can synchronize).

Algorithm:
1. Use **bisection** over `κ` in range `[0.001, 2.0]`, ~15 iterations.
2. For each trial `κ`, integrate the swing equation from small random initial conditions (`θ_i ~ U(-0.01, 0.01)`, `dθ_i/dt = 0`) for `T = 100` time units.
3. Check convergence: if `max(|dθ_i/dt|) < 1e-4` at the end, then a stable fixed point exists → `κ` is above `κ_c`.
4. `κ_c` = smallest `κ` where convergence occurs.
5. Normalize: `κ̄_c = κ_c / P_max`.

### Network model

Use **Watts-Strogatz** networks: `networkx.watts_strogatz_graph(n=50, k=4, p=q)` where `q` is the rewiring probability. `k=4` means mean degree `K̄ = 4`.

---

## Fig. 1C — Ternary Simplex Heatmap

### Data
- `n = 50`, lattice (`q = 0`).
- Sweep `(n+, n−, np)` over the simplex with `n+ + n− + np = 50`, `n+ ≥ 1`, `n− ≥ 1`, `np ≥ 0`. Use step size of 2–3 for tractability.
- For each configuration, average `κ̄_c` over an ensemble of **200** network realizations (reduce to 50 if needed for speed, but note it).

### Visual requirements (must match paper exactly)

1. **Triangle orientation**: Equilateral triangle, vertex at **top**, base at **bottom** (horizontal).
2. **Axis labels**:
   - Left edge (going up-left): **"Generators →"**, rotated ~60°, with arrow pointing toward top vertex.
   - Right edge (going up-right): **"Passive →"**, rotated ~−60° (or ~300°), with arrow pointing toward top vertex.
   - Bottom edge: **"Consumers →"**, horizontal, with arrow pointing right.
3. **Dashed line**: A white horizontal dashed line labeled **(i)** crossing the simplex at `np = 0` level (i.e., the bottom edge region — this is the cross-section plotted in Fig. 1D). The "(i)" label is in white, placed at the left side of the dashed line inside the triangle.
4. **Colormap**: Use **`viridis`** (dark purple at low values → yellow/green at high values). The color range is `[0.36, 0.52]`.
5. **Colorbar**: Vertical colorbar to the **right** of the triangle. Label: `κ̄_c` (with overbar, using LaTeX: `$\overline{\kappa}_c$`). Tick labels at `0.36` and `0.52`.
6. **Fill**: The interior of the triangle is fully filled with interpolated color. The lowest values (dark purple) should appear along the **bottom center** of the simplex. The highest values (yellow) near the **top vertex** and the **lateral edges**.
7. **Panel label**: Bold **"C"** in the top-left corner above the triangle.
8. **Background**: White.
9. **Font**: Use a serif font similar to the paper (e.g., `'Times New Roman'` or `'serif'`). Moderate font size (~10–12 pt for axis labels).
10. **No tick marks or grid lines** on the triangle edges. Only the labels and arrows.

---

## Fig. 1D — Cross-Section Line Plot

### Data
- Cross-section (i): `np = 0`, so `n+ + n− = 50`. Vary `n−` (number of consumers) from `1` to `34` (since `n+ = 50 − n−` and we also need `n+ ≥ 1`; the paper's x-axis goes from 1 to 34, which means `n+ = 49` down to `n+ = 16`, i.e., the cross-section covers roughly `n− = 1` to `n− = 34` with `np = 50 − n+ − n−` held at some small value — **actually looking at the paper, the x-axis "Consumers" goes from 1 to 34, and from the caption this is the np=0 line but with n=50 the max consumers would be 49**. Given the caption says "cross section (i)" which is drawn across the simplex, and the x-axis only goes to 34, this corresponds to a horizontal slice where approximately `np ≈ 16` (i.e., generators vary from ~1 to 34, consumers vary from ~1 to 34, np = 50 − n+ − n− ≈ 16). **Re-reading the figure**: the dashed line (i) is at roughly 1/3 height from the bottom, meaning `np ≈ n/3 ≈ 16`. So the cross-section is: `np = 16`, `n+ = 34 − n−`, `n−` from 1 to 33. The x-axis labeled "Consumers" runs from 1 to 34. Use `np = 16`, `n−` from 1 to 33 (or adjust to get x-range 1 to 34).
- Compute `κ̄_c` for `q = 0.0, 0.1, 0.4, 1.0`, each averaged over **200** realizations.
- Also compute ±1 standard deviation for the shaded bands.

### Visual requirements (must match paper exactly)

1. **Axes**:
   - X-axis: labeled **"Consumers"**, range `[1, 34]`. Tick marks at `1` and `34` only.
   - Y-axis: labeled `$\overline{\kappa}_c$` (with overbar, LaTeX), range `[0.1, 0.5]`. Tick marks at `0.1` and `0.5`. (The paper shows ticks only at 0.1 and 0.5 on the y-axis with an additional thin line at ~0.36.)
2. **Line colors** (must match exactly):
   - `q = 0.0`: **red** (like `#E05050` or `tab:red`)
   - `q = 0.1`: **steel blue / dark blue** (like `#4878A8` or `tab:blue`)
   - `q = 0.4`: **green** (like `#5AAA5A` or `tab:green`)
   - `q = 1.0`: **orange** (like `#E8A040` or `tab:orange`)
3. **Shaded bands**: Each curve has a ±1 SD shaded region in a lighter/transparent version of the same color (`alpha ≈ 0.2–0.3`).
4. **Line style**: Solid lines, moderate thickness (~1.5–2 pt).
5. **Legend**: Placed to the **right of the plot** (outside the axes), vertically stacked. Each entry shows a short colored line segment followed by the label in the format: `$q = 0.0$`, `$q = 0.1$`, `$q = 0.4$`, `$q = 1.0$`. The legend has **no border/frame**.
6. **Label "(i)"**: Placed inside the plot area, near the top-center, in gray or dark text, matching the paper.
7. **Panel label**: Bold **"D"** in the top-left corner above the plot.
8. **Axis style**: Only left and bottom spines visible (no top or right frame lines). This matches the paper's open-frame style.
9. **Font**: Serif font matching Fig. 1C.
10. **The q=0 (red) curve** should be the highest, peaking at ~0.5 at the edges and dipping to ~0.36 in the center. The other curves should be progressively lower, with `q=1.0` (orange) being the lowest at ~0.05–0.1 in the center.

---

## Combined Layout

- Produce **two separate figure files**: `fig1c.png` and `fig1d.png`.
- Also produce a **combined figure** `fig1cd.png` that places C on the left and D on the right, side by side, similar to the paper layout.
- Resolution: **300 DPI**.
- Figure size: For the combined figure, approximately 7 inches wide × 3 inches tall.

---

## Implementation Notes

- Language: **Python 3**
- Libraries: `numpy`, `scipy` (for ODE integration: `solve_ivp`), `matplotlib`, `networkx`, `numba` (for JIT speedup)
- For the ternary plot: either use `python-ternary` (`pip install python-ternary`) or manually compute barycentric-to-Cartesian coordinates and use `matplotlib.tri.Triangulation` with `tricontourf`. The manual approach often gives better visual control.
- Use `matplotlib.rcParams` to set serif fonts globally.
- ODE state vector: `[θ_1, ..., θ_n, ω_1, ..., ω_n]` where `ω_i = dθ_i/dt`.
- Add a **progress indicator** (e.g., tqdm or print statements) since computation may take several minutes.
- If computation is too slow with 200 realizations, reduce to 50 but clearly note this in the output and in the figure title/annotation.
- Save all figures to the project directory.

### Hardware Constraints

Target machine: **Intel i7 + 16 GB RAM + NVIDIA RTX 3050 Ti (4 GB VRAM)**

- **CPU only** — do NOT use GPU/CUDA. This is a CPU-bound ODE problem.
- **Max 6 parallel workers** with `multiprocessing.Pool(6)`, leave 2 cores for OS.
- **Memory < 10 GB** — process simplex configurations in batches, call `gc.collect()` between batches.
- **Numba JIT recommended**: decorate the ODE RHS function with `@numba.njit` for 5–10x speedup on Intel i7. Pre-extract the adjacency matrix as a dense `numpy.ndarray` (not a networkx graph) before passing to the numba function.
- **Windows compatibility**: Always guard multiprocessing with `if __name__ == '__main__':`. Use `mp.set_start_method('spawn')`.
- **Time budget**: ENSEMBLE_SIZE=50 sweep should complete in < 15 min. Full ENSEMBLE_SIZE=200 in < 60 min.
- **Cache results**: Save `.npz` after each computation. Check for cache before recomputing.

---

## Key Physical Insight to Verify

- `κ̄_c` should be **minimized** in the central-bottom region of the simplex (where generators ≈ consumers, few passives).
- `κ̄_c` should be **highest** at the top vertex (all passive) and along the lateral edges (extreme imbalance).
- In Fig. 1D, `κ̄_c` is U-shaped, minimized when `n+ ≈ n−`.
- Higher `q` (more random networks) → lower `κ̄_c` everywhere → easier to synchronize.
