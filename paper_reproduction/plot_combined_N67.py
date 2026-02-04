"""
Plot combined Fig. 1C and 1D for N=67 data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

import config

# Override N for this plot
N = 67
NP_CROSS = 22


def barycentric_to_cartesian(n_plus, n_minus, n_passive, n_total):
    """Convert barycentric coordinates to Cartesian."""
    f_plus = n_plus / n_total
    f_minus = n_minus / n_total
    f_passive = n_passive / n_total

    h = np.sqrt(3) / 2
    x = f_minus + 0.5 * f_passive
    y = h * f_passive

    return x, y


def plot_combined_N67(simplex_file=None, crosssec_file=None, output_file=None, show=False):
    """
    Create combined figure with Fig. 1C (left) and Fig. 1D (right) for N=67.
    """
    if simplex_file is None:
        simplex_file = os.path.join(config.OUTPUT_DIR, 'data_simplex_q0.0_N67.npz')
    if crosssec_file is None:
        crosssec_file = os.path.join(config.OUTPUT_DIR, 'data_crosssec_N67.npz')
    if output_file is None:
        output_file = os.path.join(config.OUTPUT_DIR, 'fig1cd_combined_N67.png')

    # Set up matplotlib
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['mathtext.fontset'] = 'cm'

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), dpi=config.DPI)

    # ===== FIG 1C (left) =====
    print(f"Loading simplex data from {simplex_file}")
    data = np.load(simplex_file)
    n_plus = data['n_plus']
    n_minus = data['n_minus']
    n_passive = data['n_passive']
    mean_kappa_simplex = data['mean_kappa'] * config.KAPPA_SCALE_FACTOR

    x, y = barycentric_to_cartesian(n_plus, n_minus, n_passive, N)
    h = np.sqrt(3) / 2

    # Triangulation and contour
    triang = tri.Triangulation(x, y)
    levels = 50
    vmin, vmax = mean_kappa_simplex.min(), mean_kappa_simplex.max()
    print(f"Simplex kappa range (scaled): [{vmin:.4f}, {vmax:.4f}]")
    norm = Normalize(vmin=vmin, vmax=vmax)

    tcf = ax1.tricontourf(triang, mean_kappa_simplex, levels=levels, cmap='viridis', norm=norm)

    # Triangle boundary
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, h, 0]
    ax1.plot(triangle_x, triangle_y, 'k-', linewidth=1.5)

    # Dashed line (i) - adjusted for N=67, NP_CROSS=22
    y_line = h * (NP_CROSS / N)
    x_left = y_line / h * 0.5
    x_right = 1 - y_line / h * 0.5
    ax1.plot([x_left, x_right], [y_line, y_line], 'w--', linewidth=1.5, dashes=(5, 3))
    ax1.text(x_left + 0.02, y_line + 0.02, '(i)', color='white', fontsize=9)

    # Colorbar
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.7, aspect=15, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label(r'$\overline{\kappa}_c$', fontsize=11, rotation=0, labelpad=15, y=1.05)
    cbar.set_ticks([round(vmin, 2), round(vmax, 2)])
    cbar.ax.tick_params(labelsize=9)

    # Labels
    ax1.text(-0.08, h/2, r'Generators $\rightarrow$', fontsize=10,
             rotation=60, ha='center', va='center', rotation_mode='anchor')
    ax1.text(1.08, h/2, r'$\leftarrow$ Passive', fontsize=10,
             rotation=-60, ha='center', va='center', rotation_mode='anchor')
    ax1.text(0.5, -0.08, r'Consumers $\rightarrow$', fontsize=10, ha='center', va='top')

    ax1.text(-0.05, h + 0.08, 'C', fontsize=14, fontweight='bold')

    # Add N=67 label
    ax1.text(0.5, h + 0.05, f'N={N}', fontsize=9, ha='center', va='bottom', color='gray')

    ax1.set_xlim(-0.15, 1.2)
    ax1.set_ylim(-0.15, h + 0.15)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # ===== FIG 1D (right) =====
    print(f"Loading cross-section data from {crosssec_file}")
    data = np.load(crosssec_file)
    n_minus_arr = data['n_minus']
    q_values = data['q_values']
    mean_kappa_cross = data['mean_kappa'] * config.KAPPA_SCALE_FACTOR
    std_kappa_cross = data['std_kappa'] * config.KAPPA_SCALE_FACTOR

    print(f"Cross-section n_minus range: [{n_minus_arr.min()}, {n_minus_arr.max()}]")
    print(f"Cross-section kappa range (scaled): [{mean_kappa_cross.min():.4f}, {mean_kappa_cross.max():.4f}]")

    colors = {
        0.0: '#D94040',
        0.1: '#4878A8',
        0.4: '#5AA05A',
        1.0: '#E8A040',
    }

    for qi, q in enumerate(q_values):
        color = colors.get(q, f'C{qi}')
        mean = mean_kappa_cross[qi, :]
        std = std_kappa_cross[qi, :]

        ax2.fill_between(n_minus_arr, mean - std, mean + std, color=color, alpha=0.2)
        ax2.plot(n_minus_arr, mean, color=color, linewidth=2, label=f'$q = {q}$')

    # Determine Y-axis range based on data
    y_data_min = (mean_kappa_cross - std_kappa_cross).min()
    y_data_max = (mean_kappa_cross + std_kappa_cross).max()
    y_min = 0
    y_max = min(0.5, y_data_max * 1.1)  # Cap at 0.5 or slightly above max

    n_active = N - NP_CROSS
    ax2.text(n_active / 2, y_max * 0.95, '(i)', fontsize=10, ha='center', va='top', color='#404040')

    ax2.set_xlabel('Consumers', fontsize=11)
    ax2.set_ylabel(r'$\overline{\kappa}_c$', fontsize=11, rotation=0, labelpad=15)
    ax2.set_xlim(1, n_active - 1)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xticks([1, n_active - 1])
    ax2.set_yticks([0.1, round(y_max, 1)])

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, handlelength=1.5)

    ax2.text(-0.15, 1.05, 'D', fontsize=14, fontweight='bold',
             ha='left', va='bottom', transform=ax2.transAxes)

    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=config.DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved to {output_file}")

    if show:
        plt.show()
    else:
        plt.close()

    return output_file


if __name__ == '__main__':
    import sys
    show = '--show' in sys.argv
    plot_combined_N67(show=show)
