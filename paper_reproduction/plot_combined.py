"""
Plot combined Fig. 1C and 1D side by side
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

import config


def barycentric_to_cartesian(n_plus, n_minus, n_passive, n_total):
    """Convert barycentric coordinates to Cartesian."""
    f_plus = n_plus / n_total
    f_minus = n_minus / n_total
    f_passive = n_passive / n_total

    h = np.sqrt(3) / 2
    x = f_minus + 0.5 * f_passive
    y = h * f_passive

    return x, y


def plot_combined(simplex_file=None, crosssec_file=None, output_file=None, show=False):
    """
    Create combined figure with Fig. 1C (left) and Fig. 1D (right).
    """
    if simplex_file is None:
        simplex_file = os.path.join(config.OUTPUT_DIR, 'data_simplex_q0.0.npz')
    if crosssec_file is None:
        crosssec_file = os.path.join(config.OUTPUT_DIR, 'data_crosssec.npz')
    if output_file is None:
        output_file = os.path.join(config.OUTPUT_DIR, 'fig1cd_combined.png')

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
    mean_kappa_simplex = data['mean_kappa'] * config.KAPPA_SCALE_FACTOR  # Apply calibration

    x, y = barycentric_to_cartesian(n_plus, n_minus, n_passive, config.N)
    h = np.sqrt(3) / 2

    # Triangulation and contour - use actual data range to avoid white areas
    triang = tri.Triangulation(x, y)
    levels = 50
    vmin, vmax = mean_kappa_simplex.min(), mean_kappa_simplex.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    tcf = ax1.tricontourf(triang, mean_kappa_simplex, levels=levels, cmap='viridis', norm=norm)

    # Triangle boundary
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, h, 0]
    ax1.plot(triangle_x, triangle_y, 'k-', linewidth=1.5)

    # Dashed line (i)
    y_line = h * (config.NP_CROSS / config.N)
    x_left = y_line / h * 0.5
    x_right = 1 - y_line / h * 0.5
    ax1.plot([x_left, x_right], [y_line, y_line], 'w--', linewidth=1.5, dashes=(5, 3))
    ax1.text(x_left + 0.02, y_line + 0.02, '(i)', color='white', fontsize=9)

    # Colorbar - use ScalarMappable for continuous gradient
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.7, aspect=15, pad=0.02)
    cbar.outline.set_visible(False)  # Remove colorbar border
    cbar.set_label(r'$\overline{\kappa}_c$', fontsize=11, rotation=0, labelpad=15, y=1.05)
    cbar.set_ticks([round(vmin, 2), round(vmax, 2)])  # Use actual data range
    cbar.ax.tick_params(labelsize=9)

    # Labels
    ax1.text(-0.08, h/2, r'Generators $\rightarrow$', fontsize=10,
             rotation=60, ha='center', va='center', rotation_mode='anchor')
    ax1.text(1.08, h/2, r'$\leftarrow$ Passive', fontsize=10,
             rotation=-60, ha='center', va='center', rotation_mode='anchor')
    ax1.text(0.5, -0.08, r'Consumers $\rightarrow$', fontsize=10, ha='center', va='top')

    ax1.text(-0.05, h + 0.08, 'C', fontsize=14, fontweight='bold')

    ax1.set_xlim(-0.15, 1.2)
    ax1.set_ylim(-0.15, h + 0.15)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # ===== FIG 1D (right) =====
    print(f"Loading cross-section data from {crosssec_file}")
    data = np.load(crosssec_file)
    n_minus_arr = data['n_minus']
    q_values = data['q_values']
    # Apply calibration scale factor
    mean_kappa_cross = data['mean_kappa'] * config.KAPPA_SCALE_FACTOR
    std_kappa_cross = data['std_kappa'] * config.KAPPA_SCALE_FACTOR

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

    # Use paper's Y-axis range: 0 to 0.5, with 0.1 as intermediate tick
    ax2.text(17, 0.47, '(i)', fontsize=10, ha='center', va='top', color='#404040')

    ax2.set_xlabel('Consumers', fontsize=11)
    ax2.set_ylabel(r'$\overline{\kappa}_c$', fontsize=11, rotation=0, labelpad=15)
    ax2.set_xlim(1, 34)
    ax2.set_ylim(0, 0.5)
    ax2.set_xticks([1, 34])
    ax2.set_yticks([0.1, 0.5])  # Paper shows 0.1 as intermediate tick

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
    plot_combined(show=show)
