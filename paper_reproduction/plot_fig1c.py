"""
Plot Fig. 1C - Ternary Simplex Heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

import config


def barycentric_to_cartesian(n_plus, n_minus, n_passive, n_total):
    """
    Convert barycentric (simplex) coordinates to Cartesian coordinates.

    Triangle layout (matching paper):
    - Top vertex: all passive (n_passive = n)
    - Bottom-left vertex: all generators (n_plus = n)
    - Bottom-right vertex: all consumers (n_minus = n)

    For an equilateral triangle with unit side:
    - Bottom-left at (0, 0)
    - Bottom-right at (1, 0)
    - Top at (0.5, sqrt(3)/2)
    """
    # Normalize to fractions
    f_plus = n_plus / n_total      # generators - bottom-left
    f_minus = n_minus / n_total    # consumers - bottom-right
    f_passive = n_passive / n_total  # passive - top

    # Cartesian coordinates
    # x = f_minus + 0.5 * f_passive
    # y = (sqrt(3)/2) * f_passive
    h = np.sqrt(3) / 2

    x = f_minus + 0.5 * f_passive
    y = h * f_passive

    return x, y


def plot_fig1c(data_file=None, output_file=None, show=False):
    """
    Create the ternary heatmap for Fig. 1C.
    """
    if data_file is None:
        data_file = os.path.join(config.OUTPUT_DIR, 'data_simplex_q0.0.npz')
    if output_file is None:
        output_file = os.path.join(config.OUTPUT_DIR, 'fig1c.png')

    # Load data
    print(f"Loading data from {data_file}")
    data = np.load(data_file)
    n_plus = data['n_plus']
    n_minus = data['n_minus']
    n_passive = data['n_passive']
    mean_kappa = data['mean_kappa'] * config.KAPPA_SCALE_FACTOR  # Apply calibration

    print(f"Data points: {len(mean_kappa)}")
    print(f"kappa range: [{mean_kappa.min():.4f}, {mean_kappa.max():.4f}]")

    # Convert to Cartesian
    x, y = barycentric_to_cartesian(n_plus, n_minus, n_passive, config.N)

    # Set up matplotlib for publication quality
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['mathtext.fontset'] = 'cm'

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=config.DPI)

    # Create triangulation
    triang = tri.Triangulation(x, y)

    # Mask triangles outside the simplex (if any)
    # For simplex data, all points should be valid

    # Plot filled contours - use actual data range to avoid white areas
    levels = 50
    vmin, vmax = mean_kappa.min(), mean_kappa.max()
    print(f"Using colormap range: [{vmin:.4f}, {vmax:.4f}]")
    norm = Normalize(vmin=vmin, vmax=vmax)

    tcf = ax.tricontourf(triang, mean_kappa, levels=levels, cmap='viridis', norm=norm)

    # Draw triangle boundary
    h = np.sqrt(3) / 2
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, h, 0]
    ax.plot(triangle_x, triangle_y, 'k-', linewidth=1.5)

    # Add dashed line (i) - horizontal line at np ≈ 16 (about 1/3 from bottom)
    # np/n = 16/50 = 0.32, so y = h * 0.32 ≈ 0.277
    y_line = h * (config.NP_CROSS / config.N)
    # Line extends from left edge to right edge at this height
    # At y = y_line, the left edge has x = y_line / (2*h) * 1 = y_line / h * 0.5
    # and right edge has x = 1 - y_line / h * 0.5
    x_left = y_line / h * 0.5
    x_right = 1 - y_line / h * 0.5
    ax.plot([x_left, x_right], [y_line, y_line], 'w--', linewidth=1.5, dashes=(5, 3))

    # Add "(i)" label
    ax.text(x_left + 0.02, y_line + 0.02, '(i)', color='white', fontsize=9,
            fontweight='normal', ha='left', va='bottom')

    # Add colorbar - use ScalarMappable for continuous gradient
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=15, pad=0.02)
    cbar.outline.set_visible(False)  # Remove colorbar border
    cbar.set_label(r'$\overline{\kappa}_c$', fontsize=11, rotation=0, labelpad=15, y=1.05)
    cbar.set_ticks([round(vmin, 2), round(vmax, 2)])  # Use actual data range
    cbar.ax.tick_params(labelsize=9)

    # Add axis labels with arrows
    # Left edge: "Generators →" rotated +60°
    ax.text(-0.08, h/2, r'Generators $\rightarrow$', fontsize=10,
            rotation=60, ha='center', va='center', rotation_mode='anchor')

    # Right edge: "Passive →" rotated -60°
    ax.text(1.08, h/2, r'$\leftarrow$ Passive', fontsize=10,
            rotation=-60, ha='center', va='center', rotation_mode='anchor')

    # Bottom edge: "Consumers →"
    ax.text(0.5, -0.08, r'Consumers $\rightarrow$', fontsize=10,
            ha='center', va='top')

    # Panel label
    ax.text(-0.05, h + 0.08, 'C', fontsize=14, fontweight='bold',
            ha='left', va='bottom', transform=ax.transData)

    # Clean up axes
    ax.set_xlim(-0.15, 1.2)
    ax.set_ylim(-0.15, h + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')

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
    plot_fig1c(show=show)
