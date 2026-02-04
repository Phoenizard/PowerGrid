"""
Plot Fig. 1D - Cross-Section Line Plot
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import config


def plot_fig1d(data_file=None, output_file=None, show=False):
    """
    Create the cross-section line plot for Fig. 1D.
    """
    if data_file is None:
        data_file = os.path.join(config.OUTPUT_DIR, 'data_crosssec.npz')
    if output_file is None:
        output_file = os.path.join(config.OUTPUT_DIR, 'fig1d.png')

    # Load data
    print(f"Loading data from {data_file}")
    data = np.load(data_file)
    n_minus = data['n_minus']
    q_values = data['q_values']
    # Apply calibration scale factor
    mean_kappa = data['mean_kappa'] * config.KAPPA_SCALE_FACTOR
    std_kappa = data['std_kappa'] * config.KAPPA_SCALE_FACTOR

    print(f"q values: {q_values}")
    print(f"n_minus range: [{n_minus.min()}, {n_minus.max()}]")

    # Colors matching paper
    colors = {
        0.0: '#D94040',   # red
        0.1: '#4878A8',   # blue
        0.4: '#5AA05A',   # green
        1.0: '#E8A040',   # orange
    }

    # Set up matplotlib for publication quality
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['mathtext.fontset'] = 'cm'

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3), dpi=config.DPI)

    # Plot each q value
    for qi, q in enumerate(q_values):
        color = colors.get(q, f'C{qi}')
        mean = mean_kappa[qi, :]
        std = std_kappa[qi, :]

        # X-axis is "Consumers" which is n_minus
        x = n_minus

        # Plot shaded band (±1 SD)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

        # Plot line
        ax.plot(x, mean, color=color, linewidth=2, label=f'$q = {q}$')

    # Configure axes
    ax.set_xlabel('Consumers', fontsize=11)
    ax.set_ylabel(r'$\overline{\kappa}_c$', fontsize=11, rotation=0, labelpad=15)

    ax.set_xlim(1, 34)

    # Use paper's Y-axis range: 0 to 0.5, with 0.1 as intermediate tick
    y_min, y_max = 0, 0.5
    print(f"Using Y-axis range: [{y_min:.2f}, {y_max:.2f}]")
    print(f"Data range: q=0 [{mean_kappa[0,:].min():.3f}, {mean_kappa[0,:].max():.3f}]")

    ax.set_ylim(y_min, y_max)
    ax.set_xticks([1, 34])
    ax.set_yticks([0.1, 0.5])  # Paper shows 0.1 as intermediate tick

    # Add "(i)" label
    ax.text(17, 0.47, '(i)', fontsize=10, ha='center', va='top', color='#404040')

    # Open frame style: only left and bottom spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend outside right with line handles (matching paper style)
    leg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                    frameon=False, fontsize=9, handlelength=1.5)

    # Panel label
    ax.text(-0.15, 1.05, 'D', fontsize=14, fontweight='bold',
            ha='left', va='bottom', transform=ax.transAxes)

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
    plot_fig1d(show=show)
