import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import re
import os
import sys
sys.path.append("../")

from tools import save_dir


plt.rcParams['image.cmap'] = 'RdYlGn'
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = "Helvetica"
script_name = re.sub(r"\.py$", "", os.path.basename(__file__))


def main():
    visualize_betas()


def visualize_betas():
    n = 8
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # Parameter values to iterate over
    param_values = np.linspace(1, 15, n)
    # Plot beta distributions for combinations of a and b
    for i, a in enumerate(param_values):
        for j, b in enumerate(param_values):
            ax = axes[i, j]
            x = np.linspace(0, 1, 100)
            y = beta.pdf(x, a, b)
            ax.fill_between(x, y, alpha=0.7, color='b')
            ax.set_title(f"Beta($a={a:.0f}, b={b:.0f}$)", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, max(3, ax.get_ylim()[1]))
            ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{save_dir(script_name)}/FigS4.pdf", format="pdf", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()
