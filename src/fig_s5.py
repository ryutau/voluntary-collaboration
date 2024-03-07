import pandas as pd
import numpy as np
from math import comb
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
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


def mandatory_solver(*, c, b, n, q):
    # find the gamma where C and D yield the same expected payoff
    gamma_ticks = np.arange(0, 1.001, .001)
    delta = lambda gamma: - b * comb(n-1, q-1) * (gamma ** (q-1)) * ((1-gamma) ** (n-q)) + c
    values = delta(gamma_ticks)
    cp = np.where(np.diff(np.sign(values)))[0] # find where the sign changes
    return ([round(v/1000, 2) for v in cp] + [np.nan, np.nan])[:2]


def voluntary_solver(*, c, b, n, q, r):
    # find the gamma where max(C, D) and L yield the same expected payoff
    gamma_ticks = np.arange(0, 1.001, .001)
    f_delta_coop = lambda gamma: - (- c + b * sum([comb(n-1, k) * (gamma ** (k)) * ((1-gamma) ** (n-k-1)) for k in range(q-1, n)])) + r
    f_delta_defect = lambda gamma: - (b * sum([comb(n-1, k) * (gamma ** (k)) * ((1-gamma) ** (n-k-1)) for k in range(q, n)])) + r
    delta_coop = f_delta_coop(gamma_ticks)
    delta_defect = f_delta_defect(gamma_ticks)
    cp_coop = np.where(np.diff(np.sign(delta_coop)))[0][0]
    if isinstance(delta_defect, (int, float)):
        cp_defect = np.nan
    else:
        cp_defect = np.where(np.diff(np.sign(delta_defect)))[0][0]
    return round(min([cp_coop, cp_defect])/1000, 2)

def compute_change_points(n, r=10):
    res_dict = {}
    for q in range(2, n+1):
        params = dict(
            c=10,
            b=30,
            n=n,
            q=q
        )
        p1, p2 = mandatory_solver(**params)
        params |= dict(r=r)
        p3 = voluntary_solver(**params)
        res_dict[q] = dict(p1=p1, p2=p2, p3=p3)
    res_df = pd.DataFrame(res_dict)
    return res_df

def calculate_p_coops(*, p1, p2, p3, a, b):
    p_m = beta.cdf(p2, a=a, b=b) - beta.cdf(p1, a=a, b=b)
    p_v = (beta.cdf(p2, a=a, b=b) - beta.cdf(p3, a=a, b=b)) / (1 - beta.cdf(p3, a=a, b=b))
    return p_m, p_v

def calculate_p_coop_diff(*, p1, p2, p3, a, b):
    p_m, p_v = calculate_p_coops(p1=p1, p2=p2, p3=p3, a=a, b=b)
    return p_v - p_m


def visualize_betas():
    n = 5
    param_ticks= np.linspace(1, 15, 15)
    colors = ["tab:blue", "white", "tab:orange"]  # Negative to Positive via Zero
    n_bins = 100  # Number of bins
    cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=n_bins)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    # visualize the difference in the cooperation rate
    fig, axes = plt.subplots(5, 3, figsize=(2.2*3, 10.5))
    fig.patch.set_alpha(0)
    for row, r in enumerate([5, 7.5, 10, 12.5, 15]):
        res_df = compute_change_points(n=n, r=r).fillna(1)
        for col, thr in enumerate([2, 4, 5]):
            p1, p2, p3 = res_df[thr]
            sim_result_array = [
                [calculate_p_coop_diff(**dict(p1=p1, p2=p2, p3=p3, a=a, b=b)) for a in param_ticks] for b in param_ticks
            ]
            ax = axes[row][col]
            im = ax.imshow(
                sim_result_array,
                alpha=1,
                norm=norm,
                cmap=cmap
            )
            for i, val_row in enumerate(sim_result_array):
                for j, val in enumerate(val_row):
                    if not np.isnan(val):
                        # Draw a rectangle around the cell
                        rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='grey', lw=0.4)
                        ax.add_patch(rect)

            ax.spines[["top", "right", "left", "bottom"]].set_linewidth(.5)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(r'$b$')
                ax.set_yticks([0, 4, 9, 14])
                ax.set_yticklabels([1, 5, 10, 15])
            if row == 0:
                ax.set_title(f"Threshold: {thr}", y=1.1)
            if row == 4:
                ax.set_xlabel(r'$a$')
                ax.set_xticks([0, 4, 9, 14])
                ax.set_xticklabels([1, 5, 10, 15])
    # Add a colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cbar_ax)            
    fig.text(0.01, 0.17, r"$r=15$", rotation=90, ha="center", va="center")
    fig.text(0.01, 0.33, r"$r=12.5$", rotation=90, ha="center", va="center")
    fig.text(0.01, 0.5, r"$r=10$", rotation=90, ha="center", va="center")
    fig.text(0.01, 0.66, r"$r=7.5$", rotation=90, ha="center", va="center")
    fig.text(0.01, 0.82, r"$r=5$", rotation=90, ha="center", va="center")
    plt.savefig(f"{save_dir(script_name)}/FigS5.pdf", format="pdf", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()
