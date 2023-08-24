import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from itertools import product
from scipy.stats import beta
import re
import os
import sys
sys.path.append("../")
from tools import save_dir, p_k


plt.rcParams['image.cmap'] = 'RdYlGn'
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = "Helvetica"
script_name = re.sub(r"\.py$", "", os.path.basename(__file__))

def main():
    visualize_p_coop()
    validate_cross_point_gamma_values()


def visualize_p_coop():
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 5))
    fig.subplots_adjust(hspace=.4, wspace=.2)
    min_max_list = []
    tick_max = 15
    ticks = [i / 2 for i in range(1, tick_max*2+1)]
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors=["w", "#ff7c0a"], N=2048)
    for ax_i, (q, kind) in enumerate(product([2, 4, 5], ["mandatory", "voluntary"])):
        row = ax_i % 2
        col = ax_i // 2
        ax = axes[row][col]
        ax.spines[["top", "right", "left", "bottom"]].set_linewidth(.5)
        sim_result_array = [[compute_p_coop(a, b, q=q, kind=kind) for a in ticks] for b in ticks]
        M = np.nanmax(np.ravel(sim_result_array))
        m = np.nanmin(np.ravel(sim_result_array))
        min_max_list.append({"q": q, "kind": kind, "min": m, "max": M})
        im = ax.imshow(
            sim_result_array,
            vmin=0,
            vmax=1,
            alpha=1,
            cmap=custom_cmap
        )
        ax.set_yticks([2] + [i for i in range(tick_max*2) if i % 10 == 9])
        ax.set_yticklabels([1] + [int(t) for i, t in enumerate(ticks) if i % 10 == 9], fontsize=14)
        ax.set_xticks([2] + [i for i in range(tick_max*2) if i % 10 == 9])
        ax.set_xticklabels([1] + [int(t) for i, t in enumerate(ticks) if i % 10 == 9], fontsize=14)
        ax.invert_yaxis()
        if row == 0:
            ax.set_title(f"Threshold: {q}", y=1.06, fontsize=16)
        elif row == 1:
            ax.set_xlabel(r"$a$")
        if col == 0:
            ax.set_ylabel(r"$b$")
    fig.text(0, 0.3, "Voluntary", rotation=90, ha="center", va="center", fontsize=16)
    fig.text(0, 0.73, "Mandatory", rotation=90, ha="center", va="center", fontsize=16)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f"{save_dir(script_name)}/Fig_S1.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()


def compute_p_coop(a, b, q, kind):
    # To validate the hard-coded gamma values at crossing points, see validate_cross_point_gamma_values.pdf.
    t1, t2 = {
        2: {"mandatory": (0.124, 0.414), "voluntary": (0.240, 0.414)},
        4: {"mandatory": (0.586, 0.876), "voluntary": (0.709, 0.876)},
        5: {"mandatory": (0.760, 1), "voluntary": (0.904, 1)},
    }[q][kind]
    if kind == "voluntary":
        p_l = beta.cdf(t1, a, b)
        p_d = 1 - beta.cdf(t2, a, b)
        p_c = 1 - (p_l + p_d)
    if kind == "mandatory":
        p_c = beta.cdf(t2, a, b) - beta.cdf(t1, a, b)
        p_d = 1 - p_c
    gamma = p_c / (p_c + p_d) if p_c + p_d > 0 else np.nan
    return gamma


def validate_cross_point_gamma_values():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    gamma_ticks = np.linspace(0, 1, 10001)
    for q, ax in zip([2, 4, 5], axes):
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        utils_given_defect = [util_defect(gamma, q) for gamma in gamma_ticks]
        utils_given_coop = [util_coop(gamma, q) for gamma in gamma_ticks]
        gap_c_d = pd.Series(
            {gamma: abs(u_c - u_d)
             for gamma, u_c, u_d in zip(gamma_ticks, utils_given_coop, utils_given_defect)}
        ).sort_values()
        gap_c_l = pd.Series(
            {gamma: abs(u_c - 20) for gamma, u_c in zip(gamma_ticks, utils_given_coop)}
        ).sort_values()
        ax.plot(gamma_ticks, utils_given_defect, linewidth=3, label="D")
        ax.plot(gamma_ticks, utils_given_coop, linewidth=3, label="C")
        ax.plot([0, 1], [20, 20], "g-", linewidth=3, label="L")
        ax.set_ylim(-6, 42)
        ax.set_title(f"Threshold: {q}")
        ax.set_xlabel("γ")
        cross_points = {2: [.124, .240, .414], 4: [.586, .709, .876], 5: [.760, .904, 1]}[q]
        for g in cross_points:
            ax.plot([g, g], [0, util_coop(g, q)], "k--")
            ax.text(g, -4, g, ha="center", fontsize=10)
        if q == 2:
            ax.set_yticks([i*10 for i in range(5)])
            ax.set_ylabel("Expected utility")
            ax.legend(frameon=False, loc="lower right")
        else:
            ax.set_yticks([])
            ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"{save_dir(script_name)}/validate_cross_point_gamma_values.pdf", format="pdf", dpi=300)


p_group_success = lambda gamma, q: sum([p_k(5, gamma, k) for k in range(q, 6)])
util_coop = lambda gamma, q: 30 * sum([p_k(4, gamma, k) for k in range(q-1, 5)])
util_defect = lambda gamma, q: 10 + 30 * sum([p_k(4, gamma, k) for k in range(q, 5)])


if __name__ == "__main__":
    main()
