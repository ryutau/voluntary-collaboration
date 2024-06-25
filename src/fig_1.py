import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
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
    visualize_game_analysis()
    validate_cross_point_gamma_values()


def compute_expected_utility(q, action, gamma):
    assert q in range(6)
    assert (gamma >= 0) & (gamma <= 1)
    endowment = 10
    cost_coop = 10
    u_col = 30
    u_ind = 10
    if action == "L":
        return endowment + u_ind
    elif action == "C":
        p_suc = sum([p_k(4, gamma, k) for k in range(q-1, 5)])
        return endowment - cost_coop + p_suc * u_col
    elif action == "D":
        p_suc = sum([p_k(4, gamma, k) for k in range(q, 5)])
        return endowment  + p_suc * u_col
    else:
        raise ValueError("invalid action")


def visualize_game_analysis():
    gamma_ticks = np.arange(0, 1.01, .01)
    color_dict = dict(C="C1", D="C0", L="C2")
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.3), sharex="col")
    plt.subplots_adjust(wspace=0.3, hspace=1.5)
    # B
    for i, ax in enumerate(axes[0]):
        ax.set_ylim(-2, 42)
        ax.set_yticks([10*j for j in range(5)])
        ax.set_xticks([0, .5, 1])
        ax.set_xticklabels([0, .5, 1])
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.set_xlabel("Belief about others (γ)", labelpad=7)
        for orient in ["top", "right", "bottom", "left"]:
            ax.spines[orient].set_visible(False)
        if i == 0:
            ax.set_ylabel("Expected payoff ($E[π]$)", labelpad=12)
    for q, action in product([2, 4, 5], ["D", "C", "L"]):
        ax = axes[0][{2: 0, 4: 1, 5: 2}[q]]
        ax.set_title(f"Threshold: {q}", y=1.2)
        ax.plot(
            gamma_ticks,
            [compute_expected_utility(q, action, gamma) for gamma in gamma_ticks],
            linewidth=3, color=color_dict[action], alpha=.9
        )
        ax.text(1.05, compute_expected_utility(q, action, 1), action, va="center")
    # C
    # To validate the hard-coded gamma values at crossing points, see validate_cross_point_gamma_values.pdf.
    change_points = {
        2: {
            "mandatory": zip([[0, .124], [.124, .414], [.414, 1]], ["C0", "C1", "C0"]),
            "voluntary": zip([[0, .240], [.240, .414], [.414, 1]], ["C2", "C1", "C0"]),
        },
        4: {
            "mandatory": zip([[0, .586], [.586, .876], [.876, 1]], ["C0", "C1", "C0"]),
            "voluntary": zip([[0, .709], [.709, .876], [.876, 1]], ["C2", "C1", "C0"]),
        },
        5: {
            "mandatory": zip([[0, .760], [.760, 1]], ["C0", "C1"]),
            "voluntary": zip([[0, .904], [.904, 1]], ["C2", "C1"]),
        }
    }
    lw = 11
    for q, ax in zip([2, 4, 5], axes[1]):
        for gamma_interval, br_action_color in change_points[q]["mandatory"]:
            # Adjusting intervals a little bit to eliminate the effect of linewidth.
            gamma_interval = [gamma_interval[0]+lw/240, gamma_interval[1]-lw/240]
            ax.plot(gamma_interval, [1, 1], color=br_action_color, linewidth=12)
        for gamma_interval, br_action_color in change_points[q]["voluntary"]:
            gamma_interval = [gamma_interval[0]+lw/240, gamma_interval[1]-lw/240]
            ax.plot(gamma_interval, [0, 0], color=br_action_color, linewidth=lw)
        ax.set_xlabel("Belief about others (γ)", labelpad=7)
        ax.set_ylim(-.5, 2.2)
        ax.set_xlim(-.1, 1.1)
        ax.set_yticks([0, 1])
        for orient in ["top", "right", "bottom", "left"]:
            ax.spines[orient].set_visible(False)
        if q == 2:
            ax.set_yticklabels(labels=["Voluntary", "Mandatory"])
        else:
            ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(f"{save_dir(script_name)}/expected_payoff_per_action.svg", format="svg", dpi=300)

p_group_success = lambda gamma, q: sum([p_k(5, gamma, k) for k in range(q, 6)])
util_coop = lambda gamma, q: 30 * sum([p_k(4, gamma, k) for k in range(q-1, 5)])
util_defect = lambda gamma, q: 10 + 30 * sum([p_k(4, gamma, k) for k in range(q, 5)])


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
    plt.savefig(f"{save_dir(script_name)}/validate_cross_point_gamma_values.svg", format="svg", dpi=300)


if __name__ == "__main__":
    main()
