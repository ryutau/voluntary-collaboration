import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys
sys.path.append("../")

from tools import save_dir, compute_non_reg_stats

plt.rcParams['image.cmap'] = 'RdYlGn'
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = "Helvetica"
plt.rcParams['hatch.linewidth'] = 1.5
script_name = re.sub(r"\.py$", "", os.path.basename(__file__))

def main():
    exp_data = pd.read_csv("../data/main_exp_result.csv", index_col=0)
    bootstrap_result = load_bootstrap_result(exp_data)
    plot_pies(exp_data)
    visualize_BCD(bootstrap_result)


def load_bootstrap_result(original_exp_data):
    non_reg_stats = pd.read_csv("../output/bootstrap_non_reg_stats.csv", index_col=0)
    tgt_cols = ["p_coop", "p_suc", "pop_efficiency", "group_efficiency"]
    non_reg_stats_per_condition = (
        non_reg_stats.groupby(["thr", "p_option"])[tgt_cols]
        .agg(['mean', 'std', ("ci95_upper", lambda x: np.quantile(x, q=0.975)),
              ("ci95_lower", lambda x: np.quantile(x, q=0.025))])
    )
    original_non_reg_stats = (
        pd.DataFrame(compute_non_reg_stats(original_exp_data, seed="original"))
        .set_index(["thr", "p_option"])[tgt_cols]
    )
    new_columns = pd.MultiIndex.from_tuples([(col, 'original') for col in original_non_reg_stats.columns])
    original_non_reg_stats.columns = new_columns
    return non_reg_stats_per_condition.join(original_non_reg_stats).sort_index(axis="columns")


def plot_pies(exp_data):
    N = exp_data['pid'].nunique()
    for (thr, p_option), tgt_data in exp_data.groupby(["thr", "p_option"]):
        plt.figure(figsize=(3, 3))
        options = ["C", "D"] if p_option == "A" else ["C", "D", "L"]
        cnt_actions = tgt_data.action.value_counts().reindex(options, fill_value=0)
        plt.pie(
            cnt_actions, startangle=90, labels=None,
            autopct=lambda x: int(round(N*x/100, 0)),
            counterclock=False, colors=["C1", "C0", "C2"],
            textprops={'fontsize': 30, "color": "w"},
            wedgeprops={'linewidth': 2, 'edgecolor':"white"}
        )
        plt.axis('equal')
        plt.savefig(
            f"{save_dir(script_name)}/Fig2A_pie_{thr}_{p_option}.pdf",
            format="pdf", dpi=300, bbox_inches="tight", transparent=True
        )
        plt.close()


def visualize_BCD(bootstrap_result):
    m_color = "#FF4747"
    v_color = "#8CE7F8"
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 3.2),
        gridspec_kw={'width_ratios': [1, 1, 1.5]}
    )
    fig.subplots_adjust(wspace=.5)

    # B
    ax = axes[0]
    width = 0.4
    for i, q in enumerate([2, 4, 5]):
        data = bootstrap_result["p_coop"].unstack().loc[q, :]
        labels = ["Mandatory", "Voluntary"] if i == 0 else None
        bars = ax.bar(
            [i, i+width], data["original"], width,
            yerr=[data["original"]-data["ci95_lower"], data["ci95_upper"]-data["original"]],
            color=[m_color, v_color], edgecolor=None, error_kw={"lw": 1}, label=labels
        )
    ax.set_ylabel('Cooperation rate')
    ax.xaxis.set_ticks_position('none')
    ax.set_xlabel("Threshold")
    ax.set_xticks([t + width/2 for t in range(3)], [2, 4, 5])
    ax.set_ylim(0, 1.27)
    ax.set_yticks([i/4 for i in range(5)])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, bbox_to_anchor=(0, 1.05), loc="upper left", fontsize=10)

    # C
    ax = axes[1]
    width = 0.4
    for i, q in enumerate([2, 4, 5]):
        data = bootstrap_result["p_suc"].unstack().loc[q, :]
        labels = ["Mandatory", "Voluntary"] if i == 0 else None
        ax.bar(
            [i, i+width], data["original"], width,
            yerr=[data["original"]-data["ci95_lower"], data["ci95_upper"]-data["original"]],
            color=[m_color, v_color], edgecolor=None, error_kw={"lw": 1}, label=labels
        )
    ax.set_ylabel('Group success rate')
    ax.xaxis.set_ticks_position('none')
    ax.set_xlabel("Threshold")
    ax.set_xticks([t + width/2 for t in range(3)], [2, 4, 5])
    ax.set_ylim(0, 1.27)
    ax.set_yticks([i/4 for i in range(5)])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, bbox_to_anchor=(0, 1.05), loc="upper left", fontsize=10)

    # D
    ax = axes[2]
    width = 0.27
    for i, q in enumerate([2, 4, 5]):
        data = bootstrap_result["group_efficiency"].unstack().loc[q, :]
        l1 = ["Mandatory", "Voluntary (without loners)"] if i == 0 else None
        ax.bar(
            [i, i+width], data["original"], width,
            yerr=[data["original"]-data["ci95_lower"], data["ci95_upper"]-data["original"]],
            color=[m_color, v_color], edgecolor="w", hatch=["", ""], error_kw={"lw": 1}, linewidth=1,
            label=l1
        )
        data = bootstrap_result["pop_efficiency"].loc[(q, "F")]
        l2 = "Voluntary (including loners)" if i == 0 else None
        ax.bar(
            i+width*2, data["original"], width,
            yerr=[[data["original"]-data["ci95_lower"]], [data["ci95_upper"]-data["original"]]],
            color=v_color, edgecolor="w", hatch="\\", error_kw={"lw": 1}, linewidth=1, label=l2
        )
    ax.set_ylabel('Normalized average payoff')
    ax.legend(frameon=False, bbox_to_anchor=(0, 1.07), loc="upper left", fontsize=10)
    ax.set_xticks([t + width for t in range(3)], [2, 4, 5])
    ax.set_ylim(0, 1.27)
    ax.set_yticks([i/4 for i in range(5)])
    ax.xaxis.set_ticks_position('none')
    ax.set_xlabel("Threshold")
    ax.spines[["top", "right", "left"]].set_visible(False)

    plt.savefig(f"{save_dir(script_name)}/Fig2_BCD.pdf", format="pdf", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()

if __name__ == "__main__":
    main()
