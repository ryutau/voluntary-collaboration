import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import os
import re
import math
import sys
sys.path.append("../")

from tools import save_dir, bootstrap_sampling, compute_non_reg_stats

plt.rcParams['image.cmap'] = 'RdYlGn'
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = "Helvetica"
plt.rcParams['hatch.linewidth'] = 1.5
script_name = re.sub(r"\.py$", "", os.path.basename(__file__))

def main():
    exp_data = pd.read_csv("../data/main_exp_result.csv", index_col=0)
    d_l_ratio_dic = {thr: compute_d_l_ratio(thr_data) for thr, thr_data in exp_data.groupby("thr")}
    p_leave_by_action = bootstrap_p_leave_by_action(exp_data)
    p_coop_non_loner = load_bootstrap_p_coop_non_loner(exp_data)
    non_loner_gamma_action_df = make_non_loner_gamma_action_df(exp_data)
    visualize_fig3(d_l_ratio_dic, p_leave_by_action, p_coop_non_loner, non_loner_gamma_action_df)


def bootstrap_p_leave_by_action(exp_data):
    res = []
    exp_data_per_participant = (
        exp_data.set_index("p_option").groupby(["pid", "thr"]).apply(
            lambda x: pd.Series(
                {"is_coop_in_mandatory": x.loc["A", "is_coop"],
                 "is_leave_in_voluntary": x.loc["F", "is_leave"]}
            )
        ).reset_index()
    )
    for seed in tqdm(range(1000)):
        sample_exp_data = bootstrap_sampling(exp_data_per_participant, seed)
        res.append(compute_p_leave_by_action(sample_exp_data, seed))
    bs_result_p_leave = pd.DataFrame(np.ravel(res).tolist())
    original_p_leave = (
        pd.DataFrame(compute_p_leave_by_action(exp_data_per_participant, seed="original"))
        .set_index(["thr", "action_in_m"]).p_leave.rename("original")
    )
    p_leave_by_action = (
        bs_result_p_leave.groupby(["thr", "action_in_m"])["p_leave"]
        .quantile([.025, .975]).unstack().join(original_p_leave)
    )
    return p_leave_by_action


def compute_p_leave_by_action(exp_data_per_participant, seed):
    p_leave_by_action = (
        exp_data_per_participant.groupby(["thr", "is_coop_in_mandatory"])
        ["is_leave_in_voluntary"].mean().reset_index()
    )
    p_leave_by_action.columns = ["thr", "action_in_m", "p_leave"]
    res_dict = p_leave_by_action.to_dict(orient="records")
    res_dict = [v | {"seed": seed} for v in res_dict]
    return res_dict


def load_bootstrap_p_coop_non_loner(exp_data):
    non_reg_stats = pd.read_csv(
        "/Users/moriryuutarou/Desktop/Research/OptionalPGG/Analysis/src/paper_analysis/bootstrap_non_reg_stats.csv",
        index_col=0
    )
    tgt_cols = ["p_coop_non_loner"]
    non_reg_stats_per_condition = (
        non_reg_stats.groupby(["thr", "p_option"])[tgt_cols]
        .agg(['mean', 'std', ("ci95_upper", lambda x: np.quantile(x, q=0.975)),
              ("ci95_lower", lambda x: np.quantile(x, q=0.025))])
    )
    original_non_reg_stats = (
        pd.DataFrame(compute_non_reg_stats(exp_data, seed="original"))
        .set_index(["thr", "p_option"])[tgt_cols]
    )
    new_columns = pd.MultiIndex.from_tuples([(col, 'original') for col in original_non_reg_stats.columns])
    original_non_reg_stats.columns = new_columns
    return non_reg_stats_per_condition.join(original_non_reg_stats).sort_index(axis="columns")


def make_non_loner_gamma_action_df(exp_data):
    non_loner_data = exp_data.groupby(["thr", "pid"]).filter(lambda x: "L" not in x.action.values)
    gamma_df = (
        non_loner_data.pivot_table(index=["pid", "thr"], columns=["p_option"], values="gamma_c")
        .dropna().rename(columns={"A": "gamma_mand", "F": "gamma_vol"})
    )
    action_df = (
        non_loner_data.pivot_table(index=["pid", "thr"], columns=["p_option"], values="is_coop")
        .rename(columns={"A": "is_c_mand", "F": "is_c_vol"}).astype(bool)
    )
    non_loner_gamma_action_df = gamma_df.join(action_df, how="left").reset_index()
    non_loner_gamma_action_df["color"] = non_loner_gamma_action_df.apply(
        lambda x: int(x.is_c_vol - x.is_c_mand) + 1, axis=1
    )
    non_loner_gamma_action_df["order"] = non_loner_gamma_action_df["color"].apply(
        lambda x: {1: 0, 0: 1, 2: 2}[x]
    )
    non_loner_gamma_action_df = non_loner_gamma_action_df.sort_values("order")
    return non_loner_gamma_action_df


def compute_d_l_ratio(thr_data):
    mandatory_data = thr_data[lambda d: d.p_option == 'A'].set_index('pid')
    voluntary_data = thr_data[lambda d: d.p_option == 'F'].set_index('pid')
    bins = [i/10 for i in range(11)]
    cut_by_m_gamma = mandatory_data['gamma_c'].apply(floor_float, n=1)
    data_f = voluntary_data.groupby(cut_by_m_gamma).apply(
        lambda d: pd.Series({'l_ratio': sum(d.action == 'L') / len(d), 'n': len(d)})
    ).reindex(bins).fillna({'n': 0})
    data_m = mandatory_data.groupby(cut_by_m_gamma).apply(
        lambda d: pd.Series({'d_ratio': sum(d.action == 'D') / len(d)})
    ).reindex(bins).fillna({'n': 0})
    return data_m.join(data_f)


def visualize_fig3(d_l_ratio_dic, p_leave_by_action, p_coop_non_loner, non_loner_gamma_action_df):
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    width = .4
    spec = gridspec.GridSpec(nrows=2, ncols=18, figure=fig, hspace=.9, wspace=.3)
    axes1 = [fig.add_subplot(spec[0, i:i+4]) for i in [0, 4, 8]]
    ax2 = fig.add_subplot(spec[0, 14:])
    ax3 = fig.add_subplot(spec[1, :4])
    axes4 = [fig.add_subplot(spec[1, i:i+4]) for i in [6, 10, 14]]

    bins = [i/10 for i in range(11)]

    # Leaving and defection as functions of belief
    for ax, (thr, thr_data) in zip(axes1, d_l_ratio_dic.items()):
        ax.set_title(f'Threshold: {thr}')
        x = range(11)
        ax.plot(x, thr_data["l_ratio"], color='C2', linestyle='--', marker='o', linewidth=.7, label="Leave")
        ax.plot(x, thr_data["d_ratio"], color='C0', linestyle='--', marker='o', linewidth=.7, label="Defect")
        ax.set_xticks([i for i in x if i %5 == 0], [b for i, b in enumerate(bins) if i%5 == 0])
        ax.set_ylim(-.1, 1.2)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        if thr == 2:
            ax.legend(frameon=False)
            ax.set_ylabel(r"Defection/leaving rate")
            ax.set_yticks([0, .5, 1])
        elif thr == 4:
            ax.set_xlabel('Belief about others (γ) under mandatory participation')

    # Leaving in V as a func of action in M
    for i, q in enumerate([2, 4, 5]):
        data = p_leave_by_action.loc[q, :].sort_index()
        ax2.bar(
            [i, i+width], data["original"], width,
            color=["C2", "#b3f399"], alpha=.8, edgecolor=None,
            yerr=[data["original"]-data[0.025], data[0.975]-data["original"]],
            label=["Defector", "Cooperator"],
        )
    handler, label = ax2.get_legend_handles_labels()
    ax2.legend(handler[:2], label[:2], frameon=False, bbox_to_anchor=(.96, 1.2), loc="upper right")
    ax2.set_ylabel('Leaving rate')
    ax2.xaxis.set_ticks_position('none')
    ax2.set_xlabel("Threshold")
    ax2.set_xticks([t + width/2 for t in range(3)], [2, 4, 5])
    ax2.set_ylim(0, 1.1)
    ax2.spines[["top", "right", "left"]].set_visible(False)

    # cooperation rate among non-loners
    for i, q in enumerate([2, 4, 5]):
        data = p_coop_non_loner.loc[q, "p_coop_non_loner"]
        ax3.bar(
            [i, i+width], data["original"], width=.4,
            color=["#FF4747", "#8CE7F8"], alpha=.9, edgecolor=None,
            yerr=[data["original"]-data["ci95_lower"], data["ci95_upper"]-data["original"]],
            label=["A", "F"]
        )
    handler, label = ax3.get_legend_handles_labels()
    ax3.legend(handler[:2], ["Mandatory", "Voluntary"], frameon=False, bbox_to_anchor=(-.04, 1.35), loc="upper left")
    ax3.set_ylabel('Cooperation rate')
    ax3.xaxis.set_ticks_position('none')
    ax3.set_xlabel("Threshold")
    ax3.set_xticks([t + width/2 for t in range(3)], [2, 4, 5])
    ax3.set_ylim(0, 1.15)
    ax3.spines[["top", "right", "left"]].set_visible(False)

    # correlation between changes in action and in beliefs
    for ax, (thr, thr_data) in zip(axes4, non_loner_gamma_action_df.groupby("thr")):
        ax.spines[["top", "right"]].set_visible(False)
        ax.plot([0, 1], [0, 1], "k--", zorder=1)
        cmap = ListedColormap(["C0", "lightgrey", "C1"]) if thr != 5 else ListedColormap(["lightgrey", "C1"])
        scatter = ax.scatter(
            rand_jitter(thr_data["gamma_mand"]), rand_jitter(thr_data["gamma_vol"]),
            c=thr_data["color"].values, marker="o", zorder=2, alpha=.7,
            cmap=cmap
        )
        eps = .05
        ax.set_xlim(-eps, 1+eps)
        ax.set_ylim(-eps, 1+eps)
        ax.set_aspect('equal')
        ax.set_title(f"Threshold: {thr}", pad=13)
        ax.set_yticks([0, .5, 1])
        ax.set_xticks([0, .5, 1])
        if thr == 2:
            ax.set_ylabel("Belief (γ) under \nvoluntary participation")
        elif thr == 4:
            ax.set_xlabel("Belief (γ) under mandatory participation", labelpad=10)
            handler, _ = scatter.legend_elements()
        elif thr == 5:
            ax.legend(
                handler, ["negative", "no change", "positive"],
                fontsize=10, frameon=False, loc="lower left", bbox_to_anchor=(.5, 0)
            )

    plt.savefig(f"{save_dir(script_name)}/Fig3.svg", format="svg", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


def floor_float(x, n):
    new_x = math.floor(x * 10 ** n) / (10 ** n)
    return new_x


def rand_jitter(arr):
    np.random.seed(0)
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


if __name__ == "__main__":
    main()
