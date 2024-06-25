import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import math
from pymer4.models import Lmer
from sklearn.utils import resample
from tqdm import tqdm
import sys
sys.path.append("../")

from tools import save_dir


plt.rcParams['image.cmap'] = 'RdYlGn'
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = "Helvetica"
plt.rcParams['hatch.linewidth'] = 1.5
script_name = re.sub(r"\.py$", "", os.path.basename(__file__))


def main():
    exp_data = pd.read_csv("../data/sub_exp_result.csv", index_col=0)
    plot_fig5a(exp_data)
    plot_fig5b(exp_data)
    plot_fig5c(exp_data)
    reg_analysis(exp_data)

def plot_fig5a(exp_data):
    pivot_table = exp_data.pivot_table(index='rho', columns='action', aggfunc='size').fillna(0)
    # make pie chart
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3))
    actions = ["C", "D", "L"]
    for ax, (rho, rho_data) in zip(axes[::-1], pivot_table[actions].iterrows()):
        N = sum(rho_data)
        get_nl = lambda p: "\n" if p > 28 else " "
        ax.pie(
            rho_data, labels=None,
            autopct=lambda x: int(round(N * x / 100, 0)),
            startangle=90, counterclock=False, colors=["C1", "C0", "C2"], wedgeprops={'linewidth': 1, 'edgecolor':"white"},
            textprops={"color": "w", "fontsize": 20}
        )
        ax.set_title(f"$\\rho = {rho:.1f}$", y=.97)
    plt.savefig(f"{save_dir(script_name)}/Fig5A.svg", format="svg", dpi=300, bbox_inches="tight")
    plt.close()

def compute_coop_rate(r_c, r_d, r_l, rho):
    # function to compute cooperation rate given r_c, r_d, r_l as nubers of players choosing c, d, l
    # r_l is counted in the denominator of the cooperation rate only with the rate of rho
    return r_c / (r_c + r_d + r_l * rho)

def nCk(n, k):
    # compute the number of combintations nCk
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

def gamma_nk(n, k, p):
    return nCk(n, k) * (p ** k) * ((1-p) ** (n-k))

def compute_group_success_rate(r_c, r_d, r_l, rho):
    # compute group success rate
    # group success if more than four of the five players choose cooperation
    coop_rate = compute_coop_rate(r_c, r_d, r_l, rho)
    return sum([gamma_nk(5, k, coop_rate) for k in [4, 5]])

def compute_non_reg_stats(rho, rho_data):
    n_c, n_d, n_l = [(rho_data['action'] == action).sum() for action in ['C', 'D', 'L']]
    coop_rate = compute_coop_rate(n_c, n_d, n_l, rho)
    group_success_rate = compute_group_success_rate(n_c, n_d, n_l, rho)
    return coop_rate, group_success_rate

def compute_reg_stats(exp_data):
    model_formula = 'gamma ~ rho + (1|pid)'
    model = Lmer(model_formula, data=exp_data, family='binomial')
    model.fit()
    est = model.coefs.loc['rho', 'Estimate']
    return est

def bootstrap_stats(exp_data, kind, n_bootstrap=1000):
    rho_list = [0, 0.5, 1]
    # compute base stats and prepare the strage for bootstrap
    if kind == 'non_reg':
        base_stats = {rho: compute_non_reg_stats(rho, rho_data) for rho, rho_data in exp_data.groupby('rho')}
        coop_rate_base = {rho: base_stats[rho][0] for rho in rho_list}
        group_success_rate_base = {rho: base_stats[rho][1] for rho in rho_list}
        coop_rate_store = {rho:[] for rho in rho_list}
        group_success_rate_store = {rho:[] for rho in rho_list}
    elif kind == 'reg':
        beta_base = compute_reg_stats(exp_data)
        beta_store = []

    # Bootstrap exp_data using pids, while reassiging a new pid for each sample.
    # For each bootstrap, we compute the focal stats and then we compute their empirical 95% CIs.
    unique_pids = exp_data['pid'].unique()
    for i in tqdm(range(n_bootstrap)):
        bootstrap_pids = resample(unique_pids, random_state=i)
        bootstrap_data = pd.concat([
            exp_data[exp_data['pid'] == pid].assign(pid=j)
            for j, pid in enumerate(bootstrap_pids)
        ])
        if kind == 'non_reg':
            for rho, rho_data in bootstrap_data.groupby('rho'):
                coop_rate, group_success_rate = compute_non_reg_stats(rho, rho_data)
                coop_rate_store[rho].append(coop_rate)
                group_success_rate_store[rho].append(group_success_rate)
        elif kind == 'reg':
            beta = compute_reg_stats(bootstrap_data)
            beta_store.append(beta)
    # compute 95% confidence intervals
    if kind == 'non_reg':
        coop_rate_ci = {rho:np.percentile(coop_rate_store[rho], [2.5, 97.5]) for rho in rho_list}
        group_success_rate_ci = {rho:np.percentile(group_success_rate_store[rho], [2.5, 97.5]) for rho in rho_list}
        return coop_rate_base, group_success_rate_base, coop_rate_ci, group_success_rate_ci
    elif kind == 'reg':
        beta_se = np.std(beta_store)
        beta_ci = np.percentile(beta_store, [2.5, 97.5])
        get_p = lambda x: min((x >= 0).mean(), (x <= 0).mean()) * 2
        p = get_p(np.array(beta_store))
        return beta_base, beta_se, beta_ci, p


def plot_fig5b(exp_data):
    coop_rate_base, group_success_rate_base, coop_rate_ci, group_success_rate_ci = bootstrap_stats(exp_data, kind='non_reg', n_bootstrap=1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 3))
    coop_rate_lower_ci = [coop_rate_base[rho] - coop_rate_ci[rho][0] for rho in coop_rate_base]
    coop_rate_upper_ci = [coop_rate_ci[rho][1] - coop_rate_base[rho] for rho in coop_rate_base]
    group_success_rate_lower_ci = [group_success_rate_base[rho] - group_success_rate_ci[rho][0] for rho in group_success_rate_base]
    group_success_rate_upper_ci = [group_success_rate_ci[rho][1] - group_success_rate_base[rho] for rho in group_success_rate_base]
    ax1.errorbar(coop_rate_base.keys(), coop_rate_base.values(), yerr=[coop_rate_lower_ci, coop_rate_upper_ci], fmt="s", color='C1', ms=8)
    ax2.errorbar(group_success_rate_base.keys(), group_success_rate_base.values(), yerr=[group_success_rate_lower_ci, group_success_rate_upper_ci], fmt="o", color='b', ms=9)

    for ax in [ax1, ax2]:
        ax.set_ylim(-.06, 1.06)
        ax.set_xlim(-.2, 1.2)
        ax.axhline(0, color="black", lw=.5, ls='--', zorder=0)
        ax.axhline(1, color="black", lw=.5, ls='--', zorder=0)
        ax.invert_xaxis()
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel("Loners' externality ($\\rho$)")

    ax1.set_ylabel("Effective cooperation rate")
    ax2.set_ylabel("Group success rate")
    plt.tight_layout()
    plt.subplots_adjust(wspace=.6)
    plt.savefig(f"{save_dir(script_name)}/Fig5B.svg", format="svg", dpi=300)
    plt.close()

def plot_fig5c(exp_data):
    # using exp_data, for each rho, compute the probability of cooperation as a function of belief_c
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3))
    for ax, (rho, rho_data) in zip(axes[::-1], exp_data.groupby('rho')):
        ax.set_title(f"$\\rho={rho}$")
        classes = [round(v, 2) for v in np.linspace(0.05, 0.95, 10)]
        gamma_cut = pd.cut(rho_data.belief_c/(30 - rho_data.belief_l * (1-rho)), bins=np.linspace(0, 1, 11), labels=classes, include_lowest=True)

        ax.plot(classes, rho_data.groupby(gamma_cut).apply(lambda x: (x.action == 'C').mean()), label="C", c="C1", lw=2)
        ax.plot(classes, rho_data.groupby(gamma_cut).apply(lambda x: (x.action == 'D').mean()), label="D", c="C0", lw=2)
        ax.plot(classes, rho_data.groupby(gamma_cut).apply(lambda x: (x.action == 'L').mean()), label="L", c="C2", lw=2)
        ax.set_xlim(0, 1)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        axhist = ax.twinx()
        axhist.bar(classes, gamma_cut.value_counts().reindex(classes).values, color="grey", alpha=.35, width=.05)
        axhist.set_ylim(0, 73)
        axhist.spines[['top', 'right', 'left']].set_visible(False)
        if ax == axes[0]:
            ax.set_ylabel("Probability of\nchoosing action")
            axhist.set_yticks([])
            ax.legend(frameon=False, bbox_to_anchor=(-.4, .5))
        else:
            ax.set_yticks([])
            if ax == axes[2]:
                axhist.set_ylabel("Number of participants")
            else:
                axhist.set_yticks([])

    fig.text(0.51, -.05, "Belief about others' cooperativeness", ha='center', va='center')
    # plt.tight_layout()
    plt.savefig(f"{save_dir(script_name)}/Fig5C.svg", format="svg", dpi=300, bbox_inches="tight")
    plt.close()


def reg_analysis(exp_data):
    beta_base, beta_se, beta_ci, p = bootstrap_stats(exp_data, kind='reg', n_bootstrap=1000)
    stats_series = pd.Series(
        dict(base=beta_base, se=beta_se, z=beta_base/beta_se,
             ci_lower=beta_ci[0], ci_upper=beta_ci[1], two_tailed_p=p))
    stats_series.to_csv(f"{save_dir(script_name)}/reg_res.csv")


if __name__ == "__main__":
    main()
