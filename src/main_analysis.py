import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import sys
import os
sys.path.append("../")

from tools import bootstrap_sampling, compute_non_reg_stats, compute_reg_stats

# Load original table containing all relevant variables from the experiments
exp_data = pd.read_csv("../data/exp_result.csv", index_col=0)

# Load original table containing participants' attributes
participant_data = pd.read_csv("../data/participant_attributes.csv", index_col=0)

# Create a modified table for regression analyses
reg_data = exp_data.set_index("p_option").groupby(["pid", "thr"]).apply(
    lambda data: pd.Series(dict(
        M_gamma_c=data.loc["A", "gamma_c"],
        M_is_defect=data.loc["A", "action"] == "D",
        V_is_leave=data.loc["F", "action"] == "L",
        gamma_change=data.loc["F", "gamma_c"] - data.loc["A", "gamma_c"],
        action_change=int(data.loc["F", "is_coop"]) - int(data.loc["A", "is_coop"])
    ))
).reset_index().merge(participant_data, on="pid")

# Prepare a directory to store output files
os.makedirs("../output/", exist_ok=True)


def main():
    n_simulations = 1000
    p = Pool(4)
    with Pool(4) as pool:
        imap = pool.imap_unordered(resample_and_compute_stats, range(n_simulations))
        bootstrap_result = list(tqdm(imap, total=n_simulations))
    for tgt, sort_cols in zip(["non_reg_stats", "reg_stats"], [["seed", "p_option", "thr"], ["seed"]]):
        stats_df = (
            pd.DataFrame(np.ravel([data[tgt] for data in bootstrap_result]).tolist())
            .sort_values(sort_cols)
            .reset_index(drop=True)
        )
        stats_df.to_csv(f"../output/bootstrap_{tgt}.csv")
        aggregate_stats_from_bootstrap(stats_df, tgt)


def aggregate_stats_from_bootstrap(stats_df, tgt):
    def ci95_lower(x):
        return x.quantile(.025)

    def ci95_upper(x):
        return x.quantile(.975)

    agg_funcs = ['mean','std', ci95_lower, ci95_upper]

    if tgt == "non_reg_stats":
        tgt_cols = [
            "p_coop",
            "p_coop_non_loner",
            "p_suc",
            "pop_efficiency",
            "group_efficiency",
            "roc_auc_raw",
            "roc_auc_p_piv"
        ]
        original_stats_df_per_condition = (
            pd.DataFrame(compute_non_reg_stats(exp_data, seed="original"))
            .set_index(["thr", "p_option"])[tgt_cols]
        )
        original_stats_df_per_condition.columns = pd.MultiIndex.from_tuples([(col, 'original') for col in tgt_cols])
        bs_stats_df_per_condition = stats_df.groupby(["thr", "p_option"])[tgt_cols].agg(agg_funcs)

        # computing differences between voluntary and mandatory conditions
        original_gap_df = (
            pd.DataFrame(compute_non_reg_stats(exp_data, seed="original"))
            .groupby("thr")
            [tgt_cols].apply(lambda x: x.diff().iloc[-1])
        )
        original_gap_df.columns = pd.MultiIndex.from_tuples([(col, 'original') for col in tgt_cols])
        bs_gap_df = (
            stats_df.sort_values("p_option")
            .groupby(["thr", "seed"])[tgt_cols]
            .apply(lambda x: x.diff().iloc[-1])
            .reset_index()
            .groupby("thr")[tgt_cols]
            .agg(agg_funcs)
        )
        gap_df = bs_gap_df.join(original_gap_df).sort_index(axis="columns")
        gap_df.to_csv(f"../output/gap_df_{tgt}.csv")

    elif tgt == "reg_stats":
        tgt_cols = [
            'beta1', 'beta2', 'beta3', 'beta4', 'beta_(Intercept)',
            'beta_fs_alpha', 'beta_fs_beta', 'beta_r_from_hl',
            'beta_iri_EC','beta_iri_FS', 'beta_iri_PD', 'beta_iri_PT',
            'beta_siut', 'beta_crt', 'beta_general_trust', 'beta_age',
            'beta_gendermale', 'beta_genderother', 'beta_thr'
        ]
        original_stats_df_per_condition = (
            pd.Series(compute_reg_stats(reg_data, seed=np.nan))[tgt_cols]
            .rename("original")
        )
        bs_stats_df_per_condition = stats_df[tgt_cols].agg(agg_funcs).T

    stats_df_per_condition = (
        bs_stats_df_per_condition.join(original_stats_df_per_condition)
        .sort_index(axis="columns")
    )
    stats_df_per_condition.to_csv(f"../output/stats_df_per_condition_{tgt}.csv")


def resample_and_compute_stats(seed):
    sampled_exp_data = bootstrap_sampling(exp_data, seed)
    sampled_reg_data = bootstrap_sampling(reg_data, seed)
    non_reg_stats = compute_non_reg_stats(sampled_exp_data, seed)
    reg_stats = compute_reg_stats(sampled_reg_data, seed)
    return dict(non_reg_stats=non_reg_stats, reg_stats=reg_stats)


if __name__ == "__main__":
    main()
