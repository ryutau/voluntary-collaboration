import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import sys
sys.path.append("../")

from tools import bootstrap_sampling, compute_non_reg_stats, compute_reg_stats

exp_data = pd.read_csv("../data/exp_result.csv", index_col=0)
participant_df = pd.read_csv("../data/participant_attributes.csv", index_col=0)
original_reg_data = exp_data.set_index("p_option").groupby(["pid", "thr"]).apply(
    lambda data: pd.Series(dict(
        M_gamma_c=data.loc["A", "gamma_c"],
        M_is_defect=data.loc["A", "action"] == "D",
        V_is_leave=data.loc["F", "action"] == "L",
        gamma_change=data.loc["F", "gamma_c"] - data.loc["A", "gamma_c"],
        action_change=int(data.loc["F", "is_coop"]) - int(data.loc["A", "is_coop"])
    ))
).reset_index().merge(participant_df, on="pid")

def main():
    n_simulations = 1000
    p = Pool(4)
    with Pool(4) as pool:
        imap = pool.imap_unordered(resample_and_compute_stats, range(n_simulations))
        bootstrap_result = list(tqdm(imap, total=n_simulations))
    for tgt in ["non_reg_stats", "reg_stats"]:
        stats_df = pd.DataFrame(np.ravel([data[tgt] for data in bootstrap_result]).tolist())
        stats_df.to_csv(f"../output/bootstrap_{tgt}.csv")


def resample_and_compute_stats(seed):
    sampled_exp_data = bootstrap_sampling(exp_data, seed)
    sampled_reg_data = bootstrap_sampling(original_reg_data, seed)
    non_reg_stats = compute_non_reg_stats(sampled_exp_data, seed)
    reg_stats = compute_reg_stats(sampled_reg_data, seed)
    return dict(non_reg_stats=non_reg_stats, reg_stats=reg_stats)


if __name__ == "__main__":
    main()
