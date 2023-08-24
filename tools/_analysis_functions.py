import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pymer4.models import Lmer
from ._convenient_functions import p_k


def compute_non_reg_stats(exp_data, seed):
    result_df = exp_data.pivot_table(
        index=['p_option', "thr"],
        columns=["action"],
        aggfunc="size",
        fill_value=0
    ).reset_index()
    # Individual cooperation rate
    result_df["p_coop"] = result_df["C"] / (result_df["C"] + result_df["D"])
    # Group success rate
    result_df["p_suc"] = result_df.apply(
        lambda x: sum([p_k(5, x.p_coop, k) for k in range(x.thr, 6)]),
        axis=1
    )
    # Efficiency including loners
    result_df["pop_efficiency"] = result_df.apply(compute_pop_efficiency, axis=1)
    # Efficiency without loners
    result_df["group_efficiency"] = result_df.apply(compute_group_efficiency, axis=1)
    p_coop_non_loner = (
        exp_data.groupby(["pid", "thr"]).filter(lambda x: "L" not in x.action.values)
        .groupby(["p_option", "thr"])["is_coop"].mean()
    ).rename("p_coop_non_loner").reset_index()
    # Roc-auc of raw expectation and pivotal probability in predicting the decision to cooperate.
    roc_auc_data = (
        exp_data[lambda x: ~ x.is_leave].dropna(subset=["gamma_c"])
        .groupby(["p_option", "thr"]).apply(
            lambda x: pd.Series({
                "roc_auc_raw": wrap_roc_auc_score(x.is_coop, x.gamma_c),
                "roc_auc_p_piv": wrap_roc_auc_score(x.is_coop, p_k(4, x.gamma_c, x.thr-1))
            })
        ).reset_index()
    )
    # Merging the results.
    result_df = (
        result_df.merge(p_coop_non_loner, on=["p_option", "thr"])
        .merge(roc_auc_data, on=["p_option", "thr"])
    )
    # Recording the random seed.
    result_df["seed"] = seed
    return result_df.to_dict("records")


def compute_reg_stats(reg_data, seed):
    # Does expectation in M predict defection in M?
    md = Lmer("M_is_defect ~ M_gamma_c + thr + (1| pid)", data=reg_data, family='binomial')
    md.fit(summarize=False)
    beta1 = md.coefs.loc["M_gamma_c", "Estimate"]
    # Does expectation in M predict leaving in V?
    md = Lmer("V_is_leave ~ M_gamma_c + thr + (1| pid)", data=reg_data, family='binomial')
    md.fit(summarize=False)
    beta2 = md.coefs.loc["M_gamma_c", "Estimate"]
    # Does defection in M predict leaving in V?
    md = Lmer("V_is_leave ~ M_is_defect + thr + (1| pid)", data=reg_data, family='binomial')
    md.fit(summarize=False)
    beta3 = md.coefs.loc["M_is_defectTRUE", "Estimate"]
    # Does the change in expectation from M to V predict the change in action from M to V?
    md = Lmer(
        "action_change ~ gamma_change + thr + (1| pid)",
        data=reg_data[lambda x: ~ x.V_is_leave],
        family='gaussian'
    )
    md.fit(summarize=False)
    beta4 = md.coefs.loc["gamma_change", "Estimate"]
    # Which traits predict leaving in V?
    std_reg_data = standardize_reg_data(reg_data)
    formula = (
        "V_is_leave ~ fs_alpha + fs_beta + r_from_hl + iri_EC + iri_FS + iri_PD "
        "+ iri_PT + siut + crt + general_trust + age + gender + thr + (1| pid)"
    )
    md = Lmer(formula, data=std_reg_data, family='binomial')
    md.fit(summarize=False)
    # Merging the results.
    ret_dict = (
        dict(seed=seed, beta1=beta1, beta2=beta2, beta3=beta3, beta4=beta4)
        | md.coefs["Estimate"].rename(lambda x: f"beta_{x}").to_dict()
    )
    return ret_dict


def standardize_reg_data(reg_data):
    # Standardize numerical column values.
    num_traits_cols = [
        'fs_alpha', 'fs_beta', 'r_from_hl', 'iri_EC', 'iri_FS', 'iri_PD',
        'iri_PT', 'siut', 'crt', 'general_trust', 'age',
    ]
    standardized_reg_data = pd.DataFrame(
        data=StandardScaler().fit_transform(reg_data[num_traits_cols]),
        index=reg_data["pid"],
        columns=num_traits_cols,
    ).reset_index()
    for col in ["gender", "V_is_leave", "thr"]:
        standardized_reg_data[col] = reg_data[col]
    return standardized_reg_data


def compute_pop_efficiency(data):
    N = data["C"] + data["D"] + data["L"]
    average_payoff = (10*N + 10*data["L"] - 10*data["C"] + 30*data["p_suc"]*(data["C"]+data["D"])) / N
    ideal_payoff = 10 + 30 - (10 * data["thr"]/5)
    return average_payoff / ideal_payoff


def compute_group_efficiency(data):
    N = data["C"] + data["D"]
    average_payoff = (10*N - 10*data["C"] + 30*data["p_suc"]*N) / N
    ideal_payoff = 10 + 30 - (10 * data["thr"]/5)
    return average_payoff / ideal_payoff


def wrap_roc_auc_score(y_true, y_pred):
    if sum(y_true) < len(y_true):
        return roc_auc_score(y_true, y_pred)
    else: # If cases only contains True (e.g., voluntary condition with q = 5), it returns null.
        return np.nan
