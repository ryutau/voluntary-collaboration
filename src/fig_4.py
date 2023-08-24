import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
sys.path.append("../")

from tools import p_k, save_dir


plt.rcParams['image.cmap'] = 'RdYlGn'
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = "Helvetica"
plt.rcParams['hatch.linewidth'] = 1.5
script_name = re.sub(r"\.py$", "", os.path.basename(__file__))


def main():
    exp_data = pd.read_csv("../data/exp_result.csv", index_col=0)
    cnt_actions_per_condition = exp_data.groupby(
        ["p_option", "thr"]
    ).action.value_counts().unstack().fillna(0).astype(int)
    m_color = "#FF4747"
    v_color = "#8CE7F8"
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    fig.patch.set_alpha(0.0)
    rho_ticks = np.linspace(0, 1, 6)
    for (p_option, thr), cnt_actions in cnt_actions_per_condition.iterrows():
        ax = axes[{2: 0, 4: 1, 5: 2}[thr]]
        plot_args = dict(color="b", marker="o", ls="--", linewidth=1.5, fillstyle="full", markerfacecolor="white") if p_option == "F" else dict(color=m_color)
        ax.plot(
            rho_ticks,
            [sum([p_k(5, cnt_actions.C/(cnt_actions.C+cnt_actions.D+cnt_actions.L*rho), k)
                  for k in range(thr, 6)])
             for rho in rho_ticks],
            label={"A": "$\it{Without}$ individual option", "F": "$\it{With}$ individual option"}[p_option],
            **plot_args
        )
        ax.set_ylim(-.02, 1.04)
        ax.set_xlabel(r"Loners' externality (ρ)")
        if thr == 2:
            ax.set_ylabel("Group success rate ($p_\mathrm{success}$)")
            ax.legend(frameon=False, fontsize=10)
        else:
            ax.set_ylabel("")
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.set_title(f"Threshold: ${thr}$")
    for ax in axes:
        ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(f"{save_dir(script_name)}/Fig4.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
