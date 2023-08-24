import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import sys
sys.path.append("../")

from tools import p_k, save_dir

plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = "Helvetica"
script_name = re.sub(r"\.py$", "", os.path.basename(__file__))


def main():
    exp_data = pd.read_csv("../data/exp_result.csv", index_col=0)
    plot_base(exp_data)
    plot_pie(exp_data)


def plot_base(exp_data):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i, (thr, thr_data) in enumerate(exp_data.groupby("thr")):
        for j, (p_option, tgt_data) in enumerate(thr_data.groupby("p_option")):
            ax = axes[j][i]
            hue_order = ["C", "D"] if p_option == "A" else ["C", "D", "L"]
            hist = sns.histplot(
                tgt_data,
                x="gamma_c",
                hue="action",
                hue_order=hue_order,
                palette=["C1", "C0", "C2"],
                line_kws={"width": 0},
            linewidth=0,
                multiple="stack",
                bins=np.linspace(0, 1, 11),
                ax=ax,
                legend=(thr == 2),
                alpha=1
            )
            ax.set_ylim(0, 160)
            ax.spines[["top", "right"]].set_visible(False)
            legend = hist.legend_
            if legend is not None:
                legend.set_frame_on(False)
                legend.set_title("")
            if thr != 2:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if p_option == "F":
                ax.set_xlabel("γ")
            else:
                ax.set_title(f"Threshold: {thr}")
                ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{save_dir(script_name)}/FigS2-base.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pie(exp_data):
    for (thr, p_option), tgt_data in exp_data.groupby(["thr", "p_option"]):
        plt.figure(figsize=(3, 3))
        options = ["C", "D"] if p_option == "A" else ["C", "D", "L"]
        cnt_actions = tgt_data.action.value_counts().reindex(options, fill_value=0)
        plt.pie(
            cnt_actions, startangle=90,
            autopct=lambda x: int(round(191*x/100, 0)),
            counterclock=False, colors=["C1", "C0", "C2"],
            textprops={'fontsize': 30, "color": "w"},
            explode=[.01 for _ in options], wedgeprops={"alpha": 1}
        )
        plt.axis('equal')
        plt.savefig(
            f"{save_dir(script_name)}/pie_{thr}_{p_option}.pdf",
            format="pdf", dpi=300, bbox_inches="tight", transparent=True
        )
        plt.close()


if __name__ == "__main__":
    main()
