import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import re
import os
import sys
sys.path.append("../")

from tools import p_k, floor_float, save_dir


plt.rcParams['image.cmap'] = 'RdYlGn'
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = "Helvetica"
script_name = re.sub(r"\.py$", "", os.path.basename(__file__))


def main():
    exp_data = pd.read_csv("../data/main_exp_result.csv", index_col=0)
    group_data = exp_data[lambda x: ~x.is_leave].dropna(subset=["gamma_c"])
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for (thr, p_option), tgt_data in group_data.groupby(["thr", "p_option"]):
        row = 0 if p_option == "A" else 1
        col = {2: 0, 4: 1, 5:2}[thr]
        ax = axes[row][col]
        tgt_data["p_piv"] = tgt_data.apply(lambda x: p_k(4, x.gamma_c, x.thr - 1), axis=1)

        fpr_raw, tpr_raw, thresholds_raw = roc_curve(tgt_data["is_coop"], tgt_data["gamma_c"])
        fpr_piv, tpr_piv, thresholds_piv = roc_curve(tgt_data["is_coop"], tgt_data["p_piv"])

        # ROC-curve
        ax.plot(fpr_raw, tpr_raw, label="Raw expectation ($\gamma$)", color="C1")
        ax.plot(fpr_piv, tpr_piv, label='Pivotal probability', color="C0")
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])
        ax.spines[["top", "right"]].set_visible(False)
        if row == 1:
            ax.set_xlabel('False Positive Rate')
        else:
            ax.set_title(f"Threshold: {thr}", y=1.1)
        if col == 0:
            ax.set_ylabel('True Positive Rate')
            p_option_str = "Mandatory" if p_option == "A" else "Voluntary"
            ax.text(-1.2, .5, p_option_str, fontsize=16)
        if row == 1 and col == 2:
            ax.legend(frameon=False, fontsize=12, loc="upper left")
        else:
            ax.plot([0, 1], [0, 1], 'k--')
    plt.tight_layout()
    plt.savefig(f"{save_dir(script_name)}/FigS3.pdf", format="pdf", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()
