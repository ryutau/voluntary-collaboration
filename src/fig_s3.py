import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    exp_data = pd.read_csv("../data/exp_result.csv", index_col=0)
    voluntary_data = exp_data[lambda x: x.p_option == "F"]
    voluntary_data["gamma_class"] = voluntary_data['gamma_c'].apply(floor_float, n=1)
    voluntary_data["is_defect"] = voluntary_data["action"] == "D"

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.3))
    for ax, (thr, thr_data) in zip(axes, voluntary_data.groupby("thr")):
        ax.set_title(f'Threshold: {thr}', fontsize=16)
        x = [i / 10 for i in range(11)]
        non_coop_rate = 1 - thr_data.groupby(["gamma_class"]).is_coop.mean().reindex(x)
        defect_rate = thr_data.groupby(["gamma_class"]).is_defect.mean().reindex(x)
        ax.plot(x, non_coop_rate, color='C0', linestyle='-', marker='o', label="Noncoop.")
        ax.plot(x, defect_rate, color='C0', linestyle='--', marker='.', label="Defect")
        ax.set_ylim(-.1, 1.1)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlim(-.05, 1.05)
        if thr == 2:
            ax.set_ylabel(r"Defection/Noncooperation rate")
            ax.set_yticks([0, .5, 1])
            ax.legend(frameon=False, loc="upper right", fontsize=12)
        elif thr == 4:
            ax.set_xlabel('Belief about others (γ) under voluntary participation', labelpad=13)
        ax.set_zorder(2)
        ax.patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig(f"{save_dir(script_name)}/FigS3.pdf", format="pdf", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()
