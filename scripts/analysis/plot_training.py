"""Compare the training curves of the minimal-model run and the full-model run, from TensorBoard logs.

    python scripts/analysis/plot_training.py <tb dir> media/
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB, OUT = sys.argv[1], sys.argv[2]
RUNS = {  # run dir -> label
    "2026-09-21_11-15-29": "Minimal model (Sept.)",
    "2026-11-05_10-35-29": "Full model (Nov.)",
}
PENALTIES = [  # summed for reference only: the weights differ between the two runs
   "base_lin_vel_xy_l2", "ang_vel_xy_l2", "feet_slide", "joint_deviation_arms", "dof_torques_l2",
             "dof_acc_l2", "action_rate_l2", "dof_vel_l2", "dof_pos_limits"]
THEMES = {
    "light": dict(bg="#fcfcfb", fg="#0b0b0b", fg2="#52514e", grid="#e2e1dc", series=["#2a78d6", "#eb6834"]),
    "dark": dict(bg="#1a1a19", fg="#f0efec", fg2="#c3c2b7", grid="#3a3936", series=["#3987e5", "#d95926"]),
}


def load(run):
    ea = EventAccumulator(f"{TB}/{run}", size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])

    def series(tag):
        s = ea.Scalars(tag)
        return np.array([e.step for e in s]), np.array([e.value for e in s])

    stand = series("Episode_Reward/stand_up_exp")
    pen_terms = [series(f"Episode_Reward/{t}") for t in PENALTIES if f"Episode_Reward/{t}" in tags]
    n = min(len(p[1]) for p in pen_terms)
    pen = (pen_terms[0][0][:n], np.sum([p[1][:n] for p in pen_terms], axis=0))
    return stand, pen


def smooth(y, w=25):
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")


data = {run: load(run) for run in RUNS}
LABEL_DY = [10, -10]  # the two final values are 0.02 apart: push the labels apart
for mode, th in THEMES.items():
    plt.rcParams.update({"font.size": 11, "text.color": th["fg"], "axes.labelcolor": th["fg2"],
                         "xtick.color": th["fg2"], "ytick.color": th["fg2"]})
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=th["bg"])
    ax.set_facecolor(th["bg"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color=th["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("Episode_Reward/stand_up_exp (max 2.0): the full model learns more slowly and ends 0.02 lower",
                 loc="left", fontsize=12, color=th["fg"], pad=12)
    ax.set_xlabel("PPO iteration")
    for color, dy, (run, label) in zip(th["series"], LABEL_DY, RUNS.items()):
        x, y = data[run][0]
        ys = smooth(y)
        ax.plot(x, ys, color=color, lw=2)
        ax.annotate(f"{label}: {ys[-1]:.2f}", (x[-1], ys[-1]), xytext=(10, dy), textcoords="offset points",
                    va="center", color=th["fg"], fontsize=10)
    for it, txt in [(1250, "assist gone\n(30 000 steps)"), (1667, "penalties raised\n(40 000 steps)")]:
        ax.axvline(it, color=th["fg2"], lw=1, ls=(0, (3, 3)))
        ax.text(it + 25, 0.12, txt, color=th["fg2"], fontsize=9)
    ax.set_xlim(0, 3900)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_ylim(0, 2.1)
    fig.tight_layout()
    fig.savefig(f"{OUT}/10-curves-models-{mode}.png", dpi=120, facecolor=th["bg"])
    print("wrote", f"{OUT}/10-curves-models-{mode}.png")
