"""Pelvis height over the episode across all evaluated fallen states, nominal vs. pushed at t = 5 s.

    python scripts/analysis/plot_eval.py eval_nominal.npz eval_push1.npz media/
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

NOMINAL, PUSH, OUT = sys.argv[1:4]
THEMES = {
    "light": dict(bg="#fcfcfb", fg="#0b0b0b", fg2="#52514e", grid="#e2e1dc", series="#2a78d6"),
    "dark": dict(bg="#1a1a19", fg="#f0efec", fg2="#c3c2b7", grid="#3a3936", series="#3987e5"),
}


def bands(path):
    d = np.load(path)
    h = d["height"].astype(np.float32)
    t = np.arange(h.shape[0]) * float(d["dt"])
    up = (h >= 0.65) & (d["gz"].astype(np.float32) <= -0.9)
    return t, np.percentile(h, [1, 5, 50, 95, 99], axis=1), up[-int(2 / float(d["dt"])):].all(0).mean(), h.shape[1]


panels = [("No disturbance", *bands(NOMINAL), None),
          ("Random kick up to 1 m/s per axis at t = 5 s", *bands(PUSH), 5.0)]
for mode, th in THEMES.items():
    plt.rcParams.update({"font.size": 11, "text.color": th["fg"], "axes.labelcolor": th["fg2"],
                         "xtick.color": th["fg2"], "ytick.color": th["fg2"]})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=th["bg"], sharey=True)
    for ax, (title, t, p, succ, n, push) in zip(axes, panels):
        ax.set_facecolor(th["bg"])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="y", color=th["grid"], lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.fill_between(t, p[0], p[4], color=th["series"], alpha=0.12, lw=0)
        ax.fill_between(t, p[1], p[3], color=th["series"], alpha=0.28, lw=0)
        ax.plot(t, p[2], color=th["series"], lw=2)
        ax.axhline(0.74, color=th["fg2"], lw=1, ls=(0, (3, 3)))
        ax.text(9.9, 0.755, "target 0.74 m", color=th["fg2"], fontsize=9, ha="right")
        if push is not None:
            ax.axvline(push, color=th["fg2"], lw=1, ls=(0, (3, 3)))
            ax.text(push + 0.1, 0.05, "kick", color=th["fg2"], fontsize=9)
        ax.set_title(f"{title}\n{succ:.1%} of {n} rollouts standing for the last 2 s", loc="left", fontsize=11,
                     color=th["fg"], pad=10)
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("pelvis height (m)")
    axes[0].text(1.2, 0.3, "line: median\ndark band: 5-95 %\nlight band: 1-99 %", color=th["fg2"], fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/12-eval-height-{mode}.png", dpi=120, facecolor=th["bg"])
    print("wrote", f"{OUT}/12-eval-height-{mode}.png")
