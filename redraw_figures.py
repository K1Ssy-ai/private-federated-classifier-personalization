"""Draw publication figures from archived CSV rows, without model evaluation."""
from pathlib import Path
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MaxNLocator, NullLocator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 0.9,
})
COLORS = ["#006A8E", "#A74419"]


def records(name):
    with (ROOT / "results" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(OUT / name, metadata={"Creator": "Matplotlib 3.9.2"}, facecolor="white")
    plt.close(fig)


def framework():
    fig = plt.figure(figsize=(4.69, 6.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, color="#EDF3F7"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.012",
            linewidth=0.7, edgecolor="#34495E", facecolor=color,
        ))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8)

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=7, linewidth=0.7,
            linestyle="--" if dashed else "-", color="#263238",
        ))

    ax.text(.03, .97, "(a) Shared training", weight="bold", va="top")
    box(.18, .865, .64, .065, "Shared model\nfeature extractor + affine head")
    arrow((.5, .865), (.5, .835))
    ax.text(.52, .851, "Broadcast to 20 clients", va="center")
    box(.18, .725, .64, .11, "Each client takes 10 local DP-SGD steps\n"
        "Per-record clipping (C = 1)\nGaussian noise + SGD update")
    arrow((.5, .725), (.5, .685))
    ax.text(.52, .705, "Upload full model", va="center")
    box(.18, .62, .64, .065, "Server averages 20 returned models")
    arrow((.18, .652), (.08, .652))
    ax.plot([.08, .08], [.652, .898], color="#263238", linewidth=.7)
    arrow((.08, .898), (.18, .898))
    ax.text(.09, .77, "30 rounds", rotation=90, va="center")
    box(.04, .525, .92, .055, r"Joint record-level budget: $\epsilon=4$ or $8$, $\delta=10^{-5}$", "#F2F2F2")
    ax.text(.03, .49, "(b) Local personalization", weight="bold", va="top")
    box(.05, .38, .4, .065, "Frozen shared predictor\nscores z(x)")
    arrow((.45, .412), (.55, .412))
    box(.55, .38, .4, .065, "Add client offset d\nsoftmax(z(x) + d)")
    arrow((.75, .38), (.75, .337))
    ax.text(.57, .354, "Local labels", ha="right", va="center")
    arrow((.59, .354), (.71, .323))
    box(.55, .255, .4, .08, "Cross-entropy\ngradient w.r.t. d")
    arrow((.55, .295), (.45, .295))
    box(.05, .255, .4, .08, "Per-record clipping\nC = 1 or 2")
    arrow((.25, .255), (.25, .205))
    box(.05, .125, .4, .08, "Gaussian noise\nSGD update")
    arrow((.45, .165), (.55, .165))
    box(.55, .125, .4, .08, "Updated local offset\n10 trainable values")
    arrow((.97, .166), (.97, .411), dashed=True)
    arrow((.95, .166), (.97, .166), dashed=True)
    arrow((.97, .411), (.95, .411), dashed=True)
    ax.text(.5, .072, "50 local steps; initialize d = 0", ha="center")
    ax.text(.5, .035, "Merge the offset into the shared intercept for inference", ha="center")
    save(fig, "Fig1.pdf")


def clipping():
    rows = records("02_clipping_sweep.csv")
    fig, axes = plt.subplots(4, 1, figsize=(4.69, 6.7))
    fig.subplots_adjust(left=.16, right=.84, bottom=.08, top=.90, hspace=.67)
    conditions = [("fashion", .1), ("fashion", 1), ("cifar", .1), ("cifar", 1)]
    for k, (ax, (dataset, alpha)) in enumerate(zip(axes, conditions)):
        other = ax.twinx()
        for method, label, marker, color in zip(
            ["B1", "B2"], ["Output bias", "Full head"], ["o", "s"], COLORS
        ):
            xs = [.5, 1, 2, 4, 8]
            selected = [[r for r in rows if r["dataset"] == dataset
                         and float(r["alpha"]) == alpha and r["method"] == method
                         and float(r["clip"]) == c] for c in xs]
            assert all(len(group) == 5 for group in selected)
            accuracy = [100*statistics.mean(float(r["test_accuracy"]) for r in g) for g in selected]
            clipped = [100*statistics.mean(float(r["clip_fraction"]) for r in g) for g in selected]
            ax.plot(xs, accuracy, marker=marker, markersize=3.2, label=label, color=color)
            other.plot(xs, clipped, linestyle="--", marker=marker, markersize=2.4, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks([.5, 1, 2, 4, 8], [".5", "1", "2", "4", "8"])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        other.set_ylim(-3, 103)
        other.set_yticks([0, 50, 100])
        ax.set_ylabel("Accuracy (%)")
        other.set_ylabel("Clipped (%)")
        ax.grid(axis="y", color=".85", linewidth=.35)
        name = "Fashion-MNIST" if dataset == "fashion" else "CIFAR-10"
        ax.text(0, 1.07, f"({chr(97+k)}) {name}, α = {alpha:g}", transform=ax.transAxes)
        if k == 0:
            fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .995), ncol=2, frameon=False, handlelength=1.5)
    axes[-1].set_xlabel("Clipping threshold C")
    save(fig, "Fig2.pdf")


def noise():
    rows = records("05_score_noise.csv")
    fig, axes = plt.subplots(1, 2, figsize=(4.69, 2.55))
    fig.subplots_adjust(left=.14, right=.98, bottom=.23, top=.87, wspace=.42)
    for k, (ax, level) in enumerate(zip(axes, ["nominal", "stress"])):
        all_values = []
        for method, label, marker, color in zip(
            ["bias", "full"], ["Output bias", "Full head"], ["o", "+"], COLORS
        ):
            data = [r for r in rows if r["level"] == level and r["method"] == method]
            assert len(data) == 90
            x = [float(r["pair_theory"]) for r in data]
            y = [float(r["pair_var"]) for r in data]
            ax.scatter(x, y, marker=marker, s=7, linewidths=.6, color=color, label=label)
            all_values.extend(x+y)
        low, high = min(all_values)*.9, max(all_values)*1.1
        ax.plot([low, high], [low, high], color=".25", linewidth=.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Theoretical variance")
        if k == 0:
            ax.set_ylabel("Empirical variance")
            ax.legend(loc="lower right", frameon=False, handlelength=.8, handletextpad=.3)
        ax.text(0, 1.08, f"({chr(97+k)}) {level.capitalize()}", transform=ax.transAxes)
    save(fig, "Fig3.pdf")


framework()
clipping()
noise()
