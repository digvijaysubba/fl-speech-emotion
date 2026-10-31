"""Make the plots and the results table from the JSON files in results/.

    python plot_results.py

Writes results/fig1_fedavg_splits.png, fig2_fedprox.png, fig3_split_learning.png and
results/summary.md. Every number in them is read from a run's JSON file.
Settings with no result files yet are skipped.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# setting -> (method, data split), in table order
ROWS = {
    "centralized_random": ("Centralized, plain 60 epochs", "Random 80/20 split ⚠ different test set: speakers appear in train and test"),
    "centralized_speaker": ("Centralized, plain 60 epochs", "All 20 training speakers on one machine"),
    "centralized_valstop": ("Centralized, epochs chosen on 2 validation speakers", "All 20 training speakers on one machine"),
    "fedavg_iid": ("FedAvg", "IID"),
    "fedavg_speaker": ("FedAvg", "2 speakers per client"),
    "fedavg_dirichlet0.5": ("FedAvg", "Label skew (Dir. α=0.5)"),
    "fedprox0_dirichlet0.5": ("FedProx μ=0", "Label skew (Dir. α=0.5)"),
    "fedprox0.001_dirichlet0.5": ("FedProx μ=0.001", "Label skew (Dir. α=0.5)"),
    "fedprox0.01_dirichlet0.5": ("FedProx μ=0.01", "Label skew (Dir. α=0.5)"),
    "fedprox0.1_dirichlet0.5": ("FedProx μ=0.1", "Label skew (Dir. α=0.5)"),
    "split_speaker": ("Split learning (SFL-V2)", "2 speakers per client"),
    "split_dirichlet0.5": ("Split learning (SFL-V2)", "Label skew (Dir. α=0.5)"),
}
LAST_K = 5  # "last 5 rounds" (last 5 epochs for centralized); fixed in advance, not chosen by test score


def load(setting):
    runs = [json.loads(p.read_text()) for p in config.RESULTS_DIR.glob(f"{setting}_seed*.json")]
    return sorted(runs, key=lambda r: r["seed"])


def central_line(ax):
    """Dashed reference: plain centralized final score (the early-stopped one is in the label)."""
    runs, vs = load("centralized_speaker"), load("centralized_valstop")
    if runs:
        c = np.mean([r["final"]["macro_f1"] for r in runs])
        text = f"Centralized, plain 60 epochs: {c:.3f}"
        if vs:
            text += f"  (epochs chosen on validation: {np.mean([r['final']['macro_f1'] for r in vs]):.3f})"
        ax.axhline(c, color=MUTED, lw=1.5, ls="--", zorder=1, label=text)


def draw(ax, setting, label, color):
    runs = load(setting)
    if not runs:
        return
    F = np.array([[h["macro_f1"] for h in r["history"]] for r in runs])
    mean, rounds = F.mean(0), np.arange(F.shape[1])
    ax.fill_between(rounds, F.min(0), F.max(0), color=color, alpha=0.13, lw=0)
    ax.plot(rounds, mean, color=color, lw=2, label=f"{label} (final {mean[-1]:.3f}, n={len(runs)})")


def style(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Communication round", color=INK)
    ax.set_xlim(0, config.NUM_ROUNDS)
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="lower right", frameon=False, fontsize=8)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / name)
    plt.close(fig)
    print("wrote", config.RESULTS_DIR / name)


def plots():
    ylabel = "Test macro-F1 (held-out speakers)"
    # Figure 1: FedAvg under the three ways of splitting data.
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    central_line(ax)
    draw(ax, "fedavg_iid", "IID clients", BLUE)
    draw(ax, "fedavg_speaker", "2 speakers per client", ORANGE)
    draw(ax, "fedavg_dirichlet0.5", "Label skew, Dir. α=0.5", AQUA)
    ax.set_ylabel(ylabel, color=INK)
    style(ax, "FedAvg: global-model macro-F1 after each round (line = mean over seeds, band = min–max)")
    save(fig, "fig1_fedavg_splits.png")

    # Figure 2: FedProx strength sweep on label skew (one hue, darker = larger mu).
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    for mu, color in [("0", "#86b6ef"), ("0.001", "#5598e7"), ("0.01", "#256abf"), ("0.1", "#0d366b")]:
        draw(ax, f"fedprox{mu}_dirichlet0.5", f"FedProx μ={mu}", color)
    ax.set_ylabel(ylabel, color=INK)
    style(ax, "Label skew (Dir. α=0.5): FedProx with different μ (μ=0 = same code, no proximal term)")
    save(fig, "fig2_fedprox.png")

    # Figure 3: split learning vs FedAvg, one panel per data split.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150, sharey=True)
    for ax, part, title in [(axes[0], "speaker", "2 speakers per client"),
                            (axes[1], "dirichlet0.5", "Label skew, Dir. α=0.5")]:
        central_line(ax)
        draw(ax, f"fedavg_{part}", "FedAvg", BLUE)
        draw(ax, f"split_{part}", "Split learning (SFL-V2)", ORANGE)
        style(ax, title)
    axes[0].set_ylabel(ylabel, color=INK)
    save(fig, "fig3_split_learning.png")


def table():
    """Final-round numbers plus the average of the last 5 rounds/epochs (less sensitive
    to where the final round happens to land). No best-round column: choosing the best
    round by test score would use the test set for model selection."""
    lines = [f"| Method | Data split | Accuracy, final (mean ± sd) | Macro-F1, final (mean ± sd) | "
             f"Macro-F1, avg of last {LAST_K} rounds (mean ± sd) | Macro-F1 per seed, final |",
             "|---|---|---|---|---|---|"]
    sd = lambda v: np.std(v, ddof=1) if len(v) > 1 else float("nan")
    for setting, (method, split) in ROWS.items():
        runs = load(setting)
        if not runs:
            continue
        acc = [r["final"]["accuracy"] for r in runs]
        f1 = [r["final"]["macro_f1"] for r in runs]
        last = [np.mean([h["macro_f1"] for h in r["history"][-LAST_K:]]) for r in runs]
        lines.append(f"| {method} | {split} | {100*np.mean(acc):.1f}% ± {100*sd(acc):.1f} | "
                     f"{np.mean(f1):.3f} ± {sd(f1):.3f} | {np.mean(last):.3f} ± {sd(last):.3f} | "
                     f"{', '.join(f'{v:.3f}' for v in f1)} |")
    lines.append(f"\nSeeds 0, 1, 2 for every row. Test set: the 208 recordings of held-out speakers "
                 f"{config.TEST_SPEAKERS}, except the random-split row. For centralized rows, "
                 f"'rounds' means epochs (plain: final = epoch 60, last 5 = epochs 56-60; "
                 f"early-stopped: final = the chosen epoch, last 5 = the 5 epochs up to it).")
    path = config.RESULTS_DIR / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("wrote", path)


if __name__ == "__main__":
    plots()
    table()
