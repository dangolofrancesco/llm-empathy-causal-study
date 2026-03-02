"""
merge_and_plot_big_five.py
--------------------------
1. Loads trait scores from data/user_big_five_scores.jsonl
   (keeps the LAST entry per user so re-run scores overwrite nulls)
2. Merges them into data/user_aggregate_conversations.csv
   and writes data/user_aggregate_conversations_scored.csv
3. Plots distribution of each Big Five trait (histogram + KDE)

Usage:
    python scripts/merge_and_plot_big_five.py
    python scripts/merge_and_plot_big_five.py --no-save   # don't overwrite CSV
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

TRAITS = [
    "trait_openness",
    "trait_consciousness",
    "trait_extraversion",
    "trait_agreableness",
    "trait_neuroticism",
]

TRAIT_LABELS = {
    "trait_openness":      "Openness",
    "trait_consciousness": "Conscientiousness",
    "trait_extraversion":  "Extraversion",
    "trait_agreableness":  "Agreeableness",
    "trait_neuroticism":   "Neuroticism",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

ROOT = Path(__file__).parent.parent
JSONL_PATH = ROOT / "data" / "user_big_five_scores.jsonl"
CSV_IN     = ROOT / "data" / "user_aggregate_conversations.csv"
CSV_OUT    = ROOT / "data" / "user_aggregate_conversations_scored.csv"
PLOT_OUT   = ROOT / "data" / "big_five_distributions.png"

# ── Load & deduplicate JSONL ───────────────────────────────────────────────────

def load_scores(jsonl_path: Path) -> pd.DataFrame:
    """Read the JSONL, keep the LAST (most recent) non-null record per user."""
    records: dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec["user_hashed_ip"]
            # overwrite earlier entries so the last run wins
            if records.get(uid) is None or any(rec.get(t) is not None for t in TRAITS):
                records[uid] = rec

    scores_df = pd.DataFrame(list(records.values()))
    n_total   = len(scores_df)
    n_scored  = scores_df[TRAITS[0]].notna().sum()
    n_failed  = n_total - n_scored
    print(f"[INFO] JSONL entries (unique users): {n_total}")
    print(f"[INFO]   Scored:  {n_scored}")
    print(f"[INFO]   Failed (null): {n_failed}")
    return scores_df

# ── Merge ─────────────────────────────────────────────────────────────────────

def merge(scores_df: pd.DataFrame, csv_in: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_in)
    # Drop old trait columns if present (they may all be NaN from initial creation)
    df = df.drop(columns=[t for t in TRAITS if t in df.columns], errors="ignore")
    # Also handle the typo variant "trait agreableness" (space instead of underscore)
    df = df.drop(columns=["trait agreableness"], errors="ignore")
    merged = df.merge(scores_df, on="user_hashed_ip", how="left")
    return merged

# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame, save_path: Path):
    scored = df.dropna(subset=TRAITS)
    n = len(scored)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Big Five Personality Trait Distributions  (n={n} users)",
        fontsize=16, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    axes = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(0,2),(1,0),(1,1)]]

    for ax, trait, color in zip(axes, TRAITS, COLORS):
        vals = scored[trait].dropna()

        # Histogram
        ax.hist(vals, bins=20, range=(0, 1), color=color, alpha=0.55,
                edgecolor="white", linewidth=0.5, density=True, label="histogram")

        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals, bw_method=0.15)
        xs  = np.linspace(0, 1, 300)
        ax.plot(xs, kde(xs), color=color, linewidth=2.2, label="KDE")

        # Vertical mean line
        mu = vals.mean()
        ax.axvline(mu, color="black", linewidth=1.4, linestyle="--", alpha=0.7)
        ax.text(mu + 0.02, ax.get_ylim()[1] * 0.92, f"μ={mu:.2f}",
                fontsize=8.5, color="black", va="top")

        ax.set_title(TRAIT_LABELS[trait], fontsize=12, fontweight="bold")
        ax.set_xlabel("Score (0 – 1)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide the unused 6th cell
    fig.add_subplot(gs[1, 2]).set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved → {save_path}")
    plt.show()

# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    scored = df[TRAITS].dropna()
    summary = scored.describe().T[["count", "mean", "std", "min", "50%", "max"]]
    summary.index = [TRAIT_LABELS[t] for t in TRAITS]
    summary.columns = ["n", "mean", "std", "min", "median", "max"]
    print("\n── Big Five Summary Statistics ──────────────────────────────")
    print(summary.to_string(float_format="{:.3f}".format))
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true",
                        help="Skip overwriting the scored CSV")
    args = parser.parse_args()

    scores_df = load_scores(JSONL_PATH)
    merged    = merge(scores_df, CSV_IN)

    if not args.no_save:
        merged.to_csv(CSV_OUT, index=False)
        print(f"[INFO] Scored CSV saved → {CSV_OUT}")

    print_summary(merged)
    plot_distributions(merged, PLOT_OUT)


if __name__ == "__main__":
    main()
