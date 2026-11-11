"""End-to-end offline evaluation of the served system (ALS + popularity fallback).

The notebook metric only scores visitors who were already present in training
("warm" visitors). This script scores EVERY visitor who has a future
add-to-cart / transaction in the test period, using the very same
`Recommender.recommend_items` that the API serves:

  - warm visitor  -> ALS recommendations (+ popularity filler)
  - cold visitor  -> popularity fallback

Metric: macro Recall@N (default N=10), same definition as Notebook 02:
for each visitor, the fraction of their unique future target items that appear in
the top-N, averaged over visitors. Targets exclude items the visitor already saw in
training and items that are absent from the training catalogue.

Usage:
    python evaluate_hybrid.py
    python evaluate_hybrid.py --sample 3000      # quick trial on a random subset

Requires artifacts trained on the OUTER TRAINING PERIOD only (the default of
`python train.py`, or the notebook export). Artifacts trained with --use-all-data would leak the test period.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import pickle
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TRAIN_QUANTILE = 0.8  # outer training period = first 80% of events (same as notebook / train.py)
TARGET_EVENTS = ("addtocart", "transaction")


# --------------------------------------------------------------------------- #
def read_config(artifacts_dir: Path) -> dict:
    with open(artifacts_dir / "metadata.pkl", "rb") as f:
        return pickle.load(f).get("config", {})


def resolve_cutoff_ms(config: dict, cli_value, events: pd.DataFrame) -> int:
    """Cutoff (ms) between training and test period."""
    if cli_value:
        return int(pd.Timestamp(cli_value, tz="UTC").timestamp() * 1000)
    if "train_end_ms" in config:
        if config["train_end_ms"] is None:
            raise SystemExit(
                "These artifacts were trained on ALL events (--use-all-data), so the test "
                "period leaked into training and an evaluation would be meaningless.\n"
                "Retrain on the training period only:  python train.py --force"
            )
        return int(config["train_end_ms"])
    # Artifacts exported from the notebook carry no cutoff: it used the 80% quantile.
    print("NOTE: artifacts have no recorded cutoff (notebook export?); "
          f"using the {TRAIN_QUANTILE:.0%} time quantile like the notebook.")
    return int(events["timestamp"].quantile(TRAIN_QUANTILE))


def load_events(events_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        events_path,
        usecols=["timestamp", "visitorid", "event", "itemid"],
        dtype={"visitorid": "int64", "itemid": "int64"},
    )


def load_test_targets(events: pd.DataFrame, cutoff_ms: int) -> dict:
    """Return {event: {visitor_id: set(item_ids)}} for events AFTER the cutoff."""
    test = events[events["timestamp"] > cutoff_ms]
    print(f"Test period: {len(test):,} events, {test['visitorid'].nunique():,} visitors")

    targets = {}
    for event in TARGET_EVENTS:
        sub = test[test["event"] == event]
        targets[event] = {
            int(v): {int(i) for i in items}
            for v, items in sub.groupby("visitorid")["itemid"]
        }
    return targets


def seen_items(rec, user_index: int) -> set:
    m = rec.user_item_matrix
    cols = m.indices[m.indptr[user_index]:m.indptr[user_index + 1]]
    return {int(rec.item_ids[c]) for c in cols}


def recall(recommended, eligible: set) -> float:
    return len(set(recommended) & eligible) / len(eligible)


def evaluate_event(rec, targets: dict, n: int):
    """Score one target type. Returns (rows, diagnostics)."""
    catalogue = {int(i) for i in rec.item_ids}
    popular = [int(i) for i in rec.popular_item_ids]

    rows = []  # (segment, system_recall, popularity_recall)
    pairs_total = pairs_unknown_item = 0

    for k, (visitor, items) in enumerate(targets.items(), 1):
        idx = rec.user_to_index.get(int(visitor))
        warm = idx is not None
        seen = seen_items(rec, idx) if warm else set()

        pairs_total += len(items)
        pairs_unknown_item += sum(1 for i in items if i not in catalogue)

        eligible = {i for i in items if i in catalogue and i not in seen}
        if not eligible:
            continue

        system = rec.recommend_items(visitor, n=n)
        popularity = [i for i in popular if i not in seen][:n]

        rows.append((
            "warm" if warm else "cold",
            recall(system, eligible),
            recall(popularity, eligible),
        ))

        if k % 5000 == 0:
            print(f"  ... {k:,} / {len(targets):,} visitors")

    diagnostics = {
        "target_visitors": len(targets),
        "scored_visitors": len(rows),
        "pairs_total": pairs_total,
        "pairs_unknown_item": pairs_unknown_item,
    }
    return rows, diagnostics


# --------------------------------------------------------------------------- #
def fmt(x) -> str:
    return f"{x:.4f}"


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 42):
    """95% percentile bootstrap CI of the mean of `values`."""
    import numpy as np

    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype="float64")
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = values[rng.integers(0, len(values), size=len(values))].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(event: str, rows, diagnostics, n: int) -> None:
    df = pd.DataFrame(rows, columns=["segment", "system", "popularity"])
    total = len(df)
    warm = df[df["segment"] == "warm"]
    cold = df[df["segment"] == "cold"]

    print(f"\n### {event}: macro Recall@{n}\n")
    print(f"| Visitors | Share | ALS + fallback | Popularity only |")
    print(f"| --- | ---: | ---: | ---: |")
    print(f"| All scored visitors ({total:,}) | 100% | **{fmt(df['system'].mean())}** | {fmt(df['popularity'].mean())} |")
    if len(warm):
        print(f"| Warm: seen in training ({len(warm):,}) | {len(warm) / total:.1%} | {fmt(warm['system'].mean())} | {fmt(warm['popularity'].mean())} |")
    if len(cold):
        print(f"| Cold: unseen, fallback ({len(cold):,}) | {len(cold) / total:.1%} | {fmt(cold['system'].mean())} | {fmt(cold['popularity'].mean())} |")

    diff = (df["system"] - df["popularity"]).to_numpy()
    lo, hi = bootstrap_ci(diff)
    verdict = ("distinguishable from zero" if lo > 0 or hi < 0
               else "NOT distinguishable from zero")
    print(f"\nAll visitors, paired difference (ALS + fallback minus popularity): "
          f"{diff.mean():+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}] ({verdict}).")
    if len(warm) > 1:
        wlo, whi = bootstrap_ci(warm["system"].to_numpy())
        print(f"Warm ALS Recall@{n}: {warm['system'].mean():.4f}, 95% CI [{wlo:.4f}, {whi:.4f}] "
              f"(only {len(warm):,} visitors, so the interval is wide).")

    d = diagnostics
    unknown_share = d["pairs_unknown_item"] / d["pairs_total"] if d["pairs_total"] else 0
    print(
        f"\nVisitors with future {event}: {d['target_visitors']:,}; "
        f"with at least one eligible target: {d['scored_visitors']:,}. "
        f"Target (visitor, item) pairs whose item is absent from the training catalogue: "
        f"{unknown_share:.1%}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--train-end", default=None,
                        help="UTC cutoff; default: read from artifacts' metadata "
                             "(or the 80%% time quantile for notebook-exported artifacts)")
    parser.add_argument("--n", type=int, default=10, help="recommendation list length")
    parser.add_argument("--sample", type=int, default=None,
                        help="evaluate a random subset of visitors per target (quick trial)")
    args = parser.parse_args()

    artifacts_dir, data_dir = Path(args.artifacts_dir), Path(args.data_dir)
    events_path = data_dir / "events.csv"
    if not events_path.exists() or not (artifacts_dir / "metadata.pkl").exists():
        raise SystemExit("Missing data or artifacts. Run `python train.py` first.")

    config = read_config(artifacts_dir)
    if config.get("min_interactions", 1) > 1:
        print(f"NOTE: artifacts were trained with min_interactions={config['min_interactions']}: "
              f"visitors below that threshold count as cold here.")
    events = load_events(events_path)
    cutoff_ms = resolve_cutoff_ms(config, args.train_end, events)
    print(f"Training cutoff: {pd.to_datetime(cutoff_ms, unit='ms', utc=True)}")

    from recommender import Recommender

    rec = Recommender(artifacts_dir=artifacts_dir)
    rec.user_item_matrix = rec.user_item_matrix.tocsr()

    targets = load_test_targets(events, cutoff_ms)

    for event in TARGET_EVENTS:
        event_targets = targets[event]
        if args.sample and len(event_targets) > args.sample:
            rng = random.Random(42)
            keys = rng.sample(sorted(event_targets), args.sample)
            event_targets = {k: event_targets[k] for k in keys}
        print(f"\nEvaluating '{event}' targets for {len(event_targets):,} visitors...")
        rows, diagnostics = evaluate_event(rec, event_targets, args.n)
        report(event, rows, diagnostics, args.n)

    print("\nSanity check: the notebook recorded, for the same warm visitors (696 / 266), "
          "ALS 0.0214 / 0.0134 and popularity 0.0088 / 0.0171. Visitor counts and popularity "
          "should match exactly; ALS may differ within its confidence interval.")


if __name__ == "__main__":
    main()
