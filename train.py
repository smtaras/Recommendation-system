"""One-command setup for the Retailrocket recommender.

On a fresh machine (after `pip install -r requirements.txt`) just run:

    python train.py --serve

It will:
  1. download events.csv from Kaggle into data/ (if it is not there yet);
  2. train the ALS model and write artifacts/ (if they are not there yet);
  3. smoke-test the artifacts through recommender.Recommender;
  4. start the API at http://127.0.0.1:8000/docs (only with --serve).
"""
import os

# implicit warns (and runs slower) if BLAS uses its own threads. Set before imports
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import importlib.util
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # paths do not depend on the current directory
sys.path.insert(0, str(ROOT))

KAGGLE_DATASET = "retailrocket/ecommerce-dataset"
ARTIFACT_FILES = ("als_model.npz", "user_item_matrix.npz", "metadata.pkl")

EVENT_WEIGHTS = {"view": 1, "addtocart": 3, "transaction": 5}

# Same as the selected serving configuration in notebook 02.
ALS_PARAMS = dict(factors=128, regularization=0.5, iterations=10, random_state=42)
ALPHA = 40

DEFAULT_TRAIN_END = "2015-08-18 04:23:31"

POPULAR_ITEMS_TO_KEEP = 1000


# --------------------------------------------------------------------------- #
# Step 0: dependencies
# --------------------------------------------------------------------------- #
def check_dependencies(need_server: bool) -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "implicit": "implicit",
        "kagglehub": "kagglehub",
    }
    if need_server:
        required.update({"fastapi": "fastapi", "uvicorn": "uvicorn"})

    missing = [pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None]
    if missing:
        raise SystemExit(
            "Missing Python packages: " + ", ".join(missing) + "\n"
            "Install everything with:\n"
            f"    {sys.executable} -m pip install -r requirements.txt"
        )


# --------------------------------------------------------------------------- #
# Step 1: data
# --------------------------------------------------------------------------- #
def ensure_events_csv(data_dir: Path) -> Path:
    target = data_dir / "events.csv"
    if target.exists():
        print(f"[data] Using existing {target}")
        return target

    print(f"[data] {target} not found, downloading '{KAGGLE_DATASET}' from Kaggle...")
    import kagglehub

    try:
        downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:  # network / auth problems
        raise SystemExit(
            f"\nCould not download the dataset automatically: {exc}\n\n"
            "Fix options:\n"
            "  - check your internet connection;\n"
            "  - if Kaggle asks for credentials, create an API token at\n"
            "    https://www.kaggle.com/settings and either put kaggle.json into\n"
            "    ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY environment variables;\n"
            f"  - or download the dataset manually from\n"
            f"    https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"    and put events.csv into '{data_dir}'."
        )

    found = next(downloaded.rglob("events.csv"), None)
    if found is None:
        raise SystemExit(f"events.csv was not found inside the downloaded files at {downloaded}")

    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found, target)
    print(f"[data] Copied events.csv to {target}")
    return target


# --------------------------------------------------------------------------- #
# Step 2: training
# --------------------------------------------------------------------------- #
def build_interactions(events_path: Path, train_end, min_interactions: int):
    import pandas as pd

    events = pd.read_csv(
        events_path,
        usecols=["timestamp", "visitorid", "event", "itemid"],
        dtype={"visitorid": "int64", "itemid": "int64"},
    )

    if train_end is not None:
        cutoff_ms = int(pd.Timestamp(train_end, tz="UTC").timestamp() * 1000)
        events = events[events["timestamp"] <= cutoff_ms]
        print(f"[train] Using events up to {train_end} UTC: {len(events):,} rows")
    else:
        print(f"[train] Using ALL events: {len(events):,} rows")

    events = events.assign(weight=events["event"].map(EVENT_WEIGHTS)).dropna(subset=["weight"])

    # Strongest event per visitor-item pair (max, not sum).
    pairs = events.groupby(["visitorid", "itemid"], as_index=False)["weight"].max()

    if min_interactions > 1:
        per_visitor = pairs.groupby("visitorid")["itemid"].transform("size")
        before = pairs["visitorid"].nunique()
        pairs = pairs[per_visitor >= min_interactions]
        after = pairs["visitorid"].nunique()
        print(f"[train] Visitors with >= {min_interactions} items: "
              f"{after:,} of {before:,} ({after / before:.1%})")

    return pairs


def train(events_path: Path, out_dir: Path, train_end, min_interactions: int) -> int:
    import numpy as np  # noqa: F401
    from scipy.sparse import csr_matrix, save_npz
    from implicit.cpu.als import AlternatingLeastSquares

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_interactions(events_path, train_end, min_interactions)

    # Categories are built AFTER filtering, so matrix and mappings contain
    # only visitors/items that are actually present
    visitor_cat = pairs["visitorid"].astype("category")
    item_cat = pairs["itemid"].astype("category")
    visitor_ids = visitor_cat.cat.categories.to_numpy()
    item_ids = item_cat.cat.categories.to_numpy()

    matrix = csr_matrix(
        (
            pairs["weight"].to_numpy(dtype="float32"),
            (visitor_cat.cat.codes.to_numpy(), item_cat.cat.codes.to_numpy()),
        ),
        shape=(len(visitor_ids), len(item_ids)),
    )
    print(f"[train] Matrix: {matrix.shape[0]:,} visitors x {matrix.shape[1]:,} items, "
          f"{matrix.nnz:,} non-zeros")

    print("[train] Training ALS (a few minutes on a typical laptop)...")
    model = AlternatingLeastSquares(**ALS_PARAMS)
    model.fit(matrix * ALPHA)

    # Popularity fallback: items ranked by number of distinct visitors
    popularity = pairs.groupby("itemid")["visitorid"].nunique()
    popular_item_ids = (
        popularity.sort_values(ascending=False)
        .head(POPULAR_ITEMS_TO_KEEP)
        .index.astype("int64")
        .tolist()
    )

    metadata = {
        "user_to_index": {int(v): i for i, v in enumerate(visitor_ids)},
        "item_ids": item_ids,  # position in this array == column in the matrix
        "popular_item_ids": popular_item_ids,
        "config": {
            **ALS_PARAMS,
            "alpha": ALPHA,
            "event_weights": EVENT_WEIGHTS,
            "train_end": train_end,
            "min_interactions": min_interactions,
        },
    }

    model.save(str(out_dir / "als_model.npz"))
    save_npz(out_dir / "user_item_matrix.npz", matrix)
    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("[train] Saved artifacts:")
    for name in ARTIFACT_FILES:
        size_mb = (out_dir / name).stat().st_size / 1024**2
        print(f"        {name:<22} {size_mb:8.1f} MB")

    return int(visitor_ids[0])


# --------------------------------------------------------------------------- #
# Step 3: smoke test
# --------------------------------------------------------------------------- #
def smoke_test(out_dir: Path, visitor_id=None) -> None:
    from recommender import Recommender

    rec = Recommender(artifacts_dir=out_dir)
    if visitor_id is None:
        visitor_id = next(iter(rec.user_to_index))
    known = rec.recommend_items(visitor_id, n=5)
    unknown = rec.recommend_items(-1, n=5)
    print(f"[test] OK. Known visitor {visitor_id}: {known}")
    print(f"[test] OK. Unknown visitor (popular fallback): {unknown}")


# --------------------------------------------------------------------------- #
# Step 4: API
# --------------------------------------------------------------------------- #
def serve(host: str, port: int) -> None:
    print(f"\n[serve] Starting API. Open http://{host}:{port}/docs  (Ctrl+C to stop)\n")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", host, "--port", str(port)],
        cwd=ROOT,
        check=False,
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END,
                        help="UTC cutoff for training events (default: outer train end)")
    parser.add_argument("--use-all-data", action="store_true",
                        help="ignore --train-end and train on every event")
    parser.add_argument("--min-interactions", type=int, default=1,
                        help="keep only visitors with at least this many distinct items")
    parser.add_argument("--force", action="store_true",
                        help="retrain even if artifacts already exist")
    parser.add_argument("--serve", action="store_true",
                        help="start the FastAPI server when everything is ready")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)

    check_dependencies(need_server=args.serve)

    artifacts_ready = all((out_dir / name).exists() for name in ARTIFACT_FILES)
    first_visitor = None

    if artifacts_ready and not args.force:
        print(f"[train] Artifacts already exist in {out_dir}, skipping training "
              f"(use --force to retrain).")
    else:
        events_path = ensure_events_csv(data_dir)
        train_end = None if args.use_all_data else args.train_end
        first_visitor = train(events_path, out_dir, train_end, args.min_interactions)

    smoke_test(out_dir, first_visitor)

    if args.serve:
        serve(args.host, args.port)
    else:
        print("\nAll set. Start the API with:\n"
              f"    {sys.executable} -m uvicorn app:app --reload")


if __name__ == "__main__":
    main()
