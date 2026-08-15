#!/usr/bin/env python3
"""Download all benchmark datasets from HuggingFace.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --only gdpval truthfulqa
    python scripts/download_datasets.py --gdpval-files-only
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from datasets import load_dataset

MAX_WORKERS = 32


# Each entry: (local_name, hf_repo, hf_subset, hf_split).
DATASETS: list[tuple[str, str, str | None, str | None]] = [
    ("gdpval", "openai/gdpval", None, None),
    ("truthfulqa", "truthful_qa", "generation", None),
    ("simpleqa", "basicv8vc/SimpleQA", None, None),
    ("ifeval", "google/IFEval", None, None),
    ("harmbench", "huihui-ai/harmbench_behaviors", None, None),
    ("or_bench", "bench-llm/or-bench", "or-bench-hard-1k", None),
    ("agentharm", "ai-safety-institute/AgentHarm", "harmful", None),
    ("agent_safety_bench", "thu-coai/Agent-SafetyBench", None, None),
]

# Datasets that need manual download (not on HuggingFace):
#   toolemu - https://github.com/ryoungj/ToolEmu (copy assets/all_cases.json to data/raw/toolemu/)
#   asb     - https://github.com/agiresearch/ASB (copy data/ contents to data/raw/asb/)

DATA_ROOT = Path("data/raw")


def download_one(
    name: str,
    hf_repo: str,
    hf_subset: str | None,
    hf_split: str | None,
) -> bool:
    """Download a single dataset and save as parquet.

    Returns True on success, False on failure.
    """
    out_dir = DATA_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Downloading: {name} ({hf_repo})")
    print(f"  Subset: {hf_subset or 'default'}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")

    try:
        kwargs: dict = {}
        if hf_subset:
            kwargs["name"] = hf_subset
        if hf_split:
            kwargs["split"] = hf_split

        ds = load_dataset(hf_repo, **kwargs)

        # Save each split as a separate parquet file.
        if hasattr(ds, "keys"):
            # DatasetDict - multiple splits.
            for split_name in ds:
                out_path = out_dir / f"{split_name}.parquet"
                ds[split_name].to_parquet(str(out_path))
                print(f"  Saved {split_name}: {len(ds[split_name])} rows -> {out_path}")
        else:
            # Single Dataset.
            out_path = out_dir / "data.parquet"
            ds.to_parquet(str(out_path))
            print(f"  Saved: {len(ds)} rows -> {out_path}")

        print(f"  Columns: {ds.column_names if hasattr(ds, 'column_names') else 'see splits'}")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def download_gdpval_files() -> bool:
    """Download GDPval reference and deliverable files in parallel.

    Reads the parquet to get URLs, then downloads each file into
    data/raw/gdpval/{reference_files,deliverable_files}/...
    Skips files that already exist on disk.
    """
    parquet_path = DATA_ROOT / "gdpval" / "train.parquet"
    if not parquet_path.exists():
        print("  GDPval parquet not found - skipping file download.")
        return False

    df = pd.read_parquet(parquet_path)
    base_dir = DATA_ROOT / "gdpval"

    # Collect all (url, local_relative_path) pairs.
    file_pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        for url, rel_path in zip(row["reference_file_urls"], row["reference_files"], strict=False):
            file_pairs.append((url, rel_path))
        for url, rel_path in zip(
            row["deliverable_file_urls"], row["deliverable_files"], strict=False
        ):
            file_pairs.append((url, rel_path))

    # Filter out already-downloaded files.
    to_download: list[tuple[str, Path]] = []
    skipped = 0
    for url, rel_path in file_pairs:
        local_path = base_dir / rel_path
        if local_path.exists():
            skipped += 1
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            to_download.append((url, local_path))

    total = len(file_pairs)
    print(f"  GDPval files: {total} total, {skipped} exist, {len(to_download)} to download")

    if not to_download:
        print("  All files already downloaded.")
        return True

    def _fetch(item: tuple[str, Path]) -> str | None:
        url, local_path = item
        try:
            urllib.request.urlretrieve(url, str(local_path))
            return None
        except Exception as e:
            return f"{local_path.name}: {e}"

    downloaded, failed = 0, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch, item): item for item in to_download}
        for future in as_completed(futures):
            err = future.result()
            if err:
                print(f"  FAILED: {err}")
                failed += 1
            else:
                downloaded += 1
            done = downloaded + failed
            pct = (skipped + done) * 100 // total
            print(f"\r  [{pct:3d}%] {skipped + done}/{total} files", end="", flush=True)

    print()
    print(f"  Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument(
        "--only",
        nargs="+",
        help="Only download these datasets (by local name).",
    )
    parser.add_argument(
        "--gdpval-files-only",
        action="store_true",
        help="Only download GDPval reference/deliverable files (skip parquet).",
    )
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if args.gdpval_files_only:
        print(f"{'=' * 60}")
        print("Downloading GDPval reference & deliverable files")
        print(f"{'=' * 60}")
        ok = download_gdpval_files()
        sys.exit(0 if ok else 1)

    targets = DATASETS
    if args.only:
        targets = [d for d in DATASETS if d[0] in args.only]
        if not targets:
            print(f"No matching datasets. Available: {[d[0] for d in DATASETS]}")
            sys.exit(1)

    results: dict[str, bool] = {}
    for name, repo, subset, split in targets:
        results[name] = download_one(name, repo, subset, split)

    # Download GDPval reference + deliverable files if gdpval was downloaded.
    if any(name == "gdpval" for name, *_ in targets):
        print(f"\n{'=' * 60}")
        print("Downloading GDPval reference & deliverable files")
        print(f"{'=' * 60}")
        gdpval_files_ok = download_gdpval_files()
        if not gdpval_files_ok:
            results["gdpval_files"] = False

    print(f"\n{'=' * 60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 60}")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name:20s} {status}")

    failed = [n for n, s in results.items() if not s]
    if failed:
        print(f"\nFailed downloads: {failed}")
        print("You may need to search for alternative dataset names on HuggingFace.")
        sys.exit(1)


if __name__ == "__main__":
    main()
