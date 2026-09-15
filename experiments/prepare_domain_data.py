"""Fetch public research data from a pinned official source. No inference APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import urllib.request
import zipfile
from pathlib import Path

HF_REPOSITORIES = {
    "travelplanner": "osunlp/TravelPlanner",
    "datascibench": "zd21/DataSciBench",
    "bizfinbench2": "HiThink-Research/BizFinBench.v2",
    "gdpval": "openai/gdpval",
    "genebench_pro": "openai/genebench-pro-public-package",
    "scienceagentbench": "osunlp/ScienceAgentBench",
    "scicode_verified": "shhu2001/SciCode-Verified",
    "spreadsheetbench2": "KAKA22/SpreadsheetBench-v2",
    "dabstep": "adyen/DABstep",
    "dsbench": "liqiang888/DSBench",
    "amo_bench": "meituan-longcat/AMO-Bench",
    "healthbench_professional": "openai/healthbench-professional",
    "eduagentbench": "eduagentbench/eduagentbench",
}


def prepare_assets(name: str, root: Path) -> None:
    """Unpack the authors' public input bundles, recording the exact archive."""
    if name == "dsbench":
        archive = root / "data_analysis/data.zip"
    elif name == "travelplanner":
        import gdown

        root.mkdir(parents=True, exist_ok=True)
        archive = root / "database.zip"
        if not archive.exists():
            # Official TravelPlanner README's public database bundle.
            gdown.download(
                id="1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE",
                output=str(archive),
                quiet=False,
                use_cookies=False,
            )
    else:
        raise ValueError("No separate asset bundle configured for " + name)
    target = root / "processed"
    target.mkdir(parents=True, exist_ok=True)
    files = []
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive path")
            if "__MACOSX" in path.parts or path.name == ".DS_Store":
                continue
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Unexpected symlink in input archive")
            zipped.extract(item, target)
            if not item.is_dir():
                files.append(item.filename)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    manifest = {"archive": str(archive), "sha256": digest, "files": files, "e2e_validated": False}
    (root / "_moevo_assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"dataset": name, "asset_files": len(files), "archive_sha256": digest}))


def prepare_hf(name: str, root: Path, list_only: bool, patterns: list[str] | None) -> None:
    from huggingface_hub import HfApi, snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "_moevo_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    repo = HF_REPOSITORIES[name]
    info = HfApi(token=False).dataset_info(
        repo, revision=old.get("revision", "main"), files_metadata=True
    )
    if info.siblings is None:
        raise ValueError("Dataset metadata did not contain a file inventory")
    inventory = [{"path": f.rfilename, "bytes": f.size} for f in info.siblings]
    manifest = {
        "dataset": name,
        "repository": f"https://huggingface.co/datasets/{repo}",
        "revision": info.sha,
        "inventory": inventory,
    }
    (root / "inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if list_only:
        print(
            json.dumps(
                {
                    "dataset": name,
                    "files": len(inventory),
                    "bytes": sum(f["bytes"] or 0 for f in inventory),
                    "inventory": str(root / "inventory.json"),
                    "sample": inventory[:12],
                }
            )
        )
        return
    snapshot_download(
        repo,
        repo_type="dataset",
        revision=info.sha,
        token=False,
        local_dir=root,
        cache_dir=root / ".hf-cache",
        allow_patterns=patterns,
        max_workers=4,
    )
    manifest["files"] = {}
    for item in inventory:
        path = root / item["path"]
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest["files"][item["path"]] = digest.hexdigest()
    manifest["allow_patterns"] = patterns
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"dataset": name, "files_downloaded": len(manifest["files"]), "revision": info.sha}
        )
    )


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "moevo-research"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def prepare_finqa(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        revision = json.loads(manifest_path.read_text())["revision"]
    else:
        revision = json.loads(fetch("https://api.github.com/repos/czyssrs/FinQA/commits/main"))[
            "sha"
        ]
    base = f"https://raw.githubusercontent.com/czyssrs/FinQA/{revision}"
    manifest = {
        "dataset": "FinQA",
        "repository": "https://github.com/czyssrs/FinQA",
        "revision": revision,
        "files": {},
    }
    for filename in [
        "dataset/train.json",
        "dataset/dev.json",
        "dataset/test.json",
        "README.md",
        "code/evaluate/evaluate.py",
    ]:
        data = fetch(f"{base}/{filename}")
        output = root / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        manifest["files"][filename] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for split in ["train", "dev", "test"]:
        print(
            f"FinQA {split}: {len(json.loads((root / 'dataset' / f'{split}.json').read_text()))} tasks"
        )
    print(f"Pinned source: {revision}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["finqa", *HF_REPOSITORIES], default="finqa")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--include", nargs="+")
    parser.add_argument("--assets", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root or Path("data/raw") / arguments.dataset
    if arguments.assets:
        prepare_assets(arguments.dataset, root)
    elif arguments.dataset == "finqa":
        prepare_finqa(root)
    else:
        prepare_hf(arguments.dataset, root, arguments.list_only, arguments.include)
