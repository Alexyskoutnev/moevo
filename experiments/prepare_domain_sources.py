"""Download pinned official benchmark repositories for the seven-domain suite."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

REPOSITORIES = {
    "scienceagentbench": "OSU-NLP-Group/ScienceAgentBench",
    "minif2f_v2": "roozbeh-mohit/miniF2F_v2",
    "travelplanner": "OSU-NLP-Group/TravelPlanner",
    "optibench": "yangzhch6/ReSocratic",
    "datascibench": "THUDM/DataSciBench",
    "scienceworld": "allenai/ScienceWorld",
    "bizfinbench": "HiThink-Research/BizFinBench",
    "putnambench": "trishullab/PutnamBench",
    "spreadsheetbench": "RUCKBReasoning/SpreadsheetBench",
    "spreadsheetbench2": "RUCKBReasoning/SpreadsheetBench-2",
    "bizfinbench2": "HiThink-Research/BizFinBench.v2",
    "scicode": "scicode-bench/SciCode",
    "scicode_verified": "flyingwagner/scicode-verified",
    "oragentbench": "ORAgentBench/ORAgentBench",
    "automationbench": "zapier/AutomationBench",
    "terminal_bench_science": "harbor-framework/terminal-bench-science",
    "terminal_bench_2": "harbor-framework/terminal-bench-2",
    "dsbench": "LiqiangJing/DSBench",
    "amo_bench": "meituan-longcat/AMO-Bench",
    "harvey_lab": "harveyai/harvey-labs",
    "tau3_bench": "sierra-research/tau2-bench",
    "simple_evals": "openai/simple-evals",
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "moevo-research"}), timeout=120
    ) as response:
        return response.read()


def prepare(name: str, root: Path) -> dict:
    target = root / name
    manifest_path = target / "source_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    repo = REPOSITORIES[name]
    info = json.loads(fetch(f"https://api.github.com/repos/{repo}"))
    revision = json.loads(
        fetch(f"https://api.github.com/repos/{repo}/commits/{info['default_branch']}")
    )["sha"]
    data = fetch(f"https://api.github.com/repos/{repo}/tarball/{revision}")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{name}-{revision}.tar.gz"
    archive.write_bytes(data)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        top = members[0].name.split("/")[0]
        for member in members:
            if member.name == top:
                continue
            member.name = member.name.removeprefix(top + "/")
            # One upstream archive accidentally includes a local virtualenv.
            # Never restore machine-specific environments or archive links.
            if "venv" in Path(member.name).parts or ".venv" in Path(member.name).parts:
                continue
            if member.issym() or member.islnk():
                continue
            handle.extract(member, target, filter="data")
    result = {
        "dataset": name,
        "repository": f"https://github.com/{repo}",
        "revision": revision,
        "archive_sha256": hashlib.sha256(data).hexdigest(),
        "source_downloaded": True,
        "e2e_validated": False,
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=REPOSITORIES, default=list(REPOSITORIES))
    parser.add_argument("--root", type=Path, default=Path("data/external"))
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(prepare, name, args.root): name for name in args.only}
        failed = False
        for future in concurrent.futures.as_completed(jobs):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as exc:
                failed = True
                print(json.dumps({"dataset": jobs[future], "error": str(exc)}), flush=True)
    raise SystemExit(1 if failed else 0)
