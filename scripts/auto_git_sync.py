#!/usr/bin/env python3
"""
Poll the current Git repository and automatically commit/push code changes.

This is intended for local VSCode use. It respects .gitignore, strips notebook
outputs before committing, and avoids syncing ignored data/results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


def log(message: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def git_root() -> Path:
    proc = run(["git", "rev-parse", "--show-toplevel"], Path.cwd())
    return Path(proc.stdout.strip())


def has_changes(repo: Path) -> bool:
    proc = run(["git", "status", "--porcelain"], repo)
    return bool(proc.stdout.strip())


def list_syncable_notebooks(repo: Path) -> list[Path]:
    proc = run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.ipynb",
        ],
        repo,
    )
    names = [name for name in proc.stdout.split("\0") if name]
    return [repo / name for name in names if not Path(name).name.startswith("._")]


def strip_notebook_outputs(path: Path) -> bool:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"skip notebook output strip for {path}: {exc}")
        return False

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def strip_all_notebooks(repo: Path) -> None:
    changed = []
    for path in list_syncable_notebooks(repo):
        if strip_notebook_outputs(path):
            changed.append(path.relative_to(repo))
    if changed:
        log("stripped notebook outputs: " + ", ".join(map(str, changed)))


def sync_once(repo: Path, message_prefix: str) -> bool:
    if not has_changes(repo):
        return False

    strip_all_notebooks(repo)
    run(["git", "add", "-A"], repo)

    staged = run(["git", "diff", "--cached", "--name-only"], repo).stdout.strip()
    if not staged:
        log("no staged changes after cleanup")
        return False

    message = f"{message_prefix} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    run(["git", "commit", "-m", message], repo)
    run(["git", "push"], repo)
    log("pushed commit: " + message)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=30.0, help="Polling interval in seconds")
    parser.add_argument("--quiet", type=float, default=10.0, help="Wait this long after first detecting changes")
    parser.add_argument("--once", action="store_true", help="Sync once and exit")
    parser.add_argument("--message-prefix", default="Auto sync", help="Commit message prefix")
    args = parser.parse_args()

    repo = git_root()
    log(f"auto git sync started: repo={repo}")
    log("ignored files such as data/, analysis_results/, logs/, ._* are not synced")

    if args.once:
        changed = sync_once(repo, args.message_prefix)
        return 0 if changed else 0

    while True:
        try:
            if has_changes(repo):
                log(f"changes detected; waiting {args.quiet:g}s for edits to settle")
                time.sleep(args.quiet)
                sync_once(repo, args.message_prefix)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            log("auto git sync stopped")
            return 0
        except subprocess.CalledProcessError as exc:
            log(f"git command failed: {' '.join(exc.cmd)}")
            if exc.stdout:
                log("stdout: " + exc.stdout.strip())
            if exc.stderr:
                log("stderr: " + exc.stderr.strip())
            time.sleep(args.interval)
        except Exception as exc:
            log(f"unexpected error: {exc}")
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
