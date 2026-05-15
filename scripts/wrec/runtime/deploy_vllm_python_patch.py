#!/usr/bin/env python3
"""Deploy WREC's Python-only vLLM patch into an installed vLLM package.

The local vLLM source tree cannot be imported directly unless the compiled
extensions match the environment. This script keeps site-packages deployment
reproducible by copying a fixed set of Python files, backing up the previous
installed files, and writing a compact manifest for each deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RELATIVE_FILES = (
    "vllm/model_executor/layers/attention/attention.py",
    "vllm/model_executor/layers/attention/mla_attention.py",
    "vllm/model_executor/layers/fused_moe/layer.py",
    "vllm/model_executor/layers/fused_moe/wrec_expert_residency.py",
    "vllm/model_executor/offloader/prefetch.py",
    "vllm/model_executor/offloader/uva.py",
    "vllm/model_executor/offloader/wrec_residency.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/sched/wrec_sidecar_client.py",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def resolve_vllm_package_dir(python: Path) -> Path:
    code = (
        "from pathlib import Path\n"
        "import vllm\n"
        "print(Path(vllm.__file__).resolve().parent)\n"
    )
    return Path(run_checked([str(python), "-c", code])).resolve()


def git_identity(source_root: Path) -> dict[str, str | None]:
    git_dir = source_root / ".git"
    if not git_dir.exists():
        return {"commit": None, "status_short": None}
    try:
        commit = run_checked(["git", "rev-parse", "HEAD"], source_root)
        status = run_checked(["git", "status", "--short"], source_root)
    except subprocess.CalledProcessError:
        return {"commit": None, "status_short": None}
    return {"commit": commit, "status_short": status}


def source_and_destination(
    source_root: Path,
    package_dir: Path,
    relative_file: str,
) -> tuple[Path, Path]:
    source = source_root / relative_file
    destination = package_dir.parent / relative_file
    return source, destination


def validate_sources(source_root: Path, package_dir: Path) -> None:
    missing = []
    for relative_file in RELATIVE_FILES:
        source, _ = source_and_destination(source_root, package_dir,
                                           relative_file)
        if not source.is_file():
            missing.append(str(source))
    if missing:
        raise SystemExit("Missing source files:\n" + "\n".join(missing))


def compile_destinations(python: Path, files: list[Path]) -> None:
    cmd = [str(python), "-m", "py_compile", *[str(path) for path in files]]
    run_checked(cmd)


def deploy(args: argparse.Namespace) -> Path | None:
    source_root = args.source_root.resolve()
    python = args.python.resolve()
    package_dir = resolve_vllm_package_dir(python)
    validate_sources(source_root, package_dir)

    stamp = utc_stamp()
    backup_dir = args.backup_root.resolve() / f"wrec_vllm_sitepkg_backup_{stamp}"
    manifest_dir = args.manifest_dir.resolve()
    manifest_path = manifest_dir / f"vllm_python_patch_deploy_{stamp}.json"

    files: list[dict[str, Any]] = []
    destination_paths: list[Path] = []
    for relative_file in RELATIVE_FILES:
        source, destination = source_and_destination(source_root, package_dir,
                                                    relative_file)
        backup = backup_dir / relative_file
        existed = destination.exists()
        entry = {
            "relative_file": relative_file,
            "source": str(source),
            "destination": str(destination),
            "backup": str(backup) if existed else None,
            "destination_existed": existed,
            "source_sha256": sha256_file(source),
            "previous_destination_sha256": (
                sha256_file(destination) if existed else None
            ),
        }
        files.append(entry)
        destination_paths.append(destination)

    manifest: dict[str, Any] = {
        "created_utc": stamp,
        "mode": "dry_run" if args.dry_run else "deploy",
        "source_root": str(source_root),
        "target_python": str(python),
        "vllm_package_dir": str(package_dir),
        "backup_dir": str(backup_dir),
        "git": git_identity(source_root),
        "files": files,
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return None

    for entry in files:
        source = Path(entry["source"])
        destination = Path(entry["destination"])
        if entry["destination_existed"]:
            backup = Path(entry["backup"])
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entry["deployed_destination_sha256"] = sha256_file(destination)

    compile_destinations(python, destination_paths)
    manifest["py_compile"] = "ok"

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"deployed_files={len(files)}")
    print(f"manifest={manifest_path}")
    print(f"backup_dir={backup_dir}")
    return manifest_path


def restore(args: argparse.Namespace) -> None:
    manifest_path = args.restore_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    python = Path(manifest["target_python"])
    restored: list[Path] = []
    skipped_created: list[str] = []

    for entry in manifest["files"]:
        destination = Path(entry["destination"])
        if entry["destination_existed"]:
            backup = Path(entry["backup"])
            if not backup.is_file():
                raise SystemExit(f"Missing backup file: {backup}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, destination)
            restored.append(destination)
        else:
            skipped_created.append(entry["relative_file"])

    if restored:
        compile_destinations(python, restored)
    print(f"restored_files={len(restored)}")
    if skipped_created:
        print("left_created_files=" + ",".join(skipped_created))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter for the target vLLM environment.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/root/WREC/external/vllm-0.19.0"),
        help="Root of the WREC-modified vLLM source tree.",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("/tmp"),
        help="Directory where timestamped site-packages backups are written.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("/root/WREC/results/wrec/runtime_patch_deploy"),
        help="Directory where deployment manifests are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned copy set without changing files.",
    )
    parser.add_argument(
        "--restore-manifest",
        type=Path,
        help="Restore backed-up files from a previous deployment manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.restore_manifest:
        restore(args)
    else:
        deploy(args)


if __name__ == "__main__":
    main()
