#!/usr/bin/env python3
"""Record or verify SHA-256 hashes of everything EXP-009 must leave untouched.

``--record`` hashes every file tracked at the base commit, plus digests of the
untracked PSC JSON cache and HTRU1 batches, into a manifest. It refuses to
overwrite an existing manifest. The default mode re-hashes and reports any
difference, exiting non-zero on a mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments" / "exp009_protected_hashes.sha256"
BASE_COMMIT = "291a80a7b750d699abdcc66677206e771269cc18"
DIGEST_DIRS = {
    "digest:data/psc/raw": REPO_ROOT / "data" / "psc" / "raw",
    "digest:data/raw/htru1-batches-py": REPO_ROOT / "data" / "raw" / "htru1-batches-py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_directory(path: Path) -> str:
    """Hash of sorted ``relative_name file_hash`` lines, so renames and edits show."""
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file() and p.name != ".DS_Store"):
        line = f"{file.relative_to(path).as_posix()} {sha256_file(file)}\n"
        digest.update(line.encode())
    return digest.hexdigest()


def tracked_at_base(root: Path = REPO_ROOT) -> list[str]:
    output = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", BASE_COMMIT],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line]


def compute_hashes(root: Path = REPO_ROOT) -> dict[str, str]:
    hashes = {name: sha256_file(root / name) for name in tracked_at_base(root)}
    for key, path in DIGEST_DIRS.items():
        hashes[key] = digest_directory(path)
    return hashes


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        value, name = line.split("  ", 1)
        entries[name] = value
    return entries


def compare(recorded: dict[str, str], current: dict[str, str]) -> list[str]:
    problems = []
    for name, value in sorted(recorded.items()):
        if name not in current:
            problems.append(f"missing: {name}")
        elif current[name] != value:
            problems.append(f"changed: {name}")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--record", action="store_true", help="Create the manifest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.record:
        hashes = compute_hashes()
        with open(args.manifest, "x") as f:
            for name, value in sorted(hashes.items()):
                f.write(f"{value}  {name}\n")
        print(f"Recorded {len(hashes)} hashes at {args.manifest}")
        return 0

    recorded = read_manifest(args.manifest)
    current = {}
    for name in recorded:
        if name in DIGEST_DIRS:
            current[name] = digest_directory(DIGEST_DIRS[name])
        elif (REPO_ROOT / name).is_file():
            current[name] = sha256_file(REPO_ROOT / name)
    problems = compare(recorded, current)
    if problems:
        print("\n".join(problems))
        print(f"FAILED: {len(problems)} of {len(recorded)} protected entries differ")
        return 1
    print(f"OK: all {len(recorded)} protected entries are unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
