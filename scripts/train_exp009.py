#!/usr/bin/env python3
"""Train one EXP-009 run, or a named plan of runs, on the frozen source-aware folds.

Every arm uses the EXP-008 recipe: ``ThreeClassCNN``, Adam at lr 1e-3, batch size
32, 40 epochs, class-weighted cross-entropy, and the best epoch by validation
accuracy. Only the data split, initialization, and inputs differ between arms.

  E1  HTRU1 initialization, fine-tuned on both surveys
  E2  random initialization, both surveys
  E3  HTRU1 convolutional layers frozen, both surveys
  C1  random initialization, phase-coherence control inputs
  L1  HTRU1 initialization, trained on one survey, tested on the other
  L2  random initialization, trained on one survey, tested on the other

Label-uncertain rows are excluded from training and epoch selection but are scored
for evaluation. Writes go only under ``outputs/exp009/runs/``, into a new directory
per run; an existing run makes this fail unless ``--skip-completed`` finds it
finished. ``--dry-run`` validates a plan and writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.psc_controls import row_shuffled_tensor
from src.datasets.psc_dataset import class_weights
from src.datasets.psc_folds import (
    assert_under,
    attach_identity,
    fold_vector,
    load_candidate_records,
    load_folds,
    read_identity,
    rotation,
    sha256_file,
    shared_sources,
    write_exclusive,
)
from src.datasets.psc_preprocessing import arrays_to_tensor
from src.models.three_class_head import CLASS_NAMES, CLASS_TO_INDEX, NUM_CLASSES, ThreeClassCNN

DEFAULT_MANIFEST = REPO_ROOT / "data" / "psc" / "psc_eval_multisurvey.csv"
DEFAULT_CACHE = REPO_ROOT / "data" / "psc" / "raw"
DEFAULT_IDENTITY = REPO_ROOT / "data" / "psc" / "identity" / "source_identity_v1.csv"
DEFAULT_FOLDS = REPO_ROOT / "data" / "psc" / "folds" / "folds_v1.json"
DEFAULT_PRETRAINED = REPO_ROOT / "outputs" / "models" / "baseline_cnn_best.pt"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "exp009" / "runs"
PRETRAINED_SHA256 = "7f852a3e7fc46ade884c32a4a5599cc4c2734d0fd262a2c17620a4f6e4695d5c"

EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
BASE_SEED = 42
ARMS = {
    "E1": {"init": "htru1", "freeze": False, "control": False, "cross": False},
    "E2": {"init": "random", "freeze": False, "control": False, "cross": False},
    "E3": {"init": "htru1", "freeze": True, "control": False, "cross": False},
    "C1": {"init": "random", "freeze": False, "control": True, "cross": False},
    "L1": {"init": "htru1", "freeze": False, "control": False, "cross": True},
    "L2": {"init": "random", "freeze": False, "control": False, "cross": True},
}
DIRECTIONS = (("ao327", "gbncc"), ("gbncc", "ao327"))


@dataclass(frozen=True)
class RunSpec:
    arm: str
    assignment: int
    fold: int | None = None
    source_survey: str | None = None
    target_survey: str | None = None

    @property
    def model_seed(self) -> int:
        offset = self.fold if self.fold is not None else 10 + DIRECTIONS.index((self.source_survey, self.target_survey))
        return BASE_SEED + 100 * self.assignment + offset

    @property
    def run_id(self) -> str:
        where = (
            f"f{self.fold}"
            if self.fold is not None
            else f"{self.source_survey}_to_{self.target_survey}"
        )
        return f"{self.arm}_a{self.assignment}_{where}_s{self.model_seed}"


def within_specs(arms: tuple[str, ...], assignments: tuple[int, ...]) -> list[RunSpec]:
    return [RunSpec(arm, a, fold=k) for arm in arms for a in assignments for k in range(5)]


def cross_specs(arms: tuple[str, ...], assignments: tuple[int, ...]) -> list[RunSpec]:
    return [
        RunSpec(arm, a, source_survey=src, target_survey=dst)
        for arm in arms
        for a in assignments
        for src, dst in DIRECTIONS
    ]


def plan_specs(name: str) -> list[RunSpec]:
    """The pre-specified plans: pilot (24), extension (38), and E3 completion (10)."""
    if name == "pilot":
        return within_specs(("E1", "E2", "E3", "C1"), (0,)) + cross_specs(("L1", "L2"), (0,))
    if name == "extension":
        return within_specs(("E1", "E2", "C1"), (1, 2)) + cross_specs(("L1", "L2"), (1, 2))
    if name == "e3-completion":
        return within_specs(("E3",), (1, 2))
    raise ValueError(f"unknown plan {name!r}")


def split_rows(spec: RunSpec, records: list[dict], folds: list[int]) -> dict[str, np.ndarray]:
    """Row indices for one run.

    ``val`` selects the epoch and excludes uncertain labels; ``val_scored`` and
    ``test`` keep them, because their labels have not been overturned.
    """
    n = len(records)
    fold = np.array(folds)
    uncertain = np.array([r["label_uncertain"] for r in records])
    if spec.fold is not None:
        parts = rotation(spec.fold)
        train = np.isin(fold, parts["train"]) & ~uncertain
        val_scored = np.isin(fold, parts["val"])
        test = np.isin(fold, parts["test"])
    else:
        shared = shared_sources(records)
        keep = np.array([r["source_id"] not in shared for r in records])
        survey = np.array([r["survey"] for r in records])
        in_source = (survey == spec.source_survey) & keep
        val_scored = in_source & (fold == 0)
        train = in_source & (fold != 0) & ~uncertain
        test = (survey == spec.target_survey) & keep
    return {
        "train": np.where(train)[0],
        "val": np.where(val_scored & ~uncertain)[0],
        "val_scored": np.where(val_scored)[0],
        "test": np.where(test)[0],
    }


def check_split(splits: dict[str, np.ndarray], records: list[dict]) -> None:
    """Fail unless train, validation, and test are disjoint by row, source, and observation."""
    train, val, test = (set(splits[k].tolist()) for k in ("train", "val_scored", "test"))
    if train & val or train & test or val & test:
        raise RuntimeError("split rows overlap")
    for key in ("source_id", "obs"):
        seen = {name: {records[i][key] for i in rows if records[i][key]} for name, rows in (("train", train), ("val", val), ("test", test))}
        if seen["train"] & seen["test"] or seen["train"] & seen["val"] or seen["val"] & seen["test"]:
            raise RuntimeError(f"{key} crosses splits")
    labels = {records[i]["category"] for i in train}
    missing = set(CLASS_NAMES) - labels
    if missing:
        raise RuntimeError(f"training split lacks classes: {sorted(missing)}")


def load_images(records: list[dict], cache_dir: Path, control: bool) -> torch.Tensor:
    images = []
    for rec in records:
        with open(cache_dir / f"{rec['plot_id']}.json") as f:
            payload = json.load(f)
        fvp = np.asarray(payload["FvP"], dtype=float)
        tvp = np.asarray(payload["TvP"], dtype=float)
        images.append(row_shuffled_tensor(fvp, tvp, rec["plot_id"]) if control else arrays_to_tensor(fvp, tvp))
    return torch.stack(images)


def build_model(spec: RunSpec, pretrained_state: dict | None) -> tuple[ThreeClassCNN, dict]:
    settings = ARMS[spec.arm]
    model = ThreeClassCNN()
    info = {"loaded": [], "skipped": []}
    if settings["init"] == "htru1":
        if pretrained_state is None:
            raise ValueError(f"arm {spec.arm} needs the HTRU1 checkpoint")
        info = model.load_pretrained_backbone(pretrained_state)
    if settings["freeze"]:
        model.freeze_backbone()
    return model, info


@torch.no_grad()
def _logits(model: nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    return torch.cat([model(batch.to(device)).cpu() for batch in images.split(BATCH_SIZE)])


def train_run(
    spec: RunSpec,
    images: torch.Tensor,
    labels: torch.Tensor,
    splits: dict[str, np.ndarray],
    pretrained_state: dict | None,
    device: torch.device,
    epochs: int = EPOCHS,
) -> dict:
    """Train with the fixed recipe; return the best state, epoch log, and logits."""
    torch.manual_seed(spec.model_seed)
    model, info = build_model(spec, pretrained_state)
    model.to(device)

    train_images, train_labels = images[splits["train"]], labels[splits["train"]]
    val_images, val_labels = images[splits["val"]], labels[splits["val"]]
    weights = class_weights(train_labels.tolist(), NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(spec.model_seed)
    loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    log = []
    best_accuracy, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), target)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        train_accuracy = float((_logits(model, train_images, device).argmax(1) == train_labels).float().mean())
        val_accuracy = float((_logits(model, val_images, device).argmax(1) == val_labels).float().mean())
        log.append({"epoch": epoch, "train_loss": total / max(batches, 1), "train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {
        "state": best_state,
        "log": log,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "load_info": info,
        "val_logits": _logits(model, images[splits["val_scored"]], device),
        "test_logits": _logits(model, images[splits["test"]], device),
    }


def render_epochs(log: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["epoch", "train_loss", "train_accuracy", "val_accuracy"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(log)
    return buffer.getvalue()


def render_scores(records: list[dict], splits: dict[str, np.ndarray], result: dict) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["plot_id", "split", *[f"logit_{c}" for c in CLASS_NAMES], "p_pulsar"])
    for name, rows, logits in (("val", splits["val_scored"], result["val_logits"]), ("test", splits["test"], result["test_logits"])):
        probabilities = torch.softmax(logits, dim=1)[:, CLASS_TO_INDEX["known_pulsar"]]
        for i, row_logits, p in zip(rows.tolist(), logits.tolist(), probabilities.tolist()):
            writer.writerow([records[i]["plot_id"], name, *[f"{x:.6f}" for x in row_logits], f"{p:.8f}"])
    return buffer.getvalue()


def git_state(root: Path = REPO_ROOT) -> dict:
    """Commit and whether tracked code differs. Run outputs do not count as dirty."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout
    dirty = [line for line in status.splitlines() if "outputs/exp009/" not in line]
    return {"commit": commit, "dirty": bool(dirty), "dirty_entries": dirty}


def write_run(
    run_dir: Path,
    root: Path,
    spec: RunSpec,
    records: list[dict],
    splits: dict[str, np.ndarray],
    result: dict,
    provenance: dict,
) -> None:
    """Write one run. ``metadata.json`` comes last, as the completion marker."""
    assert_under(run_dir, root)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    with open(run_dir / "best.pt", "xb") as f:
        torch.save({"model_state_dict": result["state"], "class_names": list(CLASS_NAMES), "run_id": spec.run_id}, f)
    write_exclusive(run_dir / "epochs.csv", render_epochs(result["log"]))
    write_exclusive(run_dir / "scores.csv", render_scores(records, splits, result))
    metadata = {
        **asdict(spec),
        "run_id": spec.run_id,
        "model_seed": spec.model_seed,
        "recipe": {"epochs": len(result["log"]), "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "optimizer": "Adam", "selection": "validation accuracy"},
        "best_epoch": result["best_epoch"],
        "best_val_accuracy": result["best_val_accuracy"],
        "load_info": result["load_info"],
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        **provenance,
    }
    write_exclusive(run_dir / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_pretrained(path: Path, device: torch.device) -> dict:
    actual = sha256_file(path)
    if actual != PRETRAINED_SHA256:
        raise ValueError(f"{path} hash {actual} is not the EXP-001 checkpoint {PRETRAINED_SHA256}")
    return torch.load(path, map_location=device, weights_only=True)["model_state_dict"]


def describe(spec: RunSpec, records: list[dict], splits: dict[str, np.ndarray]) -> str:
    def part(name: str) -> str:
        rows = splits[name]
        counts = [sum(records[i]["category"] == c for i in rows) for c in CLASS_NAMES]
        sources = len({records[i]["source_id"] for i in rows if records[i]["source_id"]})
        return f"{name}={len(rows)} ({'/'.join(map(str, counts))}, {sources} src)"

    return f"  {spec.run_id:32s} {part('train')} {part('val_scored')} {part('test')}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", choices=("pilot", "extension", "e3-completion"))
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--assignment", type=int, choices=(0, 1, 2))
    parser.add_argument("--fold", type=int, choices=range(5), help="Test fold, for E1, E2, E3, C1")
    parser.add_argument("--source-survey", choices=("ao327", "gbncc"), help="Training survey, for L1, L2")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan; train and write nothing")
    parser.add_argument("--skip-completed", action="store_true", help="Skip runs that finished earlier")
    parser.add_argument("--allow-dirty", action="store_true", help="Train despite uncommitted code; recorded in metadata")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    return parser.parse_args(argv)


def selected_specs(args: argparse.Namespace) -> list[RunSpec]:
    if args.plan:
        if args.arm or args.fold is not None or args.source_survey:
            raise SystemExit("--plan cannot be combined with --arm, --fold, or --source-survey")
        return plan_specs(args.plan)
    if not args.arm or args.assignment is None:
        raise SystemExit("give --plan, or --arm with --assignment")
    if ARMS[args.arm]["cross"]:
        if not args.source_survey:
            raise SystemExit(f"{args.arm} needs --source-survey")
        target = "gbncc" if args.source_survey == "ao327" else "ao327"
        return [RunSpec(args.arm, args.assignment, source_survey=args.source_survey, target_survey=target)]
    if args.fold is None:
        raise SystemExit(f"{args.arm} needs --fold")
    return [RunSpec(args.arm, args.assignment, fold=args.fold)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = selected_specs(args)
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)

    state = git_state()
    print(f"commit {state['commit'][:10]}, uncommitted code: {state['dirty']}")
    if state["dirty"] and not args.dry_run and not args.allow_dirty:
        print("Refusing to train from uncommitted code. Commit first, or pass --allow-dirty.")
        return 1

    folds = load_folds(args.folds)
    identity_hash = sha256_file(args.identity)
    manifest_hash = sha256_file(args.manifest)
    if folds["identity_sha256"] != identity_hash or folds["manifest_sha256"] != manifest_hash:
        raise ValueError("folds were built from a different identity table or manifest")

    records = load_candidate_records(args.manifest, args.cache_dir)
    attach_identity(records, read_identity(args.identity))
    plot_ids = [r["plot_id"] for r in records]
    if plot_ids != folds["plot_ids"]:
        raise ValueError("fold file row order does not match the manifest")
    labels = torch.tensor([CLASS_TO_INDEX[r["category"]] for r in records], dtype=torch.long)

    needs_htru1 = any(ARMS[s.arm]["init"] == "htru1" for s in specs)
    pretrained_state = load_pretrained(args.pretrained, device) if needs_htru1 else None

    print(f"{len(specs)} run(s); rows shown as pulsar/rfi/noise")
    prepared = []
    for spec in specs:
        splits = split_rows(spec, records, fold_vector(folds, spec.assignment, plot_ids))
        check_split(splits, records)
        if ARMS[spec.arm]["init"] == "htru1":
            _, info = build_model(spec, pretrained_state)
            if not info["loaded"]:
                raise RuntimeError(f"{spec.run_id}: nothing loaded from the checkpoint")
        prepared.append((spec, splits))
        print(describe(spec, records, splits))

    if args.dry_run:
        controls = {name: load_images(records[:3], args.cache_dir, control=True) for name in ("first", "second")}
        if not torch.equal(controls["first"], controls["second"]):
            raise RuntimeError("control inputs are not deterministic")
        print(f"Output root: {OUTPUT_ROOT}")
        existing = [s.run_id for s, _ in prepared if (OUTPUT_ROOT / s.run_id).exists()]
        print(f"Existing run directories: {existing or 'none'}")
        print("Dry run passed. Nothing was trained or written.")
        return 0

    provenance = {
        "git": state,
        "allow_dirty": bool(args.allow_dirty),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "threads": args.threads,
        "manifest_sha256": manifest_hash,
        "identity_sha256": identity_hash,
        "folds_sha256": sha256_file(args.folds),
        "pretrained_sha256": PRETRAINED_SHA256,
    }
    images_by_control: dict[bool, torch.Tensor] = {}
    for spec, splits in prepared:
        run_dir = OUTPUT_ROOT / spec.run_id
        if run_dir.exists():
            if args.skip_completed and (run_dir / "metadata.json").is_file():
                print(f"{spec.run_id}: already complete, skipping")
                continue
            print(f"Refusing to overwrite {run_dir}. Inspect or move it before rerunning.")
            return 1
        control = ARMS[spec.arm]["control"]
        if control not in images_by_control:
            images_by_control[control] = load_images(records, args.cache_dir, control)
        started = time.perf_counter()
        result = train_run(spec, images_by_control[control], labels, splits, pretrained_state, device)
        run_provenance = {**provenance, "seconds": time.perf_counter() - started}
        write_run(run_dir, OUTPUT_ROOT, spec, records, splits, result, run_provenance)
        print(f"{spec.run_id}: best epoch {result['best_epoch']}, val accuracy {result['best_val_accuracy']:.4f}, {run_provenance['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
