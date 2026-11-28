"""Train the MovieLens base DNN on CPU using an eligible split."""
import argparse
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from coldstart.data.movielens import MovieLensDataset
from coldstart.data.schema import (
    FEATURE_WIDTH, GENRES_SLICE, TITLE_SLICE, VOCAB_SIZE, Purpose, Split, require_access,
)
from coldstart.models.base_dnn import BaseDNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def write_synthetic_dataset(root, rows, seed):
    rng = np.random.default_rng(seed)
    features = rng.integers(1, VOCAB_SIZE, (rows, FEATURE_WIDTH), dtype=np.int32)
    for token_slice in (TITLE_SLICE, GENRES_SLICE):
        tokens = features[:, token_slice]
        lengths = rng.integers(1, tokens.shape[1] + 1, size=(rows, 1))
        tokens[np.arange(tokens.shape[1]) >= lengths] = 0
    labels = rng.integers(0, 2, rows, dtype=np.int32)
    split = Split.SYNTHETIC_SMOKE.value
    np.savez(root / f"{split}.npz", label=labels, features=features)
    manifest = {"splits": {split: {"source": "synthetic", "rows": rows}}}
    (root / "manifest.json").write_text(json.dumps(manifest))


def load_training_data(data, split, seed):
    if split == Split.BASE_TRAIN:
        if not data.get("root") or not data.get("holdout_path") or "rows" in data:
            raise ValueError("base_train requires root and holdout_path and uses all retained rows")
        return MovieLensDataset(
            data["root"], split, Purpose.BASE_TRAIN, holdout_path=data["holdout_path"],
        )
    if "root" in data or "holdout_path" in data:
        raise ValueError("synthetic_smoke is generated internally and cannot take a root or holdout_path")
    with TemporaryDirectory() as temp:
        root = Path(temp)
        write_synthetic_dataset(root, positive_int(data["rows"], "data.rows"), seed)
        return MovieLensDataset(root, split, Purpose.BASE_TRAIN)


def train(config_path, run_dir):
    config_bytes = Path(config_path).read_bytes()
    config = yaml.safe_load(config_bytes)
    data = config["data"]
    split = Split(data["split"])
    require_access(split, Purpose.BASE_TRAIN)

    seed = config["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    batch_size = positive_int(config["batch_size"], "batch_size")
    budget_key = "epochs" if split == Split.BASE_TRAIN else "steps"
    if {"epochs", "steps"}.intersection(config) != {budget_key}:
        raise ValueError(f"{split.value} requires {budget_key} as its only budget field (epochs or steps)")
    budget = positive_int(config[budget_key], budget_key)
    learning_rate = float(config["learning_rate"])
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset = load_training_data(data, split, seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    steps = budget * len(loader) if budget_key == "epochs" else budget
    batches = iter(loader)
    model = BaseDNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    loss_history = []
    samples_seen = 0
    loss_sum = 0.0
    for _ in range(steps):
        try:
            features, labels = next(batches)
        except StopIteration:
            batches = iter(loader)
            features, labels = next(batches)
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        batch_loss = loss.item()
        if not math.isfinite(batch_loss):
            raise RuntimeError("non-finite training BCE")
        loss.backward()
        optimizer.step()
        loss_history.append(batch_loss)
        samples_seen += len(labels)
        loss_sum += batch_loss * len(labels)

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=run_dir.parent) as staging:
        output = Path(staging)
        weights = output / "weights.pt"
        torch.save(model.state_dict(), weights)
        restored = BaseDNN()
        restored.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
        with torch.no_grad():
            expected = model.eval()(features)
            reloaded = restored.eval()(features)
        if not torch.isfinite(expected).all() or not torch.equal(expected, reloaded):
            raise RuntimeError("checkpoint reload did not reproduce finite logits")

        metrics = {
            "data_split": split.value,
            "seed": seed,
            "git_commit": git_commit,
            "steps": steps,
            "loss": loss_history,
            "rows": len(dataset),
            "samples_seen": samples_seen,
            "train_bce": loss_sum / samples_seen,
            "checkpoint_reload_verified": True,
        }
        if split == Split.BASE_TRAIN:
            metrics["epochs"] = budget
            metrics["holdout_sha256"] = hashlib.sha256(dataset.holdout_bytes).hexdigest()
            (output / "holdout.json").write_bytes(dataset.holdout_bytes)
        (output / "config.yaml").write_bytes(config_bytes)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        output.rename(run_dir)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = train(args.config, args.run_dir)
    if metrics["data_split"] == Split.BASE_TRAIN.value:
        metrics = {name: value for name, value in metrics.items() if name != "loss"}
    print(json.dumps(metrics, indent=2))
