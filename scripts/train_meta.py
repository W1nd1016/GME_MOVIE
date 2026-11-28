"""Run movie-level support/query diagnostics on a frozen MovieLens backbone."""
import argparse
import copy
import errno
import hashlib
import json
import math
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml

from coldstart.data.meta_tasks import load_meta_tasks, make_meta_split
from coldstart.models.base_dnn import BaseDNN
from coldstart.models.generators.attribute import MetaGenerator
from coldstart.train.meta import query_loss, score_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_backbone(base_run, holdout_path):
    metadata = json.loads((base_run / "metrics.json").read_text())
    holdout_hash = sha256(holdout_path)
    if (metadata["data_split"] != "base_train" or metadata["holdout_sha256"] != holdout_hash
            or sha256(base_run / "holdout.json") != holdout_hash):
        raise ValueError("backbone must be trained on the same verified holdout")
    weights = base_run / "weights.pt"
    backbone = BaseDNN()
    backbone.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    return backbone.requires_grad_(False).eval(), {
        "backbone_sha256": sha256(weights), "holdout_sha256": holdout_hash,
    }


def movie_batches(train_ids, seed, batch_size, movies_per_epoch, steps):
    """Give each group the same movie exposure and the same short final batch."""
    rng = np.random.Generator(np.random.PCG64(seed))
    step = 0
    while step < steps:
        permutation = rng.permutation(train_ids).tolist()
        epoch = (permutation * math.ceil(movies_per_epoch / len(permutation)))[:movies_per_epoch]
        for start in range(0, movies_per_epoch, batch_size):
            yield epoch[start:start + batch_size]
            step += 1
            if step == steps:
                return


def evaluate(backbone, generator, tasks, movie_ids, batch_size, inner_lr):
    """Average the query loss over movies, including the smaller final batch."""
    total_loss = 0.0
    for start in range(0, len(movie_ids), batch_size):
        batch = [tasks[mid] for mid in movie_ids[start:start + batch_size]]
        scores = score_tasks(backbone, generator, batch, inner_lr, training=False)
        total_loss += query_loss(scores, batch).item() * len(batch)
    return total_loss / len(movie_ids)


def train_cell(backbone, tasks, train_ids, validation_ids, config, movies_per_epoch):
    method, seed = config["method"], config["seed"]
    torch.manual_seed(seed)
    generator = MetaGenerator(backbone.embedding.embedding_dim, method)
    optimizer = torch.optim.Adam(generator.parameters(), lr=config["learning_rate"])
    inner_lr, batch_size = config["inner_lr"], config["batch_size"]
    steps = config["checkpoints"][-1]
    updates, checkpoints = [], []
    best_loss = math.inf
    for step, movie_ids in enumerate(movie_batches(train_ids, seed, batch_size, movies_per_epoch, steps), 1):
        batch = [tasks[mid] for mid in movie_ids]
        optimizer.zero_grad(set_to_none=True)
        scores = score_tasks(backbone, generator, batch, inner_lr, training=True)
        loss = query_loss(scores, batch)
        loss.backward()
        grad_norm = torch.linalg.vector_norm(torch.cat([
            parameter.grad.flatten() for parameter in generator.initial_parameters()
        ])).item()
        adaptation = (scores.adapted.detach() - scores.initial.detach()).norm(dim=1).mean().item()
        if not all(math.isfinite(x) for x in (loss.item(), grad_norm, adaptation)):
            raise ValueError(f"non-finite meta-training diagnostic at step {step}")
        updates.append({"movies": movie_ids, "query_logloss": loss.item(),
                        "initial_grad_norm": grad_norm, "mean_adaptation_l2": adaptation})
        optimizer.step()
        if step in config["checkpoints"]:
            train_loss, validation_loss = [
                evaluate(backbone, generator, tasks, ids, batch_size, inner_lr)
                for ids in (train_ids, validation_ids)
            ]
            if not all(map(math.isfinite, (train_loss, validation_loss))):
                raise ValueError(f"non-finite checkpoint loss at step {step}")
            checkpoints.append({"step": step, "train_query_logloss": train_loss,
                                "validation_query_logloss": validation_loss})
            if validation_loss < best_loss:
                best_loss, selected_step = validation_loss, step
                selected_weights = copy.deepcopy(generator.state_dict())
                probe = [tasks[validation_ids[0]]]
                selected_logits = score_tasks(
                    backbone, generator, probe, inner_lr, training=False
                ).query_logits.clone()
    return selected_weights, selected_logits, {
        "method": method, "seed": seed, "steps": steps,
        "movie_tasks": sum(len(update["movies"]) for update in updates),
        "selected_step": selected_step, "checkpoints": checkpoints, "updates": updates,
    }


def publish(staged, destination):
    """Publish atomically. A deterministic repeat can reuse the completed run."""
    try:
        staged.rename(destination)
        return
    except OSError as exc:
        if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise
    # The implementation commit can differ on a documentation-only rerun. Preserve
    # the original provenance only when the settings, evidence and weights agree.
    files = ("config.yaml", "movies.json", "weights.pt", "metrics.json")
    if all((destination / name).is_file() for name in files):
        old = json.loads((destination / "metrics.json").read_text())
        current = json.loads((staged / "metrics.json").read_text())
        old.pop("git_commit", None)
        current.pop("git_commit", None)
        if old == current and all((destination / n).read_bytes() == (staged / n).read_bytes()
                                  for n in files[:3]):
            return
    raise FileExistsError(f"existing run has different results or configuration: {destination}")


def run(config_path, run_root, *, smoke=False):
    config = yaml.safe_load(Path(config_path).read_text())
    batch_size = config["batch_size"]
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for name in ("learning_rate", "inner_lr"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    checkpoints = config["checkpoints"]
    if (not checkpoints or any(type(step) is not int or step <= 0 for step in checkpoints)
            or sorted(set(checkpoints)) != checkpoints):
        raise ValueError("checkpoints must be strictly increasing positive steps")
    seeds = config["seeds"]
    if not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("expected at least one 32-bit training seed")
    holdout_path, base_run = Path(config["holdout_path"]), Path(config["base_run"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    backbone, evidence = load_backbone(base_run, holdout_path)
    tasks = load_meta_tasks(config["data_root"], holdout_path)
    train, validation = make_meta_split(tasks, config["split_seed"])
    groups = config["groups"]
    if not groups or any(method not in ("attribute", "shared") or not sizes
                         for method, sizes in groups.items()):
        raise ValueError("expected attribute or shared groups with movie counts")
    for sizes in groups.values():
        if any(type(n) is not int or not batch_size <= n <= len(train) for n in sizes):
            raise ValueError("each movie count must fit a batch and the training split")
    evidence["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    common = {key: value for key, value in config.items() if key not in ("seeds", "groups")}
    cells = [(method, size, seed) for seed in seeds for method, sizes in groups.items() for size in sizes]
    if smoke:
        cells = [("attribute", len(train), seeds[0])]
        common["checkpoints"] = [1, 3]
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    for method, size, seed in cells:
        run_id = "0.2-smoke" if smoke else f"0.2-{method}-n{size}-s{seed}"
        settings = dict(common, method=method, train_movies=size, seed=seed)
        subset = train[:size]
        weights, expected, metrics = train_cell(backbone, tasks, subset, validation, settings, len(train))
        with TemporaryDirectory(dir=run_root) as temporary:
            staged = Path(temporary) / run_id
            staged.mkdir()
            torch.save(weights, staged / "weights.pt")
            restored = MetaGenerator(backbone.embedding.embedding_dim, method)
            restored.load_state_dict(torch.load(staged / "weights.pt", weights_only=True, map_location="cpu"))
            reloaded_base, _ = load_backbone(base_run, holdout_path)
            actual = score_tasks(reloaded_base, restored, [tasks[validation[0]]],
                                 settings["inner_lr"], training=False).query_logits
            if not torch.isfinite(actual).all() or not torch.equal(expected, actual):
                raise RuntimeError("reloading the selected weights did not reproduce query logits")
            metrics.update(evidence, checkpoint_reload_verified=True)
            replay_config = dict(common, groups={method: [size]}, seeds=[seed])
            (staged / "config.yaml").write_text(yaml.safe_dump(replay_config, sort_keys=False))
            (staged / "movies.json").write_text(json.dumps({"train": subset, "validation": validation}) + "\n")
            (staged / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
            publish(staged, run_root / run_id)
        print(f"{run_id}  steps={metrics['steps']}  selected_step={metrics['selected_step']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.run_root, smoke=args.smoke)
