from dataclasses import replace

import pytest
import torch
from torch import nn

from coldstart.data.meta_tasks import MovieTask, load_meta_tasks
from coldstart.data.schema import FEATURE_WIDTH, MOVIE_INDEX, USER_INDEX, YEAR_INDEX
from coldstart.models.base_dnn import BaseDNN
from coldstart.models.generators.attribute import MetaGenerator
from coldstart.train.meta import query_loss, score_tasks


def example_task(movie=50):
    support = torch.zeros((20, FEATURE_WIDTH), dtype=torch.long)
    support[:, MOVIE_INDEX] = movie
    support[:, USER_INDEX] = torch.arange(1, 21)
    support[:, YEAR_INDEX] = 60
    query = support.clone()
    query[:, USER_INDEX] += 20
    return MovieTask(movie_id=movie, attributes=support[0, YEAR_INDEX:].clone(),
                     support_features=support, support_labels=(torch.arange(20) < 15).double(),
                     query_features=query, query_labels=torch.zeros(20, dtype=torch.double))


def smooth_backbone():
    # Positive hidden activations keep the finite-difference probe away from ReLU kinks.
    model = BaseDNN(vocab_size=64, embedding_dim=1).double()
    model.mlp = nn.Sequential(nn.Linear(8, 1, bias=False), nn.ReLU(),
                              nn.Linear(1, 1, bias=False)).double()
    with torch.no_grad():
        model.embedding.weight[:, 0] = torch.arange(64) / 100 + 0.2
        model.mlp[0].weight.fill_(1)
        model.mlp[2].weight.fill_(1)
    return model.requires_grad_(False).eval()


def test_query_labels_cannot_change_current_scores():
    base = smooth_backbone()
    gen = MetaGenerator(1, "attribute").double()
    task = example_task()
    with torch.no_grad():
        gen.calibration.weight.fill_(0.5)
    result = score_tasks(base, gen, [task], 0.1, training=True)
    flipped = replace(task, query_labels=1 - task.query_labels)
    flipped_scores = score_tasks(base, gen, [flipped], 0.1, training=True)
    for name in ("initial", "adapted", "calibrated", "history_means",
                 "support_logits", "query_logits"):
        assert torch.equal(getattr(result, name), getattr(flipped_scores, name))
    assert query_loss(result, [task]) != query_loss(flipped_scores, [flipped])
    mean = base.embedding(task.support_features[:15, USER_INDEX]).mean(0)
    torch.testing.assert_close(result.history_means[0], mean)
    torch.testing.assert_close(result.calibrated[0], result.adapted[0] + 0.5 * mean)

    negative = replace(task, support_labels=torch.zeros_like(task.support_labels))
    negative_scores = score_tasks(base, gen, [negative], 0.1, training=True)
    assert torch.equal(negative_scores.initial, result.initial)
    assert torch.equal(negative_scores.support_logits, result.support_logits)
    assert not torch.equal(negative_scores.adapted, result.adapted)
    assert torch.count_nonzero(negative_scores.history_means) == 0
    assert torch.equal(negative_scores.adapted, negative_scores.calibrated)


@pytest.mark.parametrize("method", ["attribute", "shared"])
def test_support_step_is_per_movie(method):
    base = smooth_backbone()
    gen = MetaGenerator(1, method).double()
    first = example_task()
    second = replace(example_task(51), support_labels=1 - first.support_labels)
    alone = score_tasks(base, gen, [first], 0.3, training=True)
    both = score_tasks(base, gen, [first, second], 0.3, training=True)
    torch.testing.assert_close(alone.adapted[0], both.adapted[0])
    torch.testing.assert_close(alone.calibrated[0], both.calibrated[0])
    torch.testing.assert_close(alone.query_logits[0], both.query_logits[0])
    assert not torch.equal(both.adapted[0], both.adapted[1])


def test_query_gradient_updates_generator_only_with_frozen_backbone():
    base = smooth_backbone()
    gen = MetaGenerator(1, "attribute").double()
    task = example_task()
    frozen = {name: p.detach().clone() for name, p in base.named_parameters()}
    optimizer = torch.optim.Adam(gen.parameters(), lr=0.001)
    result = score_tasks(base, gen, [task], 0.7, training=True)
    query_loss(result, [task]).backward()

    # The derivative of query loss w.r.t. the year coefficient includes the support Hessian.
    year_weight = gen.attribute.weight
    analytic = year_weight.grad[0, 0].item()
    delta = 1e-5

    def perturbed_loss(value):
        with torch.no_grad():
            year_weight[0, 0] = value
        scores = score_tasks(base, gen, [task], 0.7, training=False)
        return query_loss(scores, [task]).item()

    start = year_weight[0, 0].item()
    numeric = (perturbed_loss(start + delta) - perturbed_loss(start - delta)) / (2 * delta)
    with torch.no_grad():
        year_weight[0, 0] = start
    assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-9)
    for weight in (year_weight, gen.calibration.weight):
        assert weight.grad is not None and torch.isfinite(weight.grad).all()
        assert weight.grad.norm() > 0
    optimizer.step()
    assert torch.count_nonzero(year_weight) > 0
    assert torch.count_nonzero(gen.calibration.weight) > 0
    for name, param in base.named_parameters():
        assert param.grad is None and torch.equal(param, frozen[name])


@pytest.mark.parametrize("method", ["attribute", "shared"])
def test_meta_validation_is_read_only(method):
    base = smooth_backbone()
    training_task = example_task()
    validation_task = example_task(51)

    def run(val):
        gen = MetaGenerator(1, method).double()
        optimizer = torch.optim.Adam(gen.parameters())
        for _ in range(2):
            optimizer.zero_grad()
            result = score_tasks(base, gen, [training_task], 0.1, training=True)
            query_loss(result, [training_task]).backward()
            optimizer.step()
            before = [p.detach().clone() for p in gen.parameters()]
            with torch.no_grad():
                scores = score_tasks(base, gen, [val], 0.1, training=False)
                loss = query_loss(scores, [val])
            assert all(torch.equal(p, old) for p, old in zip(gen.parameters(), before))
            assert not any(getattr(scores, field).requires_grad for field in
                           ("initial", "adapted", "calibrated", "query_logits", "support_logits"))
        return gen.state_dict(), optimizer.state_dict()["state"], loss.item()

    weights, state, val_loss = run(validation_task)
    weights_flip, state_flip, val_loss_flip = run(
        replace(validation_task, query_labels=1 - validation_task.query_labels))
    assert val_loss != val_loss_flip
    for name, value in weights.items():
        assert torch.equal(value, weights_flip[name])
    for number, buffers in state.items():
        for name, value in buffers.items():
            assert torch.equal(value, state_flip[number][name])


def test_full_backbone_stays_frozen_during_meta_update(tmp_path):
    from test_meta_tasks import make_meta_source

    root, holdout, movies = make_meta_source(tmp_path / "data")
    tasks = load_meta_tasks(root, holdout)
    torch.manual_seed(123)
    path = tmp_path / "weights.pt"
    torch.save(BaseDNN().state_dict(), path)
    backbone = BaseDNN()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(checkpoint)
    backbone.eval().requires_grad_(False)
    gen = MetaGenerator(backbone.embedding.embedding_dim, "attribute")
    optimizer = torch.optim.Adam(gen.parameters(), lr=0.001)
    batch = [tasks[key] for key in movies]
    scores = score_tasks(backbone, gen, batch, 0.1, training=True)
    query_loss(scores, batch).backward()
    optimizer.step()
    assert gen.attribute.weight.detach().norm() > 0
    for key, parameter in backbone.named_parameters():
        assert parameter.grad is None and torch.equal(parameter, checkpoint[key])


def test_meta_diagnostic_budget_and_reload():
    from scripts.train_meta import movie_batches

    train = list(range(85))
    for size in (25, 50, 85):
        batches = list(movie_batches(train[:size], 123, 4, 85, 440))
        assert batches == list(movie_batches(train[:size], 123, 4, 85, 440))
        assert len(batches) == 440
        for start in range(0, 440, 22):
            epoch = batches[start:start + 22]
            assert [len(batch) for batch in epoch] == [4] * 21 + [1]
            flat = [movie for batch in epoch for movie in batch]
            assert len(flat) == 85 and set(flat) == set(train[:size])
            assert all(len(set(batch)) == len(batch) for batch in epoch)
            assert flat == (flat[:size] * ((85 + size - 1) // size))[:85]


def test_meta_training_uses_only_pseudo_feedback(tmp_path, monkeypatch):
    import hashlib
    import json
    from pathlib import Path

    import numpy as np
    import yaml
    from coldstart.models.base_dnn import BaseDNN
    from scripts.train_meta import run
    from test_meta_tasks import make_meta_source

    root, holdout_path, _ = make_meta_source(tmp_path / "data", old_count=1058)
    base = tmp_path / "base"
    base.mkdir()
    holdout = holdout_path.read_bytes()
    (base / "holdout.json").write_bytes(holdout)
    digest = hashlib.sha256(holdout).hexdigest()
    (base / "metrics.json").write_text(json.dumps({"data_split": "base_train", "holdout_sha256": digest}))
    torch.manual_seed(44)
    torch.save(BaseDNN().state_dict(), base / "weights.pt")
    config_path = tmp_path / "config.yaml"
    source_config = Path(__file__).resolve().parents[1] / "configs" / "meta_training.yaml"
    config = yaml.safe_load(source_config.read_text())
    assert config["checkpoints"] == [22, 110, 220, 440]
    assert config["seeds"] == [123, 456, 789]
    assert config["groups"] == {"attribute": [25, 50, 85], "shared": [85]}
    config.update(data_root=str(root), holdout_path=str(holdout_path), base_run=str(base))
    config.update(seeds=[123], groups={"attribute": [85]}, checkpoints=[1, 3])
    config_path.write_text(yaml.safe_dump(config))
    original_load = np.load

    def guard(path, *args, **kwargs):
        assert Path(path).name in ("meta_a.npz", "meta_b.npz", "items.npz")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guard)
    first = tmp_path / "runs" / "0.2-attribute-n85-s123"
    second = tmp_path / "repeat" / first.name
    for directory in (first, second):
        run(config_path, directory.parent)
    metrics = json.loads((first / "metrics.json").read_text())
    assert metrics == json.loads((second / "metrics.json").read_text())
    replay = tmp_path / "from_snapshot"
    run(first / "config.yaml", replay)
    assert metrics == json.loads((replay / first.name / "metrics.json").read_text())
    assert [entry["step"] for entry in metrics["checkpoints"]] == [1, 3]
    assert metrics["selected_step"] == min(
        metrics["checkpoints"], key=lambda item: item["validation_query_logloss"]
    )["step"]
    assert metrics["steps"] == 3 and metrics["movie_tasks"] == 12
    assert metrics["checkpoint_reload_verified"]
    assert metrics["holdout_sha256"] == digest
    assert metrics["backbone_sha256"] == hashlib.sha256((base / "weights.pt").read_bytes()).hexdigest()
    split = json.loads((first / "movies.json").read_text())
    assert len(split["train"]) == 85 and len(split["validation"]) == 21
    assert set(split["train"]).isdisjoint(split["validation"])
    assert all(set(step["movies"]) <= set(split["train"]) for step in metrics["updates"])
    for step in metrics["updates"]:
        assert step["initial_grad_norm"] > 0 and step["mean_adaptation_l2"] > 0

    # Verify the selected checkpoint against the validation loss recorded when it was selected.
    from coldstart.data.meta_tasks import load_meta_tasks
    from coldstart.models.generators.attribute import MetaGenerator
    from coldstart.train.meta import query_loss, score_tasks
    generator = MetaGenerator(10, "attribute")
    weights = torch.load(first / "weights.pt", map_location="cpu", weights_only=True)
    generator.load_state_dict(weights)
    backbone = BaseDNN().requires_grad_(False).eval()
    backbone.load_state_dict(torch.load(base / "weights.pt", map_location="cpu", weights_only=True))
    tasks = load_meta_tasks(root, holdout_path)
    validation = [tasks[movie] for movie in split["validation"]]
    result = score_tasks(backbone, generator, validation, config["inner_lr"], training=False)
    assert query_loss(result, validation).item() == pytest.approx(
        min(checkpoint["validation_query_logloss"] for checkpoint in metrics["checkpoints"]), abs=1e-6
    )

    prior = (first / "metrics.json").read_bytes()
    run(config_path, first.parent)
    assert (first / "metrics.json").read_bytes() == prior

    # Perturb only held-out B labels, then exercise the actual optimizer/selection loop.
    query_path = root / "meta_b.npz"
    with np.load(query_path) as archive:
        features, labels = archive["features"], archive["label"]
    held_out = np.isin(features[:, MOVIE_INDEX], split["validation"])
    labels[held_out] = 1 - labels[held_out]
    np.savez(query_path, features=features, label=labels)
    flipped_dir = tmp_path / "flipped"
    run(config_path, flipped_dir)
    flipped = json.loads((flipped_dir / first.name / "metrics.json").read_text())
    assert flipped["updates"] == metrics["updates"]
    for before, after in zip(metrics["checkpoints"], flipped["checkpoints"]):
        assert before["train_query_logloss"] == after["train_query_logloss"]
        assert before["validation_query_logloss"] != after["validation_query_logloss"]

    config["inner_lr"] *= 2
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(FileExistsError):
        run(config_path, first.parent)
    assert (first / "metrics.json").read_bytes() == prior
    (base / "holdout.json").write_bytes(holdout + b" ")
    with pytest.raises(ValueError, match="holdout"):
        run(config_path, tmp_path / "mismatch")
    assert not (tmp_path / "mismatch" / first.name).exists()
