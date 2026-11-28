import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts import train_base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "base_smoke.yaml"


@pytest.mark.parametrize("split", ["base_train", "base_candidate", "meta_a", "meta_b", "warm_a", "warm_b", "final_eval"])
def test_train_rejects_ineligible_splits(tmp_path, split):
    config = yaml.safe_load(SMOKE_CONFIG.read_text())
    config["data"] = {"split": split, "root": str(tmp_path / "missing-data")}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match=split):
        train_base.train(config_path, run_dir)
    assert not run_dir.exists()


def test_external_smoke_data_is_rejected(tmp_path):
    config = yaml.safe_load(SMOKE_CONFIG.read_text())
    config["data"]["root"] = str(tmp_path / "external")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="root"):
        train_base.train(config_path, tmp_path / "run")


def test_seeded_smoke_and_checkpoint_reload(tmp_path, monkeypatch):
    # Observe the inputs actually seen by the model and its weights before the first step.
    original_model = train_base.BaseDNN
    trained = []
    initial_states = []
    batches = []

    def observe_model():
        model = original_model()
        trained.append(model)
        initial_states.append({name: weight.clone() for name, weight in model.state_dict().items()})
        trace = []
        batches.append(trace)

        def capture_batch(module, inputs):
            if module.training:
                trace.append(inputs[0].clone())

        model.register_forward_pre_hook(capture_batch)
        return model

    monkeypatch.setattr(train_base, "BaseDNN", observe_model)
    config = yaml.safe_load(SMOKE_CONFIG.read_text())
    run_dirs = [tmp_path / "first", tmp_path / "second"]
    for run_dir in run_dirs:
        train_base.train(SMOKE_CONFIG, run_dir)
        assert (run_dir / "config.yaml").read_bytes() == SMOKE_CONFIG.read_bytes()

    first, second = [json.loads((path / "metrics.json").read_text()) for path in run_dirs]
    assert first == second
    assert first["checkpoint_reload_verified"] is True
    assert first["seed"] == config["seed"]
    assert first["steps"] == len(first["loss"]) == config["steps"]
    assert first["git_commit"] == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    assert torch.isfinite(torch.tensor(first["loss"])).all()
    training_traces = [trace for trace in batches if trace]
    assert len(training_traces) == 2
    assert all(len(trace) == config["steps"] for trace in training_traces)
    assert all(torch.equal(left, right) for left, right in zip(*training_traces))

    states = [torch.load(path / "weights.pt", map_location="cpu", weights_only=True) for path in run_dirs]
    assert any(not torch.equal(value, initial_states[0][name]) for name, value in states[0].items())
    assert all(torch.equal(value, states[1][name]) for name, value in states[0].items())
    restored = original_model()
    restored.load_state_dict(states[0])
    with torch.no_grad():
        features = batches[0][0]
        assert torch.equal(restored.eval()(features), trained[0].eval()(features))


def test_train_rejects_mixed_steps_and_epochs(tmp_path):
    config = yaml.safe_load(SMOKE_CONFIG.read_text())
    config["epochs"] = 1
    config_path = tmp_path / "ambiguous.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="epochs"):
        train_base.train(config_path, run_dir)
    assert not run_dir.exists()


def test_real_base_epoch_and_checkpoint_reload(tmp_path, monkeypatch):
    from coldstart.data.schema import MOVIE_INDEX
    from scripts.prepare_holdout import prepare_holdout
    from test_item_holdout import make_small_dataset

    root = tmp_path / "data"
    make_small_dataset(root)
    output = tmp_path / "holdout"
    prepare_holdout(root, output, 42)
    holdout_path = output / "holdout.json"
    holdout_bytes = holdout_path.read_bytes()
    base_ids = json.loads(holdout_bytes)["base_movies"]
    with np.load(root / "base_candidate.npz") as archive:
        features = archive["features"]
        expected = features[np.isin(features[:, MOVIE_INDEX], base_ids)]

    config = yaml.safe_load((PROJECT_ROOT / "configs" / "base_holdout.yaml").read_text())
    assert config["data"]["split"] == "base_train"
    assert (config["seed"], config["learning_rate"], config["batch_size"], config["epochs"]) == (
        123, 0.001, 128, 1,
    )
    config["data"].update(root=str(root), holdout_path=str(holdout_path))
    config["batch_size"] = 8
    config_path = tmp_path / "real.yaml"
    config_path.write_text(yaml.safe_dump(config))

    original_model = train_base.BaseDNN
    batches = []
    initial_states = []
    models = []

    class ObservedModel(original_model):
        def __init__(self):
            super().__init__()
            models.append(self)
            initial_states.append({name: value.clone() for name, value in self.state_dict().items()})

        def forward(self, features):
            if self.training:
                batches.append(features.clone())
            return super().forward(features)

    monkeypatch.setattr(train_base, "BaseDNN", ObservedModel)
    run_dir = tmp_path / "run"
    train_base.train(config_path, run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert [len(batch) for batch in batches] == [8, 8, 2]
    assert Counter(map(tuple, torch.cat(batches).tolist())) == Counter(map(tuple, expected.tolist()))
    assert metrics["steps"] == len(batches)
    assert metrics["epochs"] == 1
    assert metrics["rows"] == metrics["samples_seen"] == len(expected)
    assert np.isfinite(metrics["loss"]).all()
    assert metrics["train_bce"] == pytest.approx(np.average(metrics["loss"], weights=[8, 8, 2]))
    assert (run_dir / "config.yaml").read_bytes() == config_path.read_bytes()
    assert (run_dir / "holdout.json").read_bytes() == holdout_bytes
    assert metrics["holdout_sha256"] == hashlib.sha256(holdout_bytes).hexdigest()
    assert metrics["checkpoint_reload_verified"] is True
    weights = torch.load(run_dir / "weights.pt", map_location="cpu", weights_only=True)
    assert any(not torch.equal(value, initial_states[0][name]) for name, value in weights.items())
    restored = original_model()
    restored.load_state_dict(weights)
    with torch.no_grad():
        assert torch.equal(restored.eval()(batches[0]), models[0].eval()(batches[0]))
