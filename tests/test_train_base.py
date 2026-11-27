import json
import subprocess
from pathlib import Path

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
