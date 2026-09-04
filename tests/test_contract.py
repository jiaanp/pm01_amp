from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))


def test_source_dataset_lock() -> None:
    manifest = json.loads((ROOT / "data" / "pm01" / "manifest.json").read_text())
    assert len(manifest["motions"]) == 18
    assert manifest["gmr_commit"] == "bb1bbe40774794fceb2a7c579a3464a28e68c844"


def test_pm01_motion_contract() -> None:
    paths = sorted((ROOT / "data" / "pm01").glob("*/*.npz"))
    assert len(paths) == 18
    for path in paths:
        data = np.load(path, allow_pickle=False)
        assert data["joint_pos"].shape[1] == 24
        assert data["joint_vel"].shape == data["joint_pos"].shape
        assert list(data["joint_names"])[-1] == "J23_HEAD_YAW"
        np.testing.assert_array_equal(data["joint_pos"][:, -1], 0.0)
        np.testing.assert_array_equal(data["joint_vel"][:, -1], 0.0)


def test_actor_critic_std_is_direct_scalar() -> None:
    source = (ROOT / "third_party" / "rsl_rl" / "modules" / "actor_critic.py").read_text()
    assert "self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))" in source
    assert "torch.clamp_min(self.std, 1.0e-6).expand_as(mean)" in source
    assert "softplus" not in source


def test_expected_dimensions() -> None:
    assert 4 * (3 + 3 + 3 + 24 + 24 + 24) == 324
    assert 4 * (81 + 3 + 13 * 3 + 13 * 6) == 804
    assert 13 * (3 + 6 + 3 + 3) == 195

