from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pm01_recovery_curriculum_config() -> None:
    source = (
        ROOT
        / "source/amp_mjlab_pm01/amp_mjlab_pm01/tasks/amp_loco/env_cfg.py"
    ).read_text()
    assert '"delay_reset_env_ratio": 0.7' in source
    assert '"recovery_hard_ratio": 0.6' in source
    assert '"recovery_semi_ratio": 0.2' in source
    assert '"recovery_other_ratio": 0.2' in source
    assert '"target_height": 0.76' in source
    assert "init_stratified_recovery_loader" in source
    assert "reset_from_stratified_motion_data" in source


def test_strict_hard_fall_definition_and_metrics() -> None:
    source = (
        ROOT
        / "source/amp_mjlab_pm01/amp_mjlab_pm01/tasks/amp_loco/mdp/recovery_curriculum.py"
    ).read_text()
    assert "frames.root_pos[:, 2] < 0.25" in source
    assert "math.cos(math.radians(75.0))" in source
    assert 'Recovery/strict_hard_getup_success' in source
    assert 'Recovery/strict_hard_torque_saturation_rate' in source
    assert 'Recovery/strict_hard_soft_limit_violation_rate' in source
    assert '(STATIC_RESET, "static")' in source
    assert 'extras[f"Recovery/{name}_hard_success"]' in source
    assert 'Recovery/strict_hard_first_target_jump_rad' in source
    assert 'Recovery/strict_hard_peak_joint_speed_rad_s' in source


