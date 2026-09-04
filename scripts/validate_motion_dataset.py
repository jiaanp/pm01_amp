#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "pm01"
MODEL_PATH = ROOT / "third_party" / "gmr" / "assets" / "engineai_pm01" / "pm_v2.xml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    source_files = sorted((ROOT / "data" / "source_g1").glob("*/*.npz"))
    output_files = sorted(DATA_ROOT.glob("*/*.npz"))
    if [p.relative_to(ROOT / "data" / "source_g1") for p in source_files] != [
        p.relative_to(DATA_ROOT) for p in output_files
    ]:
        raise RuntimeError("PM01 output set does not exactly match the 18-file source split.")

    model = mj.MjModel.from_xml_path(str(MODEL_PATH))
    movable_joint_ids = [i for i in range(model.njnt) if model.jnt_type[i] != mj.mjtJoint.mjJNT_FREE]
    lower = model.jnt_range[movable_joint_ids, 0]
    upper = model.jnt_range[movable_joint_ids, 1]
    summaries = []
    for path in output_files:
        data = np.load(path, allow_pickle=False)
        frames = data["joint_pos"].shape[0]
        expected_shapes = {
            "joint_pos": (frames, 24),
            "joint_vel": (frames, 24),
            "body_pos_w": (frames, len(data["body_names"]), 3),
            "body_quat_w": (frames, len(data["body_names"]), 4),
            "body_lin_vel_w": (frames, len(data["body_names"]), 3),
            "body_ang_vel_w": (frames, len(data["body_names"]), 3),
        }
        for key, shape in expected_shapes.items():
            if data[key].shape != shape:
                raise RuntimeError(f"{path.name}: {key} {data[key].shape} != {shape}")
            if not np.isfinite(data[key]).all():
                raise RuntimeError(f"{path.name}: non-finite {key}")
        quat_error = float(np.max(np.abs(np.linalg.norm(data["body_quat_w"], axis=-1) - 1.0)))
        if quat_error > 2e-4:
            raise RuntimeError(f"{path.name}: quaternion norm error {quat_error}")
        if np.max(np.abs(data["joint_pos"][:, -1])) > 1e-7 or np.max(np.abs(data["joint_vel"][:, -1])) > 1e-7:
            raise RuntimeError(f"{path.name}: head reference is not neutral")
        violation = np.maximum(lower - data["joint_pos"], data["joint_pos"] - upper)
        max_violation = float(np.maximum(violation, 0.0).max())
        if max_violation > 2e-4:
            raise RuntimeError(f"{path.name}: joint limit violation {max_violation}")
        summaries.append({"file": str(path.relative_to(ROOT)), "frames": frames, "sha256": sha256(path)})

    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    manifest_hashes = {item["output"]: item["output_sha256"] for item in manifest["motions"]}
    for item in summaries:
        if manifest_hashes.get(item["file"]) != item["sha256"]:
            raise RuntimeError(f"Manifest hash mismatch: {item['file']}")
    total_frames = sum(item["frames"] for item in summaries)
    print(f"Motion dataset OK: {len(summaries)} clips, {total_frames} frames, 24 DoF, neutral head.")


if __name__ == "__main__":
    main()

