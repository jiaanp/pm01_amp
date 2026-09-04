#!/usr/bin/env python3
"""Retarget the exact AMP_mjlab G1 NPZ set to PM01 through a clean GMR snapshot.

The bridge analytically inverts GMR's bvh_lafan1_to_g1 scale/orientation
mapping to recover equivalent LAFAN targets, then applies
bvh_lafan1_to_pm01. No previous PM01 motion is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
GMR_ROOT = ROOT / "third_party" / "gmr"
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting  # noqa: E402


GMR_COMMIT = "bb1bbe40774794fceb2a7c579a3464a28e68c844"
G1_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(name: str) -> dict:
    path = GMR_ROOT / "general_motion_retargeting" / "ik_configs" / name
    return json.loads(path.read_text())


def _reverse_map(table: dict) -> dict[str, tuple[str, list]]:
    return {entry[0]: (robot_body, entry) for robot_body, entry in table.items()}


def _wxyz_rotation(quat: np.ndarray) -> Rotation:
    return Rotation.from_quat(quat, scalar_first=True)


def _recover_lafan_targets(frame_pos: np.ndarray, frame_quat: np.ndarray, g1_cfg: dict) -> dict:
    body_index = {name: i for i, name in enumerate(G1_BODY_NAMES)}
    by_human = _reverse_map(g1_cfg["ik_match_table1"])
    root_name = g1_cfg["human_root_name"]
    root_robot, root_entry = by_human[root_name]
    root_idx = body_index[root_robot]
    root_link_q = _wxyz_rotation(frame_quat[root_idx])
    root_offset = Rotation.from_quat(root_entry[4], scalar_first=True)
    human_root_q = root_link_q * root_offset.inv()
    scaled_root_pos = frame_pos[root_idx] - root_link_q.apply(np.asarray(root_entry[3]))
    human_root_pos = scaled_root_pos / float(g1_cfg["human_scale_table"][root_name])

    targets: dict[str, list[np.ndarray]] = {}
    for human_name in g1_cfg["human_scale_table"]:
        robot_body, entry = by_human[human_name]
        idx = body_index[robot_body]
        link_q = _wxyz_rotation(frame_quat[idx])
        offset_q = Rotation.from_quat(entry[4], scalar_first=True)
        human_q = link_q * offset_q.inv()
        scaled_pos = frame_pos[idx] - link_q.apply(np.asarray(entry[3]))
        if human_name == root_name:
            human_pos = human_root_pos
        else:
            scale = float(g1_cfg["human_scale_table"][human_name])
            human_pos = human_root_pos + (scaled_pos - scaled_root_pos) / scale
        targets[human_name] = [human_pos, human_q.as_quat(scalar_first=True)]
    return targets


def _angular_velocity_w(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    frames, bodies, _ = quat_wxyz.shape
    result = np.zeros((frames, bodies, 3), dtype=np.float64)
    rotations = Rotation.from_quat(quat_wxyz.reshape(-1, 4), scalar_first=True)
    rotations = rotations.as_matrix().reshape(frames, bodies, 3, 3)

    def delta(a: np.ndarray, b: np.ndarray, denom: float) -> np.ndarray:
        rel = np.einsum("...ij,...kj->...ik", b, a)
        leading_shape = rel.shape[:-2]
        rotvec = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec()
        return rotvec.reshape(*leading_shape, 3) / denom

    if frames == 1:
        return result.astype(np.float32)
    result[0] = delta(rotations[0], rotations[1], dt)
    result[-1] = delta(rotations[-2], rotations[-1], dt)
    if frames > 2:
        result[1:-1] = delta(rotations[:-2], rotations[2:], 2.0 * dt)
    return result.astype(np.float32)


def _model_metadata(model: mj.MjModel) -> tuple[list[str], list[str], list[int]]:
    body_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    joint_names: list[str] = []
    qpos_addresses: list[int] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_FREE:
            continue
        joint_names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id))
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
    return body_names, joint_names, qpos_addresses


def retarget_file(source: Path, destination: Path, force: bool) -> dict:
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} exists; pass --force to replace generated data.")
    source_data = np.load(source, allow_pickle=False)
    required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
    if set(source_data.files) != required:
        raise ValueError(f"{source}: unexpected keys {source_data.files}")
    if source_data["body_pos_w"].shape[1] != len(G1_BODY_NAMES):
        raise ValueError(f"{source}: expected {len(G1_BODY_NAMES)} G1 bodies")

    g1_cfg = _load_config("bvh_lafan1_to_g1.json")
    retargeter = GeneralMotionRetargeting(
        src_human="bvh_lafan1",
        tgt_robot="engineai_pm01",
        solver="daqp",
        damping=5e-1,
        verbose=False,
        use_velocity_limit=False,
    )
    qpos_frames = []
    for pos, quat in zip(source_data["body_pos_w"], source_data["body_quat_w"]):
        target = _recover_lafan_targets(pos, quat, g1_cfg)
        qpos = retargeter.retarget(target)
        qpos[-1] = 0.0
        qpos_frames.append(qpos)
    qpos = np.asarray(qpos_frames, dtype=np.float64)

    model = retargeter.model
    body_names, joint_names, qpos_addresses = _model_metadata(model)
    if len(joint_names) != 24 or joint_names[-1] != "J23_HEAD_YAW":
        raise RuntimeError(f"Expected PM01 24-DoF model, got {joint_names}")

    body_pos = np.empty((len(qpos), len(body_names), 3), dtype=np.float32)
    body_quat = np.empty((len(qpos), len(body_names), 4), dtype=np.float32)
    data = mj.MjData(model)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mj.mj_forward(model, data)
        body_pos[frame] = data.xpos[1:]
        body_quat[frame] = data.xquat[1:]

    fps = float(np.asarray(source_data["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    joint_pos = qpos[:, qpos_addresses].astype(np.float32)
    joint_pos[:, -1] = 0.0
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    joint_vel[:, -1] = 0.0
    body_lin_vel = np.gradient(body_pos, dt, axis=0).astype(np.float32)
    body_ang_vel = _angular_velocity_w(body_quat, dt)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        fps=np.asarray([fps], dtype=np.float64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(body_names),
        source_relpath=np.asarray(str(source.relative_to(ROOT))),
        source_sha256=np.asarray(_sha256(source)),
        retarget_method=np.asarray("inverse_gmr_bvh_lafan1_to_g1__gmr_bvh_lafan1_to_pm01"),
        gmr_commit=np.asarray(GMR_COMMIT),
    )
    return {
        "source": str(source.relative_to(ROOT)),
        "output": str(destination.relative_to(ROOT)),
        "frames": int(len(qpos)),
        "fps": fps,
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sources = sorted((ROOT / "data" / "source_g1").glob("*/*.npz"))
    if args.limit is not None:
        sources = sources[: args.limit]
    records = []
    for index, source in enumerate(sources, 1):
        destination = ROOT / "data" / "pm01" / source.parent.name / source.name
        print(f"[{index}/{len(sources)}] {source.name}")
        records.append(retarget_file(source, destination, args.force))

    manifest_path = ROOT / "data" / "pm01" / "manifest.json"
    existing = []
    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text()).get("motions", [])
    by_output = {item["output"]: item for item in existing + records}
    manifest = {
        "schema_version": 1,
        "gmr_commit": GMR_COMMIT,
        "source_dataset": "AMP_mjlab@6c7a2947fccc973e4af8e6d90e550400f1b6fcfc",
        "motions": [by_output[key] for key in sorted(by_output)],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path} ({len(records)} motions processed).")


if __name__ == "__main__":
    main()

