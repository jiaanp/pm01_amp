#!/usr/bin/env python3
"""Visualize a retargeted PM01 reference motion in IsaacLab 2.3.1."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "amp_mjlab_pm01"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a retargeted PM01 NPZ reference motion")
parser.add_argument(
    "--motion",
    type=Path,
    default=ROOT / "data" / "pm01" / "Recovery" / "fallAndGetUp1_subject1.npz",
)
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
parser.add_argument("--once", action="store_true", help="Exit after one pass instead of looping.")
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=0, help="Testing limit; zero means unlimited.")
parser.add_argument("--real_time", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.speed <= 0.0:
    parser.error("--speed must be greater than zero")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from amp_mjlab_pm01.assets.pm01 import PM01_24DOF_CFG


@configclass
class MotionSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85)),
    )
    robot: ArticulationCfg = PM01_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main() -> None:
    motion_path = args_cli.motion.expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file does not exist: {motion_path}")

    data = np.load(motion_path, allow_pickle=False)
    required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Motion is missing fields: {sorted(missing)}")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    frame_count = int(data["joint_pos"].shape[0])
    frame = args_cli.start_frame % frame_count

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / fps, device=args_cli.device))
    sim.set_camera_view(eye=[3.0, 3.0, 1.8], target=[0.0, 0.0, 0.75])
    scene = InteractiveScene(MotionSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]

    source_joint_names = [str(name) for name in data["joint_names"]]
    source_body_names = [str(name) for name in data["body_names"]]
    if set(source_joint_names) != set(robot.joint_names):
        raise ValueError(f"Joint-name mismatch: NPZ={source_joint_names}, Isaac={robot.joint_names}")
    joint_order = [source_joint_names.index(name) for name in robot.joint_names]
    root_index = source_body_names.index("LINK_BASE")

    joint_pos = torch.as_tensor(data["joint_pos"][:, joint_order], device=sim.device)
    joint_vel = torch.as_tensor(data["joint_vel"][:, joint_order], device=sim.device)
    root_pos = torch.as_tensor(data["body_pos_w"][:, root_index], device=sim.device)
    root_quat = torch.as_tensor(data["body_quat_w"][:, root_index], device=sim.device)
    root_lin_vel = torch.as_tensor(data["body_lin_vel_w"][:, root_index], device=sim.device)
    root_ang_vel = torch.as_tensor(data["body_ang_vel_w"][:, root_index], device=sim.device)

    print(f"[INFO] Playing {motion_path.name}: {frame_count} frames at {fps:g} Hz, {frame_count / fps:.2f} s")
    print("[INFO] Press Ctrl+C or close the Isaac window to stop.")
    step = 0
    while simulation_app.is_running() and (args_cli.max_steps <= 0 or step < args_cli.max_steps):
        started = time.time()
        robot.write_root_pose_to_sim(torch.cat((root_pos[frame], root_quat[frame])).unsqueeze(0))
        robot.write_root_velocity_to_sim(torch.cat((root_lin_vel[frame], root_ang_vel[frame])).unsqueeze(0))
        robot.write_joint_state_to_sim(joint_pos[frame].unsqueeze(0), joint_vel[frame].unsqueeze(0))
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        step += 1
        frame += 1
        if frame == frame_count:
            if args_cli.once:
                break
            frame = 0
        if args_cli.real_time:
            remaining = 1.0 / (fps * args_cli.speed) - (time.time() - started)
            if remaining > 0.0:
                time.sleep(remaining)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
