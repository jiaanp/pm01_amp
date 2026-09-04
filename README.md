# PM01 AMP Recovery: Phase 1 to Phase 2

[中文文档](README_zh.md)

This project is a PM01 AMP motion-control project built on **IsaacLab 2.3.1 + RSL-RL**. Following the unified-policy design of AMP_mjlab, one policy learns locomotion (walking / running) and recovery from falls.

This repository freezes the complete two-stage training lineage that produced `model_129999.pt`:

- **Phase 1 (0 to 100000):** train from random initialization and produce `model_100000.pt`.
- **Phase 2 (100000 to 129999):** fully resume the Phase 1 checkpoint and train with the hard-fall recovery curriculum to produce `model_129999.pt`.

The repository includes the 18 PM01 reference motions, PM01 robot assets, environment configurations, and configuration snapshots for both phases.

## Core Idea

Instead of training separate locomotion and get-up controllers and switching between them at runtime, this project places both capabilities in one actor-critic policy and one AMP discriminator.

- **Motion groups:** `data/pm01/WalkandRun` contains locomotion references; `data/pm01/Recovery` contains recovery references.
- **Phase 1:** uses the original motion reset and delayed termination to establish locomotion and basic recovery under a unified AMP objective.
- **Phase 2:** strictly resumes `model_100000.pt` and uses a stratified hard-fall curriculum to improve recovery from complete falls.
- **Unified AMP training:** one actor-critic and one discriminator jointly optimize velocity tracking, motion style, and recovery.

This reduces state discontinuities that commonly arise when locomotion and get-up controllers are switched. The baseline has a 24 DoF action space, 324 actor observations, 804 critic observations, and a 195-dimensional AMP input.

## Environment Requirements

The following configuration has been validated:

- Ubuntu 24.04 LTS
- NVIDIA GeForce RTX 5090 with 32 GB VRAM
- NVIDIA Driver `580.173.02`; `nvidia-smi` reports CUDA `13.0`
- Python 3.11
- Isaac Sim `5.1.0`
- IsaacLab `v2.3.1`
- PyTorch `2.7.0 + cu128` and torchvision `0.22.0`

The desktop display normally uses about 0.2 GB VRAM. Before training, ensure no other large CUDA workload is running. Do not mix other IsaacLab versions, Isaac Sim 4.x, or Python 3.10 with this baseline because API and binary incompatibilities may result.

## Quick Start

### 0. Install the Isaac Runtime (once per machine)

The following commands create an isolated Conda environment, install Isaac Sim, install the fixed IsaacLab release, and install the RSL-RL integration. A compatible NVIDIA driver and a working `nvidia-smi` are prerequisites.

```bash
conda create -n pm01_amp python=3.11 -y
conda activate pm01_amp
python -m pip install --upgrade pip

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

cd <workspace>
git clone --branch v2.3.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git IsaacLab_2.3.1
cd IsaacLab_2.3.1
./isaaclab.sh --install rsl_rl
cd ..
```

If `isaaclab.sh --install` reports missing build tools:

```bash
sudo apt update
sudo apt install -y cmake build-essential git
```

### 1. Install This Repository

Following the AMP_mjlab installation pattern, install both this project and its pinned RSL-RL copy in the **same activated Conda environment**.

```bash
conda activate pm01_amp

git clone <your-repository-url> pm01_amp
cd pm01_amp
python -m pip install -e .

cd third_party/rsl_rl
python -m pip install -e .
cd ../..
```

`python -m pip install -e .` installs the declared project dependencies: NumPy, Gymnasium, tqdm, GitPython, TensorBoard, MuJoCo, and the PM01 task package. `third_party/rsl_rl` is the pinned AMP runner used by the baseline.

### 2. Verify the Environment and Repository

```bash
python -c "import isaacsim, isaaclab, isaaclab_rl, rsl_rl, torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
python -m pytest -q tests
python scripts/verify_provenance.py
python scripts/validate_motion_dataset.py
```

The command should print a CUDA PyTorch build and an NVIDIA GPU name. Tests and both data checks should finish without errors.

### 3. Available Tasks and Smoke Test

The main tasks are:

- `AMP-MJLAB-PM01-Phase1-v0`: train from scratch to `model_100000.pt`
- `AMP-MJLAB-PM01-Phase2-v0`: resume from `model_100000.pt` to `model_129999.pt`
- `AMP-MJLAB-PM01-Phase2-Play-v0`: playback environment for Phase 2

Run a one-iteration smoke test before any long training job:

```bash
python scripts/train.py \
  --task AMP-MJLAB-PM01-Phase1-v0 \
  --num_envs 16 --headless --max_iterations 1 \
  --run_name smoke_test
```

The training pipeline is ready when it finishes with `[INFO] Training returned`.

## Directory Layout

```text
pm01_amp/
├── assets/                    # PM01 URDF and meshes
├── checkpoints/               # model_100000.pt and model_129999.pt
├── data/
│   ├── source_g1/             # locked source reference motions
│   └── pm01/                  # 18 PM01 motions consumed by training
├── experiments/               # original env.json, agent.json, and commands
├── scripts/                   # training, playback, retargeting, validation
├── source/amp_mjlab_pm01/     # PM01 AMP environment, MDP, and PPO config
├── third_party/
│   ├── gmr/                   # pinned GMR implementation for retargeting
│   └── rsl_rl/                # pinned RSL-RL implementation
└── tests/                     # data and configuration contract tests
```

## Training

### 1. Phase 1: Train from Scratch to model_100000

```bash
python scripts/train.py \
  --task AMP-MJLAB-PM01-Phase1-v0 \
  --num_envs 4096 --headless \
  --max_iterations 100001 \
  --run_name phase1_0_to_100000
```

The runner counts from zero, so `100001` learning iterations are required to produce `model_100000.pt`. Logs are written under `logs/rsl_rl/pm01_amp_mjlab_locomotion/`.

### 2. Phase 2: Resume model_100000 to model_129999

Replace the checkpoint path with the `model_100000.pt` from your Phase 1 run. The included `checkpoints/model_100000.pt` can also be used to validate the Phase 2 workflow.

```bash
python scripts/train.py \
  --task AMP-MJLAB-PM01-Phase2-v0 \
  --num_envs 4096 --headless \
  --max_iterations 30000 \
  --run_name phase2_hardfall_100000_to_129999 \
  --resume checkpoints/model_100000.pt
```

**Do not add `--finetune` when reproducing Phase 2.** A normal `--resume` restores the actor, critic, AMP discriminator, empirical normalizers, optimizer, and training iteration. The log should start from `Learning iteration 100000/...`. `--finetune` resets the optimizer and iteration counter and is only for derived experiments.

The original commands and configuration snapshots are stored in:

- `experiments/phase1_model_100000/`
- `experiments/phase2_model_129999/`

## Evaluation and Visualization

Run the following command from a graphical desktop session:

```bash
python scripts/play.py \
  --checkpoint checkpoints/model_129999.pt \
  --task AMP-MJLAB-PM01-Phase2-Play-v0 \
  --num_envs 16
```

Playback requires a graphical display. If `Failed to open display` appears, use a local desktop, VNC / VirtualGL session, or configure `DISPLAY`. Use `--headless` for training on a server without a display.

## Motion Data Preparation

Normal training consumes `data/pm01/` directly and does not require GMR. Install and use GMR only when modifying source motions, the kinematic mapping, or regenerating PM01 reference motions:

```bash
pip install -e third_party/gmr
python scripts/retarget_g1_to_pm01.py
python scripts/validate_motion_dataset.py
```

GMR installs additional dependencies including MuJoCo, SciPy, Mink, QP solvers, and OpenCV. Some source-motion conversions also require SMPL / SMPL-X body models. These are normally subject to separate license terms and must be obtained and configured by the user. Regenerating motion data changes the training distribution; retain a copy of the baseline `data/pm01/` and update `data/pm01/manifest.json` and experiment records.

## Further Development

### Preserve Baseline Reproducibility

- Do not directly modify the baseline `AMP-MJLAB-PM01-Phase1-v0` or `AMP-MJLAB-PM01-Phase2-v0` task. Add a new task ID, environment configuration, and `--run_name` for each derived experiment.
- Record the checkpoint, `env.json`, `agent.json`, random seed, and data manifest for every run.
- Run the 16-environment, one-iteration smoke test before a 4096-environment long run.

### Main Extension Points

| Goal | Location | Compatibility note |
| --- | --- | --- |
| Phase 1 reset / curriculum | `tasks/amp_loco/phase1_env_cfg.py` | Changes no longer reproduce the original 0 to 100000 baseline |
| Phase 2 hard-fall curriculum / reward | `tasks/amp_loco/env_cfg.py` | Changes no longer strictly reproduce 100000 to 129999 |
| AMP, observations, termination, rewards | `tasks/amp_loco/mdp/` | A changed observation dimension cannot normally resume an old checkpoint |
| PPO / AMP hyperparameters | `tasks/amp_loco/agents/amp_ppo_cfg.py` | Start derived experiments only after checkpoint compatibility is understood |
| PM01 assets | `assets/` and `assets/amp_mjlab_pm01/assets/pm01.py` | Revalidate joint names, DoF order, and dynamics |

If actor, critic, or AMP dimensions change, do not use a normal `--resume`. Train from scratch, or use `--finetune --expand_observation N` only after understanding the weight-expansion implementation.

## Troubleshooting

### `python: command not found`

Activate the Conda environment first:

```bash
conda activate pm01_amp
```

### `ModuleNotFoundError: isaaclab`

IsaacLab is not installed in the active Python environment. Complete the Isaac Sim and IsaacLab setup in Quick Start step 0, then retry.

### Out of memory or slow startup

First verify with `--num_envs 16`, then increase the count gradually. The baseline uses 4096 environments. A smaller count changes sample throughput and total training time.

### Phase 2 does not start at iteration 100000

Use a normal `--resume`, do not pass `--finetune`, and confirm that the checkpoint is the Phase 1 `model_100000.pt`.

## License and Data Notice Before Publication

The PM01 URDF / meshes, GMR code, and source reference motions may each have separate licenses or redistribution terms. Before making a public release, verify the applicable EngineAI asset license, GMR license, and source-motion license, then add the required LICENSE, NOTICE, and third-party attribution files.

## Acknowledgements

- [AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab) provided the implementation reference for unified AMP locomotion and recovery.
- Thanks to [IsaacLab](https://github.com/isaac-sim/IsaacLab), [RSL-RL](https://github.com/leggedrobotics/rsl_rl), and [GMR](https://github.com/YanjieZe/GMR) for their open-source work.
