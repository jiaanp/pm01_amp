# PM01 AMP Recovery：Phase 1 到 Phase 2

本项目是基于 **IsaacLab 2.3.1 + RSL-RL** 的 PM01 AMP 运动控制项目，参考 AMP_mjlab 的统一策略思路：同一个 policy 学习 locomotion（走 / 跑）与 recovery（跌倒后起身）。

本仓库冻结了产生 `model_129999.pt` 的完整两阶段训练谱系：

- **Phase 1（0 → 100000）**：从随机初始化训练，得到 `model_100000.pt`。
- **Phase 2（100000 → 129999）**：完整恢复 Phase 1 checkpoint，在 hard-fall recovery curriculum 下继续训练，得到 `model_129999.pt`。

训练所需的 18 条 PM01 参考动作、PM01 资产、环境配置和两阶段审计快照已随仓库提供。

## 核心思路

传统方式常将走跑与起身分别训练并在运行时切换。本项目将两类能力纳入同一个 actor-critic 与同一个 AMP 判别器中：

- **动作数据分组**：`data/pm01/WalkandRun` 为走跑参考动作，`data/pm01/Recovery` 为恢复参考动作。
- **Phase 1**：使用原始的 motion reset 与 delayed termination，在统一 AMP 目标下建立 locomotion 与基础 recovery 能力。
- **Phase 2**：从 `model_100000.pt` 严格恢复训练状态，使用 stratified hard-fall curriculum 增强完全跌倒后的起身能力。
- **统一 AMP 训练**：单一 actor-critic + 单一 discriminator，在同一训练过程中联合优化速度跟踪、运动风格与恢复能力。

这样可以减少走跑 / 起身控制器切换造成的状态不连续。基线动作空间为 24 DoF；actor 观测 324 维、critic 观测 804 维、AMP 输入 195 维。

## 环境要求

已验证的环境组合：

- Linux x86_64（Ubuntu 24.04；Ubuntu 22.04 也可作为目标平台）
- NVIDIA GPU、可用驱动与 `nvidia-smi`
- Python 3.11
- Isaac Sim `5.1.0`
- IsaacLab `v2.3.1`
- PyTorch `2.7.0 + cu128` 与 torchvision `0.22.0`

不要混用 IsaacLab 其他版本、Isaac Sim 4.x 或 Python 3.10；它们可能造成 API 或二进制不兼容。

## 快速开始

### 0. 首次安装 Isaac 运行时（新机器只需一次）

以下步骤创建独立 Conda 环境、安装 Isaac Sim、固定版本 IsaacLab，以及本项目依赖。机器需预先安装兼容的 NVIDIA 驱动。

```bash
conda create -n pm01_amp python=3.11 -y
conda activate pm01_amp
python -m pip install --upgrade pip

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

cd <工作目录>
git clone --branch v2.3.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git IsaacLab_2.3.1
cd IsaacLab_2.3.1
./isaaclab.sh --install rsl_rl
cd ..
```

若 `isaaclab.sh --install` 提示缺少编译工具，执行：

```bash
sudo apt update
sudo apt install -y cmake build-essential git
```

### 1. 安装仓库

此步骤采用 AMP_mjlab 的仓库安装方式：在**同一个已激活的 Conda 环境**中，将项目和随仓库固定的 RSL-RL 都以 editable 模式安装。

```bash
conda activate pm01_amp

git clone <你的仓库地址> pm01_amp
cd pm01_amp
python -m pip install -e .

cd third_party/rsl_rl
python -m pip install -e .
cd ../..
```

`python -m pip install -e .` 安装项目声明的 NumPy、Gymnasium、tqdm、GitPython、TensorBoard、MuJoCo 等依赖，并注册 PM01 任务包。`third_party/rsl_rl` 是本项目固定版本的 AMP runner；安装它可以确保外部脚本也使用与基线一致的实现。

### 2. 验证环境和仓库

```bash
python -c "import isaacsim, isaaclab, isaaclab_rl, rsl_rl, torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
python -m pytest -q tests
python scripts/verify_provenance.py
python scripts/validate_motion_dataset.py
```

预期是输出 CUDA PyTorch 与 GPU 名称、测试全部通过、数据校验不报错。

### 3. 查看主要任务并做烟雾测试

主要任务：

- `AMP-MJLAB-PM01-Phase1-v0`：从零训练至 `model_100000.pt`
- `AMP-MJLAB-PM01-Phase2-v0`：从 `model_100000.pt` 续训至 `model_129999.pt`
- `AMP-MJLAB-PM01-Phase2-Play-v0`：Phase 2 播放环境

先运行一轮小规模训练，确认 GPU、场景、数据和 AMP 网络均能启动：

```bash
python scripts/train.py \
  --task AMP-MJLAB-PM01-Phase1-v0 \
  --num_envs 16 --headless --max_iterations 1 \
  --run_name smoke_test
```

出现 `[INFO] Training returned` 即说明训练链路可用。

## 目录说明

```text
pm01_amp/
├── assets/                    # PM01 URDF 与网格
├── checkpoints/               # model_100000.pt、model_129999.pt
├── data/
│   ├── source_g1/             # 锁定的源参考动作
│   └── pm01/                  # 训练直接读取的 18 条 PM01 动作
├── experiments/               # 原始 env.json / agent.json 与可复制命令
├── scripts/                   # train、play、重定向与数据验证工具
├── source/amp_mjlab_pm01/     # PM01 AMP 环境、MDP 与 PPO 配置
├── third_party/
│   ├── gmr/                   # 重定向使用的固定 GMR 实现
│   └── rsl_rl/                # 训练使用的固定 RSL-RL 实现
└── tests/                     # 数据与训练配置契约测试
```

## 训练前检查

首次部署时，先用少量环境运行一轮，确认 GPU、场景、数据和 AMP 网络都能启动：

```bash
python scripts/train.py \
  --task AMP-MJLAB-PM01-Phase1-v0 \
  --num_envs 16 --headless --max_iterations 1 \
  --run_name smoke_test
```

成功后会在 `logs/rsl_rl/pm01_amp_mjlab_locomotion/` 创建日志目录。`logs/` 已被 Git 忽略。

## 训练

### 1. Phase 1：从零至 model_100000

```bash
python scripts/train.py \
  --task AMP-MJLAB-PM01-Phase1-v0 \
  --num_envs 4096 --headless \
  --max_iterations 100001 \
  --run_name phase1_0_to_100000
```

训练循环从 0 计数，因此设为 `100001` 才会保存编号为 `model_100000.pt` 的 checkpoint。生成文件位于新日志目录中。

### 2. Phase 2：从 model_100000 至 model_129999

将下列 checkpoint 路径替换为上一步新生成的 `model_100000.pt`；也可使用仓库随附的 `checkpoints/model_100000.pt` 来验证 Phase 2。

```bash
python scripts/train.py \
  --task AMP-MJLAB-PM01-Phase2-v0 \
  --num_envs 4096 --headless \
  --max_iterations 30000 \
  --run_name phase2_hardfall_100000_to_129999 \
  --resume checkpoints/model_100000.pt
```

**不要在复现 Phase 2 时添加 `--finetune`。** 正常 `--resume` 会恢复策略、critic、AMP 判别器、经验归一化器、优化器状态和当前训练轮次；日志应显示从 `Learning iteration 100000/...` 开始。`--finetune` 只适合派生实验，会重置优化器和计数器，因此不能用于严格复现。

两阶段原始命令和审计快照分别见：

- `experiments/phase1_model_100000/`
- `experiments/phase2_model_129999/`

## 评估与可视化

在具有图形显示的机器上运行：

```bash
python scripts/play.py \
  --checkpoint checkpoints/model_129999.pt \
  --task AMP-MJLAB-PM01-Phase2-Play-v0 \
  --num_envs 16
```

播放需要桌面图形会话。若看到 `Failed to open display`，请在本地桌面、VNC / VirtualGL 会话中运行，或检查 `DISPLAY`；无头训练请始终添加 `--headless`。

## 运动数据准备

普通训练直接使用 `data/pm01/`，不需要 GMR。只有修改源动作、机器人运动学映射或希望重新生成数据时，才安装并使用 GMR：

```bash
pip install -e third_party/gmr
python scripts/retarget_g1_to_pm01.py
python scripts/validate_motion_dataset.py
```

GMR 的安装会额外安装 MuJoCo、SciPy、Mink、QP solver、OpenCV 等依赖。某些源动作转换还需要 SMPL / SMPL-X body model；这些模型通常受单独许可约束，使用者必须自行合法取得并按 GMR 文档配置。重定向链路依赖 `third_party/gmr/`、`assets/` 与 `data/source_g1/`。重新生成数据会改变训练分布；建议保留原始 `data/pm01/` 副本，并更新 `data/pm01/manifest.json` 与实验记录。

## 二次开发

### 保持基线可复现

- 不要直接修改 `AMP-MJLAB-PM01-Phase1-v0` 或 `AMP-MJLAB-PM01-Phase2-v0` 的基线配置。为新实验新增 task ID、环境配置与 `--run_name`。
- 每次实验记录使用的 checkpoint、`env.json`、`agent.json`、随机种子和数据 manifest。
- 先执行 16 环境 / 1 iteration 烟雾测试，再运行 4096 环境长训练。

### 主要修改入口

| 目标 | 修改位置 | 注意事项 |
| --- | --- | --- |
| Phase 1 reset / curriculum | `tasks/amp_loco/phase1_env_cfg.py` | 改动会使 0→100000 不再是原始基线 |
| Phase 2 hard-fall curriculum / reward | `tasks/amp_loco/env_cfg.py` | 改动会使 100000→129999 不再严格可复现 |
| AMP、观测、终止、奖励函数 | `tasks/amp_loco/mdp/` | 改观测维度不能直接正常恢复旧 checkpoint |
| PPO / AMP 超参数 | `tasks/amp_loco/agents/amp_ppo_cfg.py` | 与 checkpoint 对齐后再做派生实验 |
| PM01 资产 | `assets/` 与 `assets/amp_mjlab_pm01/assets/pm01.py` | 必须重新验证关节名、DoF 顺序与动力学 |

当前基线的 actor 观测为 324 维、critic 观测为 804 维、AMP 判别器输入为 195 维、动作空间为 24 DoF。若修改这些维度，不能使用普通 `--resume`；应从头训练，或仅在理解权重扩展逻辑后使用 `--finetune --expand_observation N`。

## 常见问题

### `python: command not found`

未激活正确 Conda 环境。执行 `conda activate pm01_amp` 后再运行项目命令。

### `ModuleNotFoundError: isaaclab`

当前 Python 没有安装 IsaacLab。确认 `python -c "import isaaclab"` 成功后再安装本项目。

### 显存不足或启动过慢

先用 `--num_envs 16` 验证；再逐步增大到显存允许的数量。正式复现使用的数量是 4096，减少环境数会改变采样吞吐和训练时间。

### Phase 2 没有从 100000 轮开始

确认使用了普通 `--resume`，没有使用 `--finetune`，并检查传入的是 Phase 1 生成的 `model_100000.pt`。

## 发布前的许可证与数据说明

PM01 URDF / 网格、GMR 代码和源参考动作可能分别受到不同许可证或分发条款约束。公开发布前，请确认 EngineAI 资产许可、GMR 许可、源动作数据许可，并在仓库中加入相应 LICENSE、NOTICE 与第三方归属文件。


## 致谢

- 感谢 AMP_mjlab 提供统一 AMP locomotion / recovery 的实现参考。
- 感谢 [IsaacLab](https://github.com/isaac-sim/IsaacLab)、[RSL-RL](https://github.com/leggedrobotics/rsl_rl) 与 [GMR](https://github.com/YanjieZe/GMR) 的开源工作。
