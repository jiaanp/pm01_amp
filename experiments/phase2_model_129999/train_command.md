# Phase 2: model_100000 to model_129999

```bash
cd "$(git rev-parse --show-toplevel)"

python scripts/train.py \
  --task AMP-MJLAB-PM01-Phase2-v0 \
  --num_envs 4096 \
  --headless \
  --max_iterations 30000 \
  --run_name phase2_hardfall_100000_to_129999 \
  --resume checkpoints/model_100000.pt
```

Do not pass `--finetune`: Phase 2 restores the policy, critic, AMP discriminator,
observation normalizers, optimizer, and iteration counter from `model_100000.pt`.
