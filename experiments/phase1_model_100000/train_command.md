# Phase 1: random initialization to model_100000

```bash
cd "$(git rev-parse --show-toplevel)"

python scripts/train.py \
  --task AMP-MJLAB-PM01-Phase1-v0 \
  --num_envs 4096 \
  --headless \
  --max_iterations 100001 \
  --run_name phase1_0_to_100000
```

This starts from random initialization. In this runner, 100001 learning iterations
produce the checkpoint numbered `model_100000.pt`.
