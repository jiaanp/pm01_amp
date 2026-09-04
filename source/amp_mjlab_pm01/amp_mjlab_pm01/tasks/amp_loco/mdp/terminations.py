from __future__ import annotations

import torch
from isaaclab.managers import TerminationManager


class DelayedTerminationManager(TerminationManager):
    """Exact AMP_mjlab delayed-reset state machine adapted to IsaacLab."""

    def __init__(self, base: TerminationManager, delay_env_mask: torch.Tensor, max_delay_steps: int):
        self.__dict__.update(base.__dict__)
        self._delay_env_mask = delay_env_mask
        self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self._max_delay_steps = max_delay_steps

    def compute(self) -> torch.Tensor:
        dones = super().compute()
        if self._max_delay_steps <= 0:
            return dones
        delay_and_done = self._delay_env_mask & dones
        self._delay_counters[delay_and_done] += 1
        not_ready = delay_and_done & (self._delay_counters < self._max_delay_steps)
        self._terminated_buf[not_ready] = False
        ready = delay_and_done & (self._delay_counters >= self._max_delay_steps)
        self._delay_counters[ready] = 0
        self._delay_counters[self._delay_env_mask & ~dones] = 0
        return self._truncated_buf | self._terminated_buf

