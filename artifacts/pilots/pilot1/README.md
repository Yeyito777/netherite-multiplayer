# Pilot 1: shared-policy self-play

Configuration: 1,024 CUDA matches, rollout 64, repeat 4, ten PPO chunks on an
NVIDIA L4. The core environment sustained about 0.53M env-ticks/s after warmup.

Verdict: useful smoke, not a successful training recipe. Accepted hits recovered
from 55 at chunk 2 to 329 at chunk 8 and five deaths occurred in chunk 9, proving
that matches can generate combat and terminal events. However, evaluation win
rate against the fixed approach-and-attack baseline remained zero.

The main recipe defect was conceptual: one shared policy was updated from both
sides of a symmetric zero-sum match. Opposing policy gradients can cancel. The
trainer was changed after this receipt to use two independent adversarial
policies. The obsolete shared-policy checkpoint is intentionally not retained;
`metrics.jsonl` and `metrics.png` are the evidence.
