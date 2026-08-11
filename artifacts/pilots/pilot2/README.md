# Pilot 2: two independent policies

Configuration: 2,048 CUDA matches, rollout 64, repeat 4, 30 PPO chunks, learning
rate 1e-4 on an NVIDIA L4. Warm throughput was about 0.54M env-ticks/s.

Verdict: two independent policies removed shared-gradient cancellation and
produced increasing self-play engagement, including nine deaths in chunk 28.
It did not learn a generally competent boxer: role 0 remained at zero wins
against the fixed approach-and-charged-attack baseline and sometimes learned
non-engagement. Sparse symmetric self-play from random initialization is the
remaining recipe defect.

The next recipe bootstraps one learner against a stationary target, then a
fixed chaser/charged-punch baseline. Only after it learns engagement is the
policy cloned into two independently optimized adversaries.
