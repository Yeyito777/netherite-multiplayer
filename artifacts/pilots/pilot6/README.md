# Pilot 6: class-balanced behavior cloning

Configuration: corrected `movement_v2` observations, 128 teacher steps, ten
class-balanced behavior-cloning epochs, and 60 adversarial PPO chunks over 2,048
parallel L4 matches. The PPO learning rate was `3e-5` and entropy coefficient was
`0.001`.

Class balancing restored attack behavior: the run produced 56,221 hits and 158
kills, and the final stationary evaluation recorded 1,767 punches. It still did
not produce a robust policy: the final greedy policy won only 2/128 stationary
episodes and lost 32/128 to the scripted chaser (the remaining episodes drew).

The optimization diagnostics identify the next root cause. Approximate PPO KL was
`0.069` in chunk 0 and remained near `0.03` at the end, with about 20% of samples
clipped. Those updates are much larger than a normal PPO trust region and can
erase the behavioral-cloning initialization before the curriculum stabilizes.

Verdict: failed deployment candidate; only metrics are retained. The next recipe
preserves and evaluates the pre-PPO behavioral-cloning checkpoint, reports
per-action-head BC accuracy, and stops PPO epochs when KL exceeds `0.01`.
