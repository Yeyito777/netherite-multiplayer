# Pilot 3: staged engagement curriculum

Configuration: 2,048 CUDA matches, rollout 64, repeat 4, 80 PPO chunks,
learning rate 1e-4, entropy coefficient 0.005, 15 stationary-target chunks,
20 scripted-boxer chunks, then 45 independent self-play chunks on an NVIDIA
L4.

Warm simulator/trainer throughput was about 0.58–0.63 million environment ticks
per second. The run produced 68,753 accepted hits and 383 deaths. In the
adversarial phase alone it produced 54,716 accepted hits and 299 deaths, a clear
improvement in self-play engagement over pilots 1 and 2.

Verdict: **not an accepted boxing policy**. The final greedy policy scored no
kills in 128 held-out episodes against stationary, random, or scripted-boxer
opponents. Its stochastic form generated some combat but only one win and four
losses against the scripted boxer in a separate 128-episode diagnostic. The
curriculum raised interaction frequency without teaching a stable approach
policy; in particular, the stronger boxer phase can reward evasive behavior.

The next recipe adds potential-based distance-progress shaping only during the
bootstrap phases and records separate greedy stationary, greedy boxer, and
sampled boxer evaluations. The failed checkpoint is intentionally not retained;
`metrics.jsonl` and `metrics.png` are the auditable training receipt.
