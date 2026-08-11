# Pilot 4: behavior-cloned warm start plus adversarial PPO

Configuration: 2,048 CUDA matches, rollout 64, repeat 4, 64 scripted teacher
collection steps, four behavior-cloning epochs, then 60 chunks of two-policy
adversarial PPO. Learning rate was 5e-5 and entropy coefficient 0.002 on an
NVIDIA L4.

The teacher warm start reached 92.8% aggregate categorical action accuracy. The
subsequent adversarial phase executed 15.73 million agent decisions at a median
0.553 million simulated environment ticks per second. It produced 43,121
accepted hits and 145 decisive kills. Both independently optimized roles killed
the other during unscripted self-play.

Held-out evaluation shows a real learned transition rather than loss alone: the
greedy role-0 policy went from 0/128 wins against a stationary target through
chunk 10 to 128/128 at chunk 15 and retained 128/128 through chunk 59. In the
larger final 512-seed receipt it again won 512/512 stationary matches. Against a
randomly moving policy it dealt 130.8 damage versus 65.3 while taking no losses,
but matches timed out before either side died. It remains substantially weaker
than the deterministic approach-and-charged-punch teacher: greedy role 0 lost
512/512, while sampled role 0 won 10, lost 60, and drew 442.

Verdict: **accepted as the first simulation-side MVP checkpoint**, not as a
Minecraft-deployable or strong boxer. It demonstrates stable target engagement,
unscripted adversarial hits, and deaths, while retaining an honest hard-baseline
failure. Deployment remains gated on the instrumented Minecraft 1.11.2 oracle
and simulator-vs-server parity.

Files:

- `metrics.jsonl`: per-chunk training receipt.
- `metrics.png`: reward, hits/s, kills, PPO health, and held-out learning plot.
- `evaluation.json`: final 512-seed stationary/random/teacher evaluation.
- `selfplay.pt`: both policies, optimizers, configuration, and final chunk.
