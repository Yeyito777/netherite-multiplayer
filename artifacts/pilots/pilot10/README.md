# Pilot 10: extended adversarial boxing self-play

Pilot 10 used a 50-epoch class-balanced boxing-teacher warm start, a 10-chunk
stationary curriculum, a 10-chunk scripted-boxer curriculum, and 100 chunks where
two independently optimized policies fought only each other. It trained 31.46
million agent decisions on the L4 and recorded 1,381,869 hits and 13,584 kills.

A separate 256-lane checkpoint tournament ran each matchup for at most 600 policy
decisions. Every greedy and sampled fight reached a terminal death rather than a
horizon timeout. Greedy fights averaged 270.4 decisions to death; sampled fights
averaged 279.6. Simultaneous knockouts are counted as draws by the deterministic
combat kernel.

Sampled head-to-head result:

- role 0 wins: 23;
- role 1 wins: 24;
- simultaneous knockouts/draws: 209;
- hits: 3,211 vs 3,197;
- horizon timeouts: 0.

This near-balanced sampled tournament is the deployment candidate. The high draw
rate is expected from symmetric unarmed damage and simultaneous intent commit,
but both independently trained agents reliably engage and finish fights.
