# Pilot 21 — parity-corrected V2

Pilot 21 is the first V2 checkpoint trained after the deterministic parity
campaign corrected vanilla attack-cooldown semantics.

## Recipe

- schema: `iron_gear_20hz_v5`
- 1024 CUDA environments, 256-tick rollouts, 60 chunks
- behavior cloning: 1024 steps / 80 epochs / 0.25 perturbation
- PPO: 2 epochs, LR 3e-5, target KL 0.01
- measured deployment randomization:
  - action hold probability: 0.10 per role
  - extra tick probability: 0.02
- approximately 31.5 million agent decisions
- GPU training time: about 6m25s including periodic evaluation

## Simulator evaluation

All 256 fights complete in every clean and perturbed native/swapped evaluation
mode. Under the deployment-perturbed sampled assignment:

- native: 131 role-0 wins, 124 role-1 wins, 1 draw; 1541/1492 hits
- swapped: 67 role-0 wins, 188 role-1 wins, 1 draw; 1244/1668 hits

Independent adversarial policies remain skill-asymmetric under role swapping;
this is recorded rather than hidden.

## Real Minecraft gate

One stochastic real-Minecraft fight completed in 344 decisions:

- role 1 won;
- damage: role 0 = 11.46, role 1 = 20.0;
- hits: role 0 = 11, role 1 = 17;
- effective cadence: 19.83 Hz;
- p95 decision interval: 52.41 ms;
- exact one-client-tick transitions: 99.71% / 99.42%;
- same-world-tick action application: 97.67%;
- within-one-world-tick application: 99.71%.

See `PARITY_REPORT.md` for the contract, deterministic action tapes, and residual
combat limitations.
