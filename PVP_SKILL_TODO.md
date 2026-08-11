# Netherite PvP skill proof TODO

Overarching objective: prove that two independently trained adversarial agents can
box *well* at 20 Hz with precise pursuit and reacquisition, rather than merely
landing occasional hits while orbiting.

Work proceeds top-to-bottom. An item is checked only after its implementation and
its evidence are committed.

- [x] Establish a quantitative pilot-10 orbiting baseline: bearing error, yaw-action saturation, forward-while-behind fraction, reach occupancy, hits, deaths, and decisions-to-death.
- [x] Version the boxing observation/action/checkpoint contract so legacy 5 Hz checkpoints cannot be silently interpreted as fine-control policies.
- [x] Add a first incremental 20 Hz controller with a finer bounded yaw action grid and remove unused pitch exploration from the boxing policy.
- [x] Add deterministic tests for fine yaw decoding, one-tick action semantics, action symmetry, observation parity, and CPU/CUDA trajectory parity.
- [x] Extend training and evaluation receipts with pursuit-quality metrics and explicit acceptance thresholds against pilot 10.
- [x] Train and evaluate the first 20 Hz adversarial self-play pilot on the stopped-budget GCloud L4 workflow.
- [x] Diagnose that pilot against the baseline; adjust yaw parameterization, curriculum, movement coupling, or reward shaping if it does not materially reduce orbiting.
- [x] If the discrete fine-control pilot remains quantization-limited, implement and test a bounded continuous-yaw PPO head with correct sampling and log probabilities. *(Not triggered: pilot 13 reaches 2.34-degree sampled bearing error with only 2.17% yaw saturation.)*
- [x] Train the final two-policy adversarial model and run held-out, role-swapped, randomized-start tournaments with no training updates.
- [x] Require a skill gate: materially lower bearing/orbit error and time-to-damage than pilot 10, reliable fight completion, balanced role outcomes, and no regression in CPU/CUDA parity.
- [x] Align the Java deployment adapter with the final 20 Hz action contract and add randomized asymmetric real-match starts.
- [x] Deploy both final policies into real private Minecraft 1.11.2 and run repeated matches, preserving authoritative vanilla hit/damage/death receipts.
- [x] Record a representative improved fight that visibly demonstrates tight reacquisition and sustained boxing rather than perimeter orbiting.
- [x] Publish final plots, checkpoint, simulator/real receipts, video evidence, limitations, and a concise architecture/training report.
- [x] Stop all billable GPU resources, verify repository cleanliness/tests, and complete the overarching goal.
