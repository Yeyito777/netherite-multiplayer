# Training ↔ deployment parity campaign

- [x] Define measurable tick/action/observation parity contract and acceptance gates.
- [x] Instrument each real-client policy transition with client tick, world/server epoch, send/receive timing, and action sequence.
- [x] Add deployment timing summary (Hz, jitter, p50/p95/p99, skipped/duplicated ticks, two-client skew).
- [x] Add deterministic action-tape runner and parity fixtures for idle, movement/look, weapon switch, shield, and attacks.
- [x] Run baseline real-Minecraft timing/tape campaign against Pilot 20 and archive a report.
- [x] Fix avoidable deployment timing/action-semantic mismatches discovered by the baseline.
- [x] Re-run parity gates and document residual sim-to-real mismatch.
- [x] Add narrowly targeted training randomization/evaluation for measured residual delay/jitter.
- [x] Train and evaluate a corrected, deployment-randomized V2 candidate.
- [x] Run regression tests, commit, and push the parity campaign state.
