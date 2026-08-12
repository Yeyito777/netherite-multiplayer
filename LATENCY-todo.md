# V2.1 latency-robust self-play

- [x] Define the per-player RTT, jitter, action-delay, observation-delay, and policy-input contract.
- [x] Implement a deterministic CPU/CUDA-compatible network-domain wrapper with independent per-player latency queues.
- [x] Add each policy's own current ping to its observation without exposing the opponent's ping.
- [x] Vary each player's ping continuously within ±5% of its episode baseline.
- [x] Add a V2.1 checkpoint/action/observation schema and transfer the V2 policy into its expanded input layer.
- [x] Make evaluation report behavior and outcomes across clean and latency-randomized conditions.
- [x] Make real-Minecraft deployment measure/feed each client's own RTT to the corresponding policy.
- [x] Add tests for delay semantics, ping privacy/range, masked resets, checkpoint compatibility, and deployment inputs.
- [x] Run CPU and CUDA regression/parity tests.
- [x] Train an adversarial V2.1 candidate and evaluate it under multiple latency profiles.
- [x] Deploy the candidate to real Minecraft, verify 20 Hz timing, and inspect dual-POV behavior.
- [x] Commit and push the completed V2.1 campaign.
