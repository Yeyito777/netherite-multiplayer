# V2 sword / axe / shield self-play

- [x] Freeze the backward-compatible V2 action, observation, and checkpoint contracts.
- [x] Implement deterministic iron armor, sword, axe, shield, cooldown, durability, and blocking mechanics in the shared CPU/CUDA kernel.
- [x] Add focused combat semantics, reset/determinism, symmetry, and CPU/CUDA parity tests.
- [x] Add the V2 actor heads, observations, V1.2 weight transfer, scripted teacher, behavior cloning, and adversarial PPO metrics.
- [x] Equip and observe the V2 loadout in the real Minecraft 1.11.2 oracle.
- [x] Deploy weapon selection and shield-use actions through both real clients.
- [ ] Build Java golden traces for sword/axe/armor/shield interactions and close simulator discrepancies.
- [x] Train and evaluate V2 self-play, diagnose degeneracy or turtling, and adjust the curriculum if needed.
- [ ] Deploy a held-out V2 fight in real Minecraft, record synchronized per-POV audio/video, and publish the receipts.
- [ ] Run the complete test suite, document results, commit, and push the accepted V2 implementation to `main`.
