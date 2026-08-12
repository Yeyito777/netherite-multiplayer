# V2 sword / axe / shield self-play

- [x] Freeze the backward-compatible V2 action, observation, and checkpoint contracts.
- [ ] Implement deterministic iron armor, sword, axe, shield, cooldown, durability, and blocking mechanics in the shared CPU/CUDA kernel.
- [ ] Add focused combat semantics, reset/determinism, symmetry, and CPU/CUDA parity tests.
- [ ] Add the V2 actor heads, observations, V1.2 weight transfer, scripted teacher, behavior cloning, and adversarial PPO metrics.
- [ ] Equip and observe the V2 loadout in the real Minecraft 1.11.2 oracle.
- [ ] Deploy weapon selection and shield-use actions through both real clients.
- [ ] Build Java golden traces for sword/axe/armor/shield interactions and close simulator discrepancies.
- [ ] Train and evaluate V2 self-play, diagnose degeneracy or turtling, and adjust the curriculum if needed.
- [ ] Deploy a held-out V2 fight in real Minecraft, record synchronized per-POV audio/video, and publish the receipts.
- [ ] Run the complete test suite, document results, commit, and push the accepted V2 implementation to `main`.
