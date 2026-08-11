# Netherite Multiplayer

High-throughput Minecraft **1.11.2** simulation in C/CUDA, extended with
deterministic two-player shared worlds, adversarial self-play, and deployment of
the trained policies into real Forge clients.

This project builds on [Infatoshi/netherite](https://github.com/Infatoshi/netherite).
Our main branch adds a fixed two-player PvP environment, CPU/CUDA parity tests,
20 Hz fine-control policies, training and evaluation tooling, and a
real-Minecraft deployment bridge for private test servers.

## Boxing MVP

- 32x32 stone arena, two unarmored players, fist combat only.
- Two independent feed-forward actor-critic policies trained with behavioral
  cloning, PPO, and adversarial self-play.
- Batched CPU and CUDA backends with deterministic shared-state combat.
- Versioned 20 Hz controls: V1's nine-value yaw grid, V1.1's continuous yaw,
  and the experimental V1.2 continuous yaw+pitch actor.
- Frozen V1 Pilot 13, V1.1 Pilot 15, and experimental Pilot 17 checkpoints,
  each trained for 52.43 million adversarial decisions after DAgger cloning.
- Real Minecraft 1.11.2 deployment with separate full-frame client recordings.

Start with [`docs/MULTIPLAYER_MVP.md`](docs/MULTIPLAYER_MVP.md),
[`PVP_SKILL_REPORT.md`](PVP_SKILL_REPORT.md),
[`artifacts/pilots/pilot13/README.md`](artifacts/pilots/pilot13/README.md),
[`artifacts/pilots/pilot15/README.md`](artifacts/pilots/pilot15/README.md), and
[`artifacts/pilots/pilot17/README.md`](artifacts/pilots/pilot17/README.md).

## Monorepo map

| Path | Purpose |
|---|---|
| `blaze/pvp/` | Two-player C/CUDA simulator, Python wrapper, trainer, evaluator, and parity tests |
| `java/Minecraft/` | Forge 1.11.2 oracle/mod and dual-client bridge |
| `java/deploy_pvp_checkpoint.py` | Closed-loop policy deployment into two real clients |
| `verify/trace/` | Deterministic scheduler and trace-contract tests |
| `artifacts/pilots/pilot13/` | Accepted frozen checkpoints, metrics, plots, and evaluations |
| `artifacts/fights/` | Compact real-deployment receipts; generated videos are stored externally |

## Upstream Netherite

The underlying project is a from-scratch C/CUDA Minecraft 1.11.2 implementation
(bit-verified against the Java game) with batched CUDA RL:

<p align="center">
  <img src="docs/assets/zoom_farm.gif" width="800"
       alt="one agent's observation zooming out to 7,200 live batched worlds">
</p>
<p align="center"><i>One env's semantic camera, zooming out to 7,200 live
worlds stepping in lockstep on one GPU (recorded from a real batch).</i></p>

## Platforms

| | Support |
|--|---------|
| **Linux x86_64** | Full stack (build, CUDA train, Java oracle). |
| **macOS** | Viewer / SSH only (Moonlight, mcwindow). No native game or CUDA train. |
| **Windows** | Not supported as a build host. |

No Mojang content is shipped. You need a legal Minecraft ownership and JDK 8.

## License

The multiplayer/PvP additions authored in this repository are available under
the MIT License. Pre-existing Netherite, Forge, Malmo, Minecraft, and third-party
components retain their own copyright and license status. See [`LICENSE`](LICENSE)
and [`NOTICE.md`](NOTICE.md); the root MIT grant does not relicense upstream code.

## Using an LLM on this repo

Open **[`AGENTS.md`](AGENTS.md)** (Claude also loads [`CLAUDE.md`](CLAUDE.md)). Or paste:

```
Read AGENTS.md in this repo and follow it. Task: <what you want done>
```

## Clean Linux box (one command)

```bash
bash scripts/setup_and_verify.sh          # bootstrap + build + --quick sweep
bash scripts/setup_and_verify.sh --demo   # + physics/pixel tape replay + SBS MP4
# -> demos/pixel_match_sbs.mp4  (oracle | magma side-by-side)
```

Prism is optional. Bootstrap uses ForgeGradle; details in [`docs/BOOTSTRAP.md`](docs/BOOTSTRAP.md).
Pixel demo uses the shipped canonical tape under `verify/demo/`.
