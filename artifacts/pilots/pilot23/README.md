# Pilot 23 — latency-aware V2.1

Pilot 23 continues Pilot 22 after correcting the network wrapper to preserve TCP/FIFO packet order. It trains two independent policies in adversarial self-play with per-lane, per-player RTT sampled from 20–200 ms. Each RTT varies smoothly by ±5% during the fight. Uplink actions and downlink observations cross separate tick-quantized delay queues. Each policy receives four delivered observation frames and only its own current normalized ping.

Training also charges 0.02 reward for a weapon switch and awards 0.5 for disabling the opponent's shield. These shape tactical axe use without imposing an action cooldown that vanilla does not have.

## Training

- 1,024 parallel CUDA matches
- 60 × 256-decision PPO chunks
- 20 Hz control
- 20–200 ms independently sampled RTT per player
- smooth ±5% in-fight RTT variation
- initialized from Pilot 22
- approximately 11m55s on one L4

## Evaluation

Every bucket used 256 fights, a 1,200-decision horizon, stochastic policies, and role-swapped evaluation. Native sampled results:

| RTT domain | Completed | Role 0 wins | Role 1 wins | Draws | Attack | Block intent | Axe | Mutual block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 ms | 256 | 109 | 147 | 0 | 2.6% | 48.6% | 30.3% | 22.6% |
| 20–40 ms | 256 | 97 | 159 | 0 | 3.6% | 51.4% | 30.0% | 20.2% |
| 50–100 ms | 256 | 115 | 141 | 0 | 5.6% | 43.3% | 18.4% | 3.9% |
| 100–200 ms | 256 | 109 | 115 | 32 | 16.4% | 12.3% | 0.1% | 0.0% |
| 20–200 ms | 256 | 116 | 131 | 9 | 11.0% | 26.2% | 10.4% | 1.4% |

The behavior changes with the policy's own latency rather than assuming frame-perfect synchronization: lower-ping play uses shields and axes much more, while high-ping play commits to attacks and avoids brittle axe/shield toggling.

## Real-Minecraft smoke deployment

A localhost Minecraft 1.11.2 fight completed in 331 decisions (16.85 s):

- 19.65 measured decisions/s; 19.69 effective Hz
- 10 hits per role
- role 0 dealt 20 damage and won; role 1 dealt 12.87
- action application skew stayed within one world tick
- observed weapon switching: 1.87/s for role 0 and 2.78/s for role 1

The LAN client reports 0 ms Minecraft ping, as expected; real remote servers provide the nonzero per-client value used by the policy input.
