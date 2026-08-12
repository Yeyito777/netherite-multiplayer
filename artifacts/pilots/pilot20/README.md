# Pilot 20: V2 iron gear tactical self-play

Pilot 20 is the accepted first V2 candidate. It transfers Pilot 17's continuous
20 Hz yaw/pitch and locomotion trunk, expands the observation from 25 to 35
values, and adds sword/axe selection plus exclusive idle/attack/shield intent.

Training used 1,024 CUDA lanes, 1,024-step disturbed behavior cloning, a
20-chunk scripted tactical bootstrap, and 40 adversarial PPO chunks. The frozen
checkpoint has two independent 23,444-parameter policies.

A 256-lane held-out evaluation completed every fight. Native/swapped greedy
assignments produced 195/61 and 115/141 outcomes with no draws. Hybrid sampled
movement/combat produced 148/107/1 and swapped 91/165/0. Tactical use was
nondegenerate: axes were selected about 16-19% of player ticks, shield intent
about 33-36%, and mutual active blocking only 7-10% of lane ticks.

The real 1.11.2 deployment completed a held-out fight at 19.85 decisions/s:
role 0 won after 350 decisions, with accepted hits 20/14 and damage 20.0/15.36.
Its actions included 67/101 axe-selected ticks, 208/145 shield-use ticks, and
36/45 weapon switches, demonstrating that all new controls transferred.
