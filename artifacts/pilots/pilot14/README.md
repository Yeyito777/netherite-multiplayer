# Pilot 14: rejected continuous-yaw diagnostic

First full continuous-yaw run. It exposed a pre-existing PPO rollout bug: the
native environment reused its observation buffer before the trainer cloned it,
so actions and old log probabilities were paired with the next state. The fake
KL was approximately 3.0 and every chunk stopped after one update. The checkpoint
is retained for forensics but is not an accepted adversarial policy.

Its strong smoothness comes mostly from behavioral cloning. Pilot 15 retrains the
same schema after fixing the observation/action pairing.
