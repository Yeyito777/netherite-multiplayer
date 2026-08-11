# Pilot 12: brake-turn teacher without recovery coverage

Pilot 12 changed the teacher to stop and rotate whenever bearing error exceeded 30
degrees, then ran 80 pure self-play chunks. It trained 41.94 million decisions,
1,308,249 hits, and 44,312 kills.

Held-out sampled self-play improved mean bearing error from pilot 10's 73.97
degrees to 18.71 and time-to-death from 1,118.5 to 470.1 Minecraft ticks. However,
forward-while-behind remained common because teacher trajectories almost never
visited recovery states; high aggregate BC accuracy therefore concealed a
covariate-shift failure.

Verdict: diagnostic success, not final candidate. It motivated deterministic
DAgger-style action disturbances during teacher data collection.
