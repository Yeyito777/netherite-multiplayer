# Pilot 16: lower-learning-rate comparison

Corrected continuous-yaw PPO run using the same 52.43-million-decision recipe as
Pilot 15 but with learning rate 1e-5 instead of 3e-5. It completed all held-out
fights, but deployment-equivalent yaw variation was 0.700 degrees/tick versus
Pilot 15's 0.390 and bearing error was 2.98 versus 2.72 degrees. Pilot 15 is the
selected V1.1 candidate; this checkpoint is retained as the controlled recipe
comparison.
