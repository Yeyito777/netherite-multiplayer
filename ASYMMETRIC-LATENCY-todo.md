# Asymmetric-latency control investigation and repair

- [x] Preserve the Pilot 23 failure as a deterministic regression test and ordered latency matrix.
- [x] Replace aggregate-only latency evaluation with per-role 2D RTT matrix diagnostics.
- [x] Add a balanced training curriculum that deliberately oversamples large asymmetric RTT gaps and alternates the disadvantaged role.
- [x] Make the behavioral teacher predict target motion through its own observation plus action delay.
- [x] Add modest look-stability shaping and report yaw saturation/variation by latency side.
- [ ] Train a feed-forward repair candidate and require disadvantaged-player hits in every matrix direction.
- [ ] If the feed-forward repair fails, add recurrent temporal state and retrain.
- [ ] Run CPU/CUDA and deployment regression suites.
- [ ] Deploy the accepted candidate under asymmetric simulated ping and numerically verify both agents engage.
- [ ] Record/send dual-POV videos only after the asymmetric skill gates pass.
- [ ] Commit and push the completed repair campaign.
