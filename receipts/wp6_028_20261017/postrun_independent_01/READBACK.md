# WP6-028 independent saved-output review

Passed with no blockers in the registered software scope. One private NPZ was read once: 37 members, 182,940 values, 1,386,296 payload bytes. All 12 call tensor hashes and 60 latent hashes match. Replay, common random libraries/prefix/scale, chargeoff M0 and shapeoff A identities are byte-exact. All 48 rows pass saved-ADC geometry, lineage, seven flags and PSD algebra checks; no ADC was rerun.

The fixed float64 reduction bounds were applied without retuning:

| Reduction | Maximum absolute residual | Maximum allowed bound |
| --- | ---: | ---: |
| Q | 3.1086244689504383e-15 | 4.4545196981633846e-13 |
| tail | 6.6613381477509392e-16 | 1.0262664022226321e-13 |
| baseline_mean_V | 0 | 2.3627020568981164e-17 |

No original CP/DATASET, torch/n01, NN, RNG, ADC or training was used. Native receipt: exit 0, 12 calls, 48 response rows; frozen states unchanged. Research status remains response_match_not_established; weights are not promoted, no deployment seed is selected, and full WP5/WP6 is incomplete.
