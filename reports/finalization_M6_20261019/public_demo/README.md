# Synthetic constrained calibration demo

This example uses one constructed 384-sample pulse, one mean QP and a rank-three analytic covariance. It reads no measured/source event, calibration profile or neural checkpoint. `smooth_mean.py` and `linear_random_transport.py` are exact copies of the frozen mathematical modules used by the evaluated S candidate.

From the package root, in an environment with the versions listed in `runtime/requirements-public-demo.txt`:

```bash
python -B public_demo/run_synthetic.py --output synthetic_replay_01
python -B REBUILD_FIGURES.py
```

The synthetic output directory must be new. The first command writes arrays and a JSON receipt checking reference-area closure, peak preservation, linearity and the analytic covariance identity. The second command rebuilds the six figures from saved aggregate statistics, without model inference. Figure regeneration may differ in rendering bytes across platforms and fonts; its input statistics are hash-identified.

The bundled `verified_result/RESULT.json` records the executed Windows/Python 3.13 check: one QP, reference-area residual 8.12e-18 V ns, linearity error 2.17e-19 V and covariance-identity residual 3.39e-21 V². The constructed mapping query uses absolute carrier coordinates [110,175]. These numerical results are synthetic implementation checks. S failed the measured native-readout protection contract.

An earlier demo supplied local coordinates to that absolute-coordinate interface. The shipped result corrects the demo; preparation spent two one-event QPs in total, with the original receipt retained in the local QA record. The mathematical modules, neural API and scientific candidate evaluation were unchanged.

The direct package-version list is an environment record, not a hermetic cross-platform installer. The private Linux neural-inference environment and its weights are not included in the shareable bundle.
