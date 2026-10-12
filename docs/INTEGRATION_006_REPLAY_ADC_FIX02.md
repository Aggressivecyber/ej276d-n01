# Independent replay ADC representation correction

The completed id_fix02 science, original source manifest, original verifier,
numeric_fix01 verifier, and both failed QA executions remain immutable.
The scientific decision remains `sensitivity_detected`. This correction changes
no scientific threshold, candidate, random input, model or scientific output.

The numeric_fix01 verifier passed its analytical checks and reached the first
128-row TRAIN deterministic replay batch. Noise, extension, signal and all five
generated-view hashes matched the completed endpoint. It then failed the old
baseline-change check: CUDA ADC voltages were compared with NumPy-reconstructed
ADC voltages using exact floating inequality. Saved-batch inspection reproduced
the maximum false changed-fraction 0.17. The maximum voltage difference was
1.1102230246251565e-16 V (1.82e-12 ADC steps); integer-code mismatches were zero.
The saved signal prefix was exactly zero.

The failing path and first saved-batch diagnostic locate the interruption before
the second arm of the first chunk. This QA attempt repeated 128 noise forwards,
384 extension forwards and 128 deterministic response forwards, plus 384 new
candidate ADC rows and 128 signal-reference ADC rows. It also independently
recomputed 21,646 REAL readouts and the 2,000 registered paired bootstrap samples.
These are reconstructed execution counts from the immutable failing path and
saved first batch; the failed verifier did not persist live per-forward counters.
They are not new unique random draws or training updates.

A separate `scripts/verify_integration006_adc_fix02.py` preserves numeric_fix01
and changes the baseline-change reference to the original same-device
`digitize(prefix, ste=False)` operation. This restores the driver's precise
meaning instead of comparing differently rounded voltage representations.

It additionally checks every replayed ADC view at all 480 samples with an
independent NumPy calculation: finite arrays and exact shape; integer codes in
[0,16383]; output codes exactly equal nearest-even rounding of the clipped
pre-ADC input; and output voltages within 4 times float64 epsilon V
(8.881784197001252e-16 V) of `code/16383 - 0.5`. Codes have zero tolerance.
The tiny voltage-reconstruction allowance does not permit an off-grid value
such as 0.1 ADC step. Waveform byte hashes still must match exactly.

Artificial tests cover equivalent-code voltage representations, nonzero
baseline changes, nearest-even boundaries, clipping, wrong code, off-grid
voltage, nonfinite input and shape errors. The new source, tests, review,
original failures, unchanged COMPLETE and original manifest are hash-bound
before the next full replay. New receipts use `verification_adc_fix02` and a
new supervisor directory. This is a QA correction, not a scientific retry or
an authorization to train the response model.
