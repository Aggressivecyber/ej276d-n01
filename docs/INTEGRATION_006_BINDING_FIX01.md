# Integration006 binding fix01: explicit engineering continuation

The first launch of the registered zero-training integration006 stopped at
`frozen_models`, before `load_training`, `load_validation`, any scientific model
forward, bootstrap draw or optimizer step. It exited with code1 at
2026-10-12T11:51:11Z. Its source commit is
`f697698023bf228155c14dbee66d2a1a8ac76a8e`; its complete source package,
source manifest, FAILURE.json, console and supervisor receipts remain immutable.
The previous output directory will never be reused or cleared.

Diagnosis identified a representation error in the new binding guard. The two
old checkpoint configurations and model statistics match their JSON references.
Their loss statistics have23 Tensor leaves serialized as scalars/lists in the
reference JSON by the original `data.write_json` function. Comparing the raw
checkpoint dictionary directly with the JSON dictionary rejects that format.

This separately recorded engineering continuation is within the user's request
to proceed with the next stage. It permits one fresh execution after the narrow
repair, its artificial regression tests and independent review. It is not an
automatic retry loop or permission to alter a scientific candidate. The original
configuration and protocol hashes, all11 reference hashes, all input bytes,
three hypotheses, seeds, chunks, four draws per parent, bootstrap budget, gates,
zero-training budget and resource bounds are unchanged.

The only scientific-driver change allowed here is exact metadata comparison
after Tensor-to-JSON conversion using `detach().cpu().tolist()`. No tolerance,
rounding, float32 recast, NaN-to-null conversion or key omission is permitted.
The repaired guard must reject changed numbers, keys, shapes and nonfinite
values. Both actual frozen checkpoints must match the reference statistics
exactly after this representation conversion, without model forward or data
decode. Their original files and weights are never changed.

A new source manifest binds the reviewed repair and this amendment before the
first SOURCE/REAL dataset decode or scientific forward. Checkpoint metadata was
decoded during diagnosis; this is recorded rather than misrepresented as a
data-free first launch. The new remote root is
`/home/rylan/projects/ej276d-n01/integration_006_20261012_binding_fix01`.
The local result root is `runs/integration_006_20261012/binding_fix01`.
The initial failed attempt remains under `runs/integration_006_20261012/runs`.

If this continuation encounters another implementation error, preserve it and
diagnose explicitly. Do not clear directories, repeat random draws or silently
change the registered protocol. Any scientific sensitivity or interface failure
is a result to report, never a reason to choose another kernel or train longer.

The final receipt, ledger and release must distinguish both engineering attempts,
the actual completed scientific forward count, identical-input verification
replays and zero training updates. All older negative scientific gates remain.
