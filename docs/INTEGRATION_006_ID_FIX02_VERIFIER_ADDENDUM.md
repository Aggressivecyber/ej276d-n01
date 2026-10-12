# Pre-execution verifier portability addendum for id_fix02

The complete artificial end-to-end regression now exercises the independent
verifier on Windows. It found that NumPy's default integer dtype differs across
the local Windows fixture and the registered Linux runtime. The scientific
driver already saves parent-row indices, draw indices and fixed-bin parent
counts as explicit int64. The independent verifier must generate references
of that same declared type rather than relying on a platform default.

Before id_fix02 source freeze, five verifier statements were narrowed to add
explicit `dtype=np.int64` for those identity/count references. Values, shapes,
ordering, byte comparisons, numerical tolerances, readouts, bootstrap and
decision rules are unchanged. Linux reference bytes are unchanged; Windows
now checks the declared int64 schema. The old verifier bytes are preserved at
`provenance/integration006_before_id_fix02/verify_integration006.py`.

This is part of the registered identity-serialization engineering correction,
with its addendum hash bound by the new source manifest and independent review.
It does not amend any scientific configuration, registered hypothesis, gate,
random seed, dataset or model parameter. Prior nonzero partial execution counts
remain as recorded in the main id_fix02 amendment.
