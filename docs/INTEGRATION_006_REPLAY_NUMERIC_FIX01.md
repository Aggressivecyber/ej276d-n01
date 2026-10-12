# Independent replay numerical check correction

The completed id_fix02 scientific endpoint and its source manifest remain
immutable. Its registered decision is `sensitivity_detected`; this correction
does not change any scientific threshold, seed, draw, parameter, waveform or
decision. It applies only to independent numerical QA after completion.

The first verifier exited with code 1 before any response/noise replay forward.
It had decoded the bound datasets and checked the saved event readouts and REAL
curves. A dimensionless conditional matrix A computed by NumPy's generic solve
differed from the saved Torch Cholesky solution by at most
1.3638561223107692e-15. A near-zero entry exceeded the universal 1e-15 absolute
comparison floor. The original verifier, failed receipt and native exit record
are preserved, including the 21,646 REAL rows independently recomputed.

A separate verifier, `scripts/verify_integration006_numeric_fix01.py`, retains
the original independent computations, data checks, exact RNG/waveform hashes,
bootstrap, decisions and immutable bindings. For dimensionless A only, its
absolute comparison floor is fixed at 64 times float64 epsilon
(1.4210854715202004e-14), with the original relative tolerance 3e-10. Other
analytical quantities keep their original atol=1e-15 and rtol=3e-10.

Additionally, both the saved A and independently computed A must satisfy

    ||A K_BB - K_TB||inf /
    (||A||inf ||K_BB||inf + ||K_TB||inf) <= 64 * eps_float64.

Both residuals use the independently constructed, registered NumPy kernel.
Shapes, finiteness and a positive finite denominator are mandatory. This is a
fixed arithmetic check, not a tuned scientific acceptance criterion. Artificial
tests must retain rejection of material perturbations and invalid values.

The new script, tests, independent review, unchanged scientific COMPLETE and
manifest, and original failed verification are hash-bound before the next full
same-device replay. It writes a new `verification_numeric_fix01` receipt and
supervisor directory; it never overwrites the failed verification or science.
All computations reuse the original frozen inputs. No new candidate, seed,
draw or training update is introduced. This QA correction cannot release
formal response training or establish physical noise validity.
