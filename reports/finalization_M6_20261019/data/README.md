# Data dictionary

All tables retain the complete executed condition grid. `head` is the original frozen model-head index; `H` is the noise-extension hypothesis. Neither is an acquisition identifier. Each system/condition evaluates 6,593 SOURCE parents against the same 21,047 REAL records.

|File|Meaning|
|---|---|
|METRICS_ALL_CONDITIONS.csv|24 rows: M0, BASE054, 070, S × two heads × three H. Metric columns are distributional distances from REAL, including columns ending in `_rate`; they are not observed population means or absolute event rates.|
|CONDITIONAL_MEANS_AND_DENOMINATORS.csv|192 cells using eight frozen common output-Q bins. `fake_rows` is exported as `source_rows`. Means/medians use defined values; defined and undefined counts are separate for every quantity and population.|
|PROTECTION_FAILURE_CARD.csv|48 S-vs-M0 component comparisons with raw distances, research T, D/T, ΔD/T and original 0.05T allowance. Positive T is an artificial research scale, not hardware calibration.|
|NEGATIVE_LOBE_FAILURE_CARD.csv|Six separate W1-distance comparisons for integrated negative voltage. The original 039 rule permits no distance increase relative to M0.|
|FOUR_SYSTEM_COMPARISON_STATUS.csv|Planned P0/P1/CPU-Net/N01-final status. Blank score cells mean no result; they must not be read as zero.|
|PLANNED_EXPERIMENT_STATUS.csv|Executed, negative, untriggered and unrun experiments kept distinct.|
|PUBLIC_EVIDENCE_SUMMARY.json|Compact aggregate result and engineering verification counts.|
|FIGURE_DATA.json|Population-level curves, moments, covariance and signed decomposition summaries used by the six figures. No individual waveform or parent labels.|
|AGGREGATION_RECEIPT.json|The 41 pre-existing source files read to form the aggregates and their byte identities.|

For scalar distances, Q/front/tail/negative area use V ns, voltage and baseline use V, rise/FWHM use software ns, and PSD uses a dimensionless ratio. Composite wave, QPSD, noise and joint distances follow their frozen feature normalization. The `wave` metric is the original W1-based multiview composite, not the earlier energy training loss.

Q-bin membership is system-specific under common fixed boundaries. The corresponding REAL median-Q coordinate does not pair an individual SOURCE event with a REAL event. Displayed population covariance within one output-Q bin combines different events and cannot identify a same-parent response kernel.
