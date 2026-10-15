"""Freeze the preregistered numeric tolerance table before WP2 exists."""
from pathlib import Path
import hashlib
import json
import datetime as dt
import shutil

root = Path('E:/GAN/research/N01')
source = root/'runs/wp1_013_20261015/runtime_fix01_collected_03/runs/primary/artifacts/CALIBRATED_TOLERANCES.json'
independent = root/'receipts/wp1_013_20261015/postrun_independent_01/DECISION_READBACK.json'
target = root/'CALIBRATED_TOLERANCES_WP1_013.json'
receipt = root/'receipts/wp1_013_20261015/postrun_root_01/TOLERANCE_FREEZE_RECEIPT.json'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
table, review = read(source), read(independent)
assert review['passed'] is True and review['T'] == table['T_by_hypothesis']
assert review['decision'] == 'adequate'
assert table['hardware_calibrated'] is False
assert not target.exists() and not receipt.exists()
shutil.copy2(source, target)
assert sha(target) == sha(source)
refs = {str(p.relative_to(root)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in (
    source, target, independent, root/'wp1_013_config.json', root/'WP1_SOURCE_MANIFEST.json',
    root/'OBSERVATION_CONTRACT.json', root/'DATA_ROLES.json', root/'docs/ACCEPTANCE_CONTRACT.md')}
result = dict(schema='n01/wp1-numeric-tolerance-freeze/v1',
    frozen_UTC=dt.datetime.now(dt.timezone.utc).isoformat(), status='frozen_before_WP2_candidate',
    table=str(target.relative_to(root)), table_sha256=sha(target),
    references=refs, rule='max_7_registered_tolerance_cases(mean_2_banks(self_distance))',
    eligible_cases=dict(gain_percent_signed=[-3,3], tail_percent_signed=[-3,3],
        AC_noise_amplitude_percent_signed=[-10,10], extra_Gaussian_width_software_ns=[1]),
    case_winners=review['winning_cases'],
    zero_or_undefined_thresholds='not_calibrated; never automatic PASS',
    immutable_for_following_candidates=True, formula_changed=False,
    RR_used_for_thresholds=False, hardware_calibrated=False,
    independent_check_scope='Recomputed saved self-distance summaries and formula; calibration waveforms were not all retained and original distances were not rerun.',
    current_candidate_scientific_acceptance=False, response_match_not_established=True,
    new_checkpoint_loads=0, new_model_forwards=0, new_AD_queries=0, new_training_updates=0)
with receipt.open('x',encoding='utf-8',newline='\n') as f:
    json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    f.write('\n')
print(json.dumps(dict(status=result['status'], table_sha256=sha(target), receipt_sha256=sha(receipt))))
