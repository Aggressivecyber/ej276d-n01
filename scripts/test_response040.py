"""Selection and stream semantics; no dataset, checkpoint or scientific run."""
import argparse
from pathlib import Path
from run_response040 import select_common_candidate
from run_response039 import seed039
from n01.data import write_json

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
records = [dict(candidate_id=1, seeds=[dict(decision=dict(feasible=True, score=.7)),
                                      dict(decision=dict(feasible=False, score=.6))]),
           dict(candidate_id=2, seeds=[dict(decision=dict(feasible=True, score=.9)),
                                      dict(decision=dict(feasible=True, score=.8))]),
           dict(candidate_id=3, seeds=[dict(decision=dict(feasible=True, score=.8)),
                                      dict(decision=dict(feasible=True, score=.9))]),
           dict(candidate_id=4, seeds=[dict(decision=dict(feasible=True, score=.2)),
                                      dict(decision=dict(feasible=True, score=1.))])]
assert select_common_candidate(records)['candidate_id'] == 2
assert select_common_candidate([records[0], records[3]]) is None
for seed in (2026101839, 2026101840):
    for stream in ('shape', 'noise005', 'extension'):
        ids = [seed039(seed, purpose, bank, stream) for purpose in ('FIT', 'CONFIRM', 'CONFIRM040')
               for bank in range(4)]
        assert len(set(ids)) == len(ids)
write_json(args.out, dict(status='PASS', selection_requires_both_seeds=True, tie_preserves_frozen_order=True,
           unchanged_or_infeasible_cannot_select=True, new_confirmation_streams_distinct=True,
           checkpoint_loads=0, dataset_loads=0, AD_queries=0, optimizer_steps=0,
           scientific_validity_established=False))
print('response040 manufactured selection/stream checks: PASS')
