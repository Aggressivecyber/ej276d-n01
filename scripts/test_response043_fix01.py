"""Regression for actual runner replay block, including arrays and real mismatches."""
import argparse
import ast
import copy
import json
from pathlib import Path

import numpy as np
import torch
from n01.data import write_json
import run_response043_fix01 as fixed

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
scratch = args.out.parent/'replay_regression_cases'
scratch.mkdir(parents=True, exist_ok=False)
tree = ast.parse(Path(fixed.__file__).read_text())
blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
          and ast.unparse(node.test) == "row['candidate_id'] == 0"]
assert len(blocks) == 1
replay_code = compile(ast.Module(body=blocks, type_ignores=[]), fixed.__file__, 'exec')
prior = {'fast': {'values': {'wave': .25, 'undefined': None},
    'fake_flags': {'bad': {'count': 1, 'denominator': 4}},
    'observation_flags': {'crop': {'count': 0, 'denominator': 4}},
    'conditional': {'counts': [2, 2], 'values': [[.1, .2], [.3, .4]]}}}
(scratch/'prior.json').write_text(json.dumps(prior), encoding='utf-8')
native = copy.deepcopy(prior)
native['fast']['conditional']['counts'] = np.array([2, 2], dtype=np.int64)
native['fast']['conditional']['values'] = torch.tensor([[.1, .2], [.3, .4]], dtype=torch.float64)
native['fast']['values']['wave'] = np.float64(.25)
original_failure = False
try:
    bool(native != prior)
except ValueError as error:
    assert 'truth value of an array' in str(error)
    original_failure = True
assert original_failure
checks = {}
for case in ('equal_arrays', 'scalar_changed', 'flag_changed', 'array_order_changed', 'undefined_changed'):
    report = copy.deepcopy(native)
    if case == 'scalar_changed': report['fast']['values']['wave'] += .01
    if case == 'flag_changed': report['fast']['fake_flags']['bad']['count'] = 2
    if case == 'array_order_changed': report['fast']['conditional']['values'] = report['fast']['conditional']['values'].flip(0)
    if case == 'undefined_changed': report['fast']['values']['undefined'] = 0.
    out = scratch/case
    scope = dict(row={'candidate_id': 0}, index=0, report=report, out=out, ROOT=scratch,
        p={'base_replay_files': ['prior.json']}, r=fixed.r, json=json)
    rejected = False
    try:
        exec(replay_code, scope)
    except RuntimeError as error:
        assert str(error) == '043 zero diffusion does not exactly replay042'
        rejected = True
    assert rejected == (case != 'equal_arrays')
    assert (out/'seed0_BASE042_REPLAY.json').exists() == (case == 'equal_arrays')
    if case == 'equal_arrays':
        assert json.loads((out/'seed0_BASE042_CURRENT.json').read_text()) == prior
        assert json.loads((out/'seed0_BASE042_REPLAY.json').read_text())['full_report_exact']
    checks[case] = 'PASS'
write_json(args.out, dict(status='PASS', actual_fixed_runner_block_executed=True,
    original_array_comparison_bug_reproduced=True, checks=checks,
    numpy_array_tensor_numpy_scalar_and_undefined_preserved=True,
    full_report_equality_required=True, checkpoint_loads=0, dataset_loads=0, AD_queries=0,
    scientific_acceptance=False))
print('response043_fix01 actual replay block regression: PASS')
