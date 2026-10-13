"""Exercise the resume validator on CPU state with non-model metadata doubles."""
from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone

import torch

from metadata_audit import ROOT, OUT, HYPOTHESES, STOCHASTIC, sha

sys.path.insert(0, str(ROOT / 'src'))
from n01 import response008_resume01 as implementation


class MetadataModule:
    """Returns saved tensors; never constructs or executes a scientific model."""
    def __init__(self, state, signature):
        self.state = state
        self.signature = signature

    def state_dict(self):
        return self.state

    def named_parameters(self):
        return [(row['name'], self.state[row['name']]) for row in self.signature]


class MetadataOptimizer:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state


class MetadataDiscriminators(dict):
    def __init__(self, state):
        self.state = state
        for h in HYPOTHESES:
            own = {name.split('.', 1)[1]: value for name, value in state.items() if name.startswith(h + '.')}
            signature = [dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
                         for name, value in own.items() if name.endswith(('.bias', '.weight_orig'))]
            self[h] = MetadataModule(own, signature)

    def state_dict(self):
        return self.state


def validator(cp):
    engine = object.__new__(implementation.Response008Resume01Engine)
    engine.config = cp['config']
    engine.engineering_fixture = False
    engine.fix01_identity = copy.deepcopy(implementation.REGISTERED_FIX01_IDENTITY)
    engine.parameter_signature = cp['parameter_signature']
    engine.statistics = cp['statistics']
    engine.common_state_digest = cp['common_initial_state_sha256']
    engine.initial_state_digest = cp['initial_state_sha256']
    engine.initial_D_digest = cp['initial_D_sha256']
    engine.noise_state_digest = cp['noise005_state_sha256']
    engine.model = MetadataModule(cp['model'], cp['parameter_signature'])
    engine.discriminators = MetadataDiscriminators(cp['discriminators'])
    engine.g_optimizer = MetadataOptimizer(cp['g_optimizer'])
    engine.d_optimizers = {h: MetadataOptimizer(cp['d_optimizers'][h]) for h in HYPOTHESES}
    engine.source = range(6593)
    engine.real = range(21047)
    return engine


def main():
    torch.set_num_threads(2)
    source = ROOT / 'src/n01/response008_resume01.py'
    source_sha = sha(source)
    cp = torch.load(ROOT / 'runs/window008_fix01_20261013/runs/science/train/full/latest.pt',
                    map_location='cpu', weights_only=False)
    engine = validator(cp)
    steps = engine._validate_import_state(cp, implementation.REGISTERED_FIX01_IDENTITY)
    assert len(steps) == 180 and set(steps.values()) == {6000, 3000}
    results = [dict(case='actual_full6000_positive_metadata_control', passed=True, rejected=False)]
    shared = next(i for i, row in enumerate(cp['parameter_signature']) if not row['name'].startswith(STOCHASTIC))
    stochastic = next(i for i, row in enumerate(cp['parameter_signature']) if row['name'].startswith(STOCHASTIC))

    def reject(name, alter):
        candidate = copy.deepcopy(cp)
        alter(candidate)
        try:
            engine._validate_import_state(candidate, implementation.REGISTERED_FIX01_IDENTITY)
        except (ValueError, RuntimeError, TypeError, KeyError) as error:
            results.append(dict(case=name, passed=True, rejected=True, rejection=str(error)))
        else:
            results.append(dict(case=name, passed=False, rejected=False))

    reject('failure_purpose', lambda s: s.update(checkpoint_purpose='failure'))
    reject('mid_step_flag', lambda s: s.update(failure_may_be_mid_step=True))
    reject('automatic_resume_true', lambda s: s.update(automatic_resume_allowed=True))
    reject('wrong_schema', lambda s: s.update(schema='response008/v1'))
    reject('wrong_arm', lambda s: s.update(arm='simple'))
    reject('artificial_role_mismatch', lambda s: s.update(engineering_fixture=True))
    reject('wrong_config_seed', lambda s: s['config']['seeds'].update(train_response=0))
    reject('wrong_fix01_identity', lambda s: s['recovery'].update(config_sha256='0' * 64))
    reject('wrong_model_implementation', lambda s: s.update(model_implementation='n01.model008.N01Response008'))
    reject('wrong_signature_dtype', lambda s: s['parameter_signature'][0].update(dtype='torch.float64'))
    reject('G_6398_is_not_boundary', lambda s: s.update(G_updates=6398))
    reject('D_partial_hypothesis', lambda s: s['D_updates'].update(fast=3001))
    reject('wrong_new_counters', lambda s: s.update(new_G_updates=3001))
    reject('wrong_parent_lineage', lambda s: s['lineage'].update(source_checkpoint_sha256='0' * 64))
    reject('noise_update_present', lambda s: s.update(noise_updates=1))
    reject('population_update_present', lambda s: s.update(population_learning=True))
    reject('statistics_mutated', lambda s: s['statistics'].update(engineering_fixture=True))
    reject('frozen_initializer_mutated', lambda s: s.update(common_initial_state_sha256='0' * 64))
    reject('G_step_uniform_6000_on_stochastic', lambda s: s['g_optimizer']['state'][stochastic]['step'].fill_(6000))
    reject('G_step_uniform_3000_on_shared', lambda s: s['g_optimizer']['state'][shared]['step'].fill_(3000))
    reject('G_missing_slot', lambda s: s['g_optimizer']['state'].pop(shared))
    reject('G_positional_ID_reorder', lambda s: s['g_optimizer']['param_groups'][0]['params'].reverse())
    reject('G_hyperparameter_changed', lambda s: s['g_optimizer']['param_groups'][0].update(lr=0.1))
    reject('G_nonfinite_moment', lambda s: s['g_optimizer']['state'][shared]['exp_avg'].fill_(float('nan')))
    reject('G_negative_second_moment', lambda s: s['g_optimizer']['state'][shared]['exp_avg_sq'].fill_(-1))
    reject('G_wrong_moment_dtype', lambda s: s['g_optimizer']['state'][shared].update(exp_avg=s['g_optimizer']['state'][shared]['exp_avg'].double()))
    reject('D_wrong_step', lambda s: s['d_optimizers']['fast']['state'][0]['step'].fill_(6000))
    reject('D_missing_slot', lambda s: s['d_optimizers']['slow']['state'].pop(0))
    reject('D_hypothesis_missing', lambda s: s['d_optimizers'].pop('common_level'))
    reject('SOURCE_sampler_stale_position', lambda s: s['source_sampler'].update(position=s['source_sampler']['position'] - 1))
    reject('REAL_sampler_visit_missing', lambda s: s['real_sampler']['visits'].__setitem__(0, 0))
    reject('SOURCE_order_not_permutation', lambda s: s['source_sampler']['order'].__setitem__(0, s['source_sampler']['order'][1]))
    reject('paired_stream_missing', lambda s: s['streams'].pop('extension'))
    reject('paired_stream_bad_dtype', lambda s: s['streams'].update(response=s['streams']['response'].float()))
    reject('paired_stream_bad_shape', lambda s: s['streams'].update(noise005=s['streams']['noise005'][:16]))
    reject('last_metric_counter_mismatch', lambda s: s['last_metrics'].update(G_updates=5999))
    reject('projection_update_mismatch', lambda s: s['last_metrics']['projection_context'].update(update=5999))

    clone = implementation.cpu_tree(cp['g_optimizer'])
    implementation.assert_tree_bitwise_equal(cp['g_optimizer'], clone, 'optimizer_clone')
    assert clone['state'][shared]['exp_avg'].data_ptr() != cp['g_optimizer']['state'][shared]['exp_avg'].data_ptr()
    results.append(dict(case='recursive_CPU_optimizer_clone_and_bytes', passed=True, rejected=False))
    try:
        implementation.assert_tree_bitwise_equal(torch.tensor([0.0]), torch.tensor([-0.0]), 'signed_zero')
    except ValueError:
        results.append(dict(case='signed_zero_tensor_bytes_differ', passed=True, rejected=True))
    else:
        results.append(dict(case='signed_zero_tensor_bytes_differ', passed=False, rejected=False))
    try:
        engine.restore('not_a_resume_path')
    except ValueError as error:
        results.append(dict(case='generic_restore_prohibited', passed=True, rejected=True, rejection=str(error)))
    else:
        results.append(dict(case='generic_restore_prohibited', passed=False, rejected=False))
    try:
        engine.import_legacy_full_stage3000('not_a_resume_path', expected_sha256='0' * 64)
    except ValueError as error:
        results.append(dict(case='old_stage3000_import_prohibited', passed=True, rejected=True, rejection=str(error)))
    else:
        results.append(dict(case='old_stage3000_import_prohibited', passed=False, rejected=False))
    assert sha(source) == source_sha, 'Engine changed during metadata review'
    review = dict(schema='n01/window008-resume01-independent-engine-counterexamples/v1',
                  UTC=datetime.now(timezone.utc).isoformat(), source_sha256=source_sha,
                  status='passed' if all(row['passed'] for row in results) else 'failed', cases=results,
                  positive_actual_checkpoint_metadata=True, scientific_model_construction=False,
                  scientific_model_forwards=0, optimizer_objects_created=0, optimizer_updates=0,
                  DEV_numeric_decodes=0, GPU_calls=0, new_scientific_jobs=0,
                  scope='Pure _validate_import_state and CPU tree utilities with metadata doubles; native engine import and execution are separate.')
    (OUT / 'ENGINE_METADATA_COUNTEREXAMPLES.json').write_text(json.dumps(review, indent=2) + '\n', 'utf8')
    print(json.dumps(dict(status=review['status'], cases=len(results), source_sha256=source_sha,
                          failed=[row for row in results if not row['passed']]), indent=2))


if __name__ == '__main__':
    main()
