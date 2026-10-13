"""Hash-bound Window008 resume01, retaining the complete fix01 cycle6000.

This adapter keeps model008_fix01 and the registered Window008 step/VJP intact.
Its only scientific import is the complete periodic fix01 checkpoint6000;
failure checkpoints and generic/automatic restoration remain inadmissible.
"""
from __future__ import annotations

import copy
from pathlib import Path
import re

import numpy as np
import torch

from .data import sha256
from .response008 import HYPOTHESES, Response008Engine, cpu_state, state_digest, statistics_digest
from .response008_fix01 import (MODEL_IMPLEMENTATION, RUN_ID as FIX01_RUN_ID,
    SCHEMA as FIX01_SCHEMA, STOCHASTIC_PREFIXES, Response008Fix01Engine,
    assert_tree_equal, optimizer_named_state, parameter_signature, require,
    tree_digest, validate_sampler, validate_streams)
from .train import atomic_torch_save, utc


RUN_ID = 'window008_resume01_20261013'
SCHEMA = 'response008_resume01/v1'
REGISTERED_IMPORT_SHA256 = '921ec4c58387a5ad53fdf55ade10169930536106141e9070f5e4a8df60ab6c58'
LEGACY_STAGE3000_SHA256 = '48a20d256c3d87f2f087aa6a3cc3a2dd9279715c137ad1d5c073dcd22bfaab9c'
IDENTITY_KEYS = {'run_id', 'config_sha256', 'source_manifest_sha256',
                 'base_configuration_sha256', 'base_source_manifest_sha256',
                 'fix01_configuration_sha256', 'fix01_source_manifest_sha256'}
REGISTERED_FIX01_IDENTITY = dict(
    run_id=FIX01_RUN_ID,
    config_sha256='1fc57e0daf06f3eadf2a457361a4342c953f265039cec48b9401616c703d6402',
    source_manifest_sha256='2a28ecca10b3815ac7af0387eab19072ff92df0f90f196cac79a86f20a1f0180',
    base_configuration_sha256='f5500c65f2d17a2fe6964bfc5908dedb68c27b9391cb30bf30c0ba161509d145',
    base_source_manifest_sha256='97b020dc68e41fa1e893e66fdf1091b1bf9500b98281a5dd7e2545ee9ce00929')


def assert_tree_bitwise_equal(first, second, label='state'):
    """Compare complete CPU/native state, including tensor bytes and types."""
    if isinstance(first, torch.Tensor):
        require(isinstance(second, torch.Tensor) and first.dtype == second.dtype
                and first.shape == second.shape, 'Tensor dtype/shape differs: ' + label)
        left = first.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        right = second.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        require(torch.equal(left, right), 'Tensor bytes differ: ' + label)
    elif isinstance(first, np.ndarray):
        require(isinstance(second, np.ndarray) and first.dtype == second.dtype
                and first.shape == second.shape
                and np.ascontiguousarray(first).tobytes() == np.ascontiguousarray(second).tobytes(),
                'Array bytes differ: ' + label)
    elif isinstance(first, dict):
        require(isinstance(second, dict) and list(first) == list(second), 'Dictionary keys/order differ: ' + label)
        for key in first:
            assert_tree_bitwise_equal(first[key], second[key], label + '/' + str(key))
    elif isinstance(first, (list, tuple)):
        require(type(first) is type(second) and len(first) == len(second), 'Sequence differs: ' + label)
        for index, (left, right) in enumerate(zip(first, second)):
            assert_tree_bitwise_equal(left, right, label + '/' + str(index))
    else:
        require(type(first) is type(second) and first == second, 'Value differs: ' + label)


def cpu_tree(value):
    """Clone nested optimizer/state trees without discarding their layout."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return type(value)((key, cpu_tree(item)) for key,item in value.items())
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def validate_phase_optimizer(model, saved, expected_groups, *, mean_steps, joint_steps=None):
    """Validate every named AdamW slot against its actual training phase."""
    require(isinstance(saved, dict), 'AdamW state must be a dictionary')
    assert_tree_equal(saved['param_groups'], expected_groups, 'optimizer_param_groups')
    by_name = optimizer_named_state(model, saved)
    parameters = dict(model.named_parameters())
    expected_steps = {}
    for name, slot in by_name.items():
        expected = joint_steps if joint_steps is not None and name.startswith(STOCHASTIC_PREFIXES) else mean_steps
        expected_steps[name] = expected
        require(slot is not None, 'Missing named AdamW history: ' + name)
        require(set(slot) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unexpected AdamW slot keys: ' + name)
        step = slot['step']
        require(isinstance(step, torch.Tensor) and step.dtype == torch.float32 and tuple(step.shape) == ()
                and bool(torch.isfinite(step)) and float(step) == expected,
                'Named AdamW phase step differs: ' + name)
        for key in ('exp_avg', 'exp_avg_sq'):
            moment, parameter = slot[key], parameters[name]
            require(isinstance(moment, torch.Tensor) and moment.shape == parameter.shape
                    and moment.dtype == parameter.dtype and bool(torch.isfinite(moment).all()),
                    'AdamW moment dtype/shape/finiteness differs: ' + name + '/' + key)
        require(bool((slot['exp_avg_sq'] >= 0).all()), 'Negative AdamW second moment: ' + name)
    require(len(saved['state']) == len(parameters), 'Incomplete named AdamW state')
    return by_name, expected_steps


def validate_module_state(module, saved, label):
    reference = module.state_dict()
    require(isinstance(saved, dict) and list(saved) == list(reference), label + ' key/order differs')
    for name, expected in reference.items():
        value = saved[name]
        require(isinstance(value, torch.Tensor) and value.shape == expected.shape and value.dtype == expected.dtype
                and bool(torch.isfinite(value).all()), label + ' dtype/shape/finiteness differs: ' + name)


class Response008Resume01Engine(Response008Fix01Engine):
    """Retain fix01 weights and named AdamW history without changing the step."""

    def __init__(self, data, statistics, config, noise_checkpoint, common, *, arm,
                 recovery_identity, device='cuda:0', engineering_fixture=False):
        require(set(recovery_identity) == IDENTITY_KEYS and recovery_identity['run_id'] == RUN_ID,
                'Resume01 identity keys/run differ')
        require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
                    for key, value in recovery_identity.items() if key != 'run_id'),
                'Resume01 identity SHA256 differs')
        fix01_identity = dict(run_id=FIX01_RUN_ID,
            config_sha256=recovery_identity['fix01_configuration_sha256'],
            source_manifest_sha256=recovery_identity['fix01_source_manifest_sha256'],
            base_configuration_sha256=recovery_identity['base_configuration_sha256'],
            base_source_manifest_sha256=recovery_identity['base_source_manifest_sha256'])
        if not engineering_fixture:
            require(fix01_identity == REGISTERED_FIX01_IDENTITY, 'Registered fix01 identity differs')
        metadata_only = bool(data.get('metadata_only_placeholder', False))
        if metadata_only:
            require(not engineering_fixture and torch.device(device).type == 'cpu'
                    and bool((np.asarray(data['source_ids']) < 0).all())
                    and bool((np.asarray(data['real_ids']) < 0).all())
                    and all(bool((np.asarray(data[key]) == 0).all()) for key in
                            ('source_carrier384', 'real', 'real_baseline100')),
                    'Metadata-only import requires explicit CPU zero placeholders')
        super().__init__(data, statistics, config, noise_checkpoint, common, arm=arm,
            recovery_identity=fix01_identity, device=device, engineering_fixture=engineering_fixture)
        self.fix01_identity = copy.deepcopy(fix01_identity)
        self.recovery_identity = copy.deepcopy(recovery_identity)
        self.metadata_only_placeholder = metadata_only
        self._resume01_import_attempted = False
        self._step_in_progress = False
        self._step_failed = False
        self.lineage = dict(origin='fresh_common_initialization', base_run_id=config['run_id'],
            parent_run_id=FIX01_RUN_ID, source_checkpoint_sha256=None,
            inherited_G_updates=0, inherited_D_updates=dict(self.imported_D_updates),
            model_implementation=MODEL_IMPLEMENTATION)

    def restore(self, path):
        raise ValueError('No generic/automatic resume01 restore; only the registered fix01 periodic6000 import')

    def import_legacy_full_stage3000(self, path, *, expected_sha256):
        raise ValueError('Resume01 rejects the old stage3000 import; only fix01 periodic6000 is registered')

    def enforce_frozen_references(self):
        self.extension.assert_frozen()
        require(state_digest(self.extension.model005.state_dict()) == self.noise_state_digest, 'Frozen005 changed')
        require(type(self.model).__module__ + '.' + type(self.model).__name__ == MODEL_IMPLEMENTATION,
                'Resume01 signal model implementation changed')
        require(parameter_signature(self.model) == self.parameter_signature, 'Resume01 parameter signature changed')

    def native_state_tree(self):
        return dict(model=self.model.state_dict(), discriminators=self.discriminators.state_dict(),
            g_optimizer=self.g_optimizer.state_dict(),
            d_optimizers={h:optimizer.state_dict() for h,optimizer in self.d_optimizers.items()},
            source_sampler=self.source_sampler.state_dict(), real_sampler=self.real_sampler.state_dict(),
            streams=self.streams.state_dict())

    def _validate_import_state(self, state, fix01_identity):
        require(isinstance(state, dict) and state.get('schema') == FIX01_SCHEMA
                and state.get('arm') == 'full' and state.get('config') == self.config,
                'Fix01 import schema/arm/config differs')
        require(state.get('engineering_fixture') is self.engineering_fixture,
                'Scientific/artificial checkpoint role differs')
        require(not self.engineering_fixture or state.get('artificial_parser_fixture') is True,
                'Engineering import requires a labelled artificial parser fixture')
        require(state.get('checkpoint_purpose') == 'periodic'
                and state.get('failure_may_be_mid_step') is False
                and state.get('automatic_resume_allowed') is False,
                'Only a complete periodic checkpoint is admissible; failure/mid-step purpose rejected')
        require(fix01_identity == self.fix01_identity and state.get('recovery') == self.fix01_identity,
                'Fix01 recovery identity differs')
        require(state.get('model_implementation') == MODEL_IMPLEMENTATION
                and state.get('parameter_signature') == self.parameter_signature, 'Fix01 model identity/signature differs')
        d6000 = {h:3000 for h in HYPOTHESES}
        require(type(state.get('G_updates')) is int and state['G_updates'] == 6000
                and state.get('D_updates') == d6000 and all(type(v) is int for v in state['D_updates'].values())
                and state.get('stage') == 'joint_stochastic', 'Only full G6000/D3000 complete-cycle counters are admissible')
        require(state.get('new_G_updates') == 3000 and state.get('new_D_updates') == d6000,
                'Fix01 executed-new counters differ')
        expected_lineage = dict(origin='stage3000_recovery', base_run_id=self.config['run_id'],
            source_checkpoint_sha256=LEGACY_STAGE3000_SHA256, inherited_G_updates=3000,
            inherited_D_updates={h:0 for h in HYPOTHESES})
        require(state.get('lineage') == expected_lineage, 'Fix01 parent lineage differs')
        require(state.get('noise_updates') == 0 and state.get('population_learning') is False,
                'Fix01 noise/population history differs')
        stat_hash = statistics_digest(self.statistics)
        require(state.get('statistics_sha256') == stat_hash
                and statistics_digest(state['statistics']) == stat_hash, 'Fix01 statistics differ')
        require(state.get('common_initial_state_sha256') == self.common_state_digest
                and state.get('initial_state_sha256') == self.initial_state_digest
                and state.get('initial_D_sha256') == self.initial_D_digest
                and state.get('noise005_state_sha256') == self.noise_state_digest, 'Fix01 frozen references differ')
        validate_module_state(self.model, state['model'], 'Fix01 G')
        validate_module_state(self.discriminators, state['discriminators'], 'Fix01 D')
        _, g_steps = validate_phase_optimizer(self.model, state['g_optimizer'],
            self.g_optimizer.state_dict()['param_groups'], mean_steps=6000, joint_steps=3000)
        require(set(state['d_optimizers']) == set(HYPOTHESES), 'Fix01 D optimizer hypotheses differ')
        for h,optimizer in self.d_optimizers.items():
            validate_phase_optimizer(self.discriminators[h], state['d_optimizers'][h],
                optimizer.state_dict()['param_groups'], mean_steps=3000)
        groups = self.config['distribution_microbatches']
        validate_sampler(state['source_sampler'], len(self.source), 6000*groups*self.config['source_parents_per_microbatch'])
        validate_sampler(state['real_sampler'], len(self.real), 6000*groups*self.config['real_rows_per_microbatch'])
        validate_streams(state['streams'])
        metrics = state.get('last_metrics', {})
        require(metrics.get('arm') == 'full' and metrics.get('stage') == 'joint_stochastic'
                and metrics.get('G_updates') == 6000 and metrics.get('D_updates') == d6000
                and metrics.get('recovery_run_id') == FIX01_RUN_ID
                and metrics.get('model_implementation') == MODEL_IMPLEMENTATION
                and metrics.get('inherited_G_updates') == 3000
                and metrics.get('new_G_updates') == 3000 and metrics.get('new_D_updates') == d6000,
                'Fix01 last completed metrics/counters differ')
        require(metrics.get('projection_context', {}).get('update') == 6000,
                'Fix01 completed projection update differs')
        return g_steps

    def import_fix01_full_periodic6000(self, path, *, expected_sha256, fix01_identity):
        require(self.arm == 'full' and self.update == 0 and self.resume_validation is None
                and not self._resume01_import_attempted and not self.g_optimizer.state_dict()['state']
                and self.d_updates == {h:0 for h in HYPOTHESES},
                'Fix01 periodic6000 import is permitted once into a fresh full engine')
        self._resume01_import_attempted = True
        require(isinstance(expected_sha256, str) and re.fullmatch('[0-9a-f]{64}', expected_sha256),
                'Registered checkpoint SHA256 format differs')
        require(self.engineering_fixture or expected_sha256 == REGISTERED_IMPORT_SHA256,
                'Unregistered scientific checkpoint hash; only the periodic6000 bytes are permitted')
        path = Path(path).resolve()
        require(sha256(path) == expected_sha256, 'Registered fix01 periodic6000 bytes changed')
        state = torch.load(path, map_location='cpu', weights_only=False)
        require(sha256(path) == expected_sha256, 'Fix01 periodic6000 checkpoint changed during read')
        g_steps = self._validate_import_state(state, fix01_identity)
        self.model.load_state_dict(state['model'], strict=True)
        self.discriminators.load_state_dict(state['discriminators'], strict=True)
        self.g_optimizer.load_state_dict(state['g_optimizer'])
        for h,optimizer in self.d_optimizers.items():
            optimizer.load_state_dict(state['d_optimizers'][h])
        self.source_sampler.load_state_dict(state['source_sampler'])
        self.real_sampler.load_state_dict(state['real_sampler'])
        self.streams.load_state_dict(state['streams'])
        self.update = 6000
        self.d_updates = {h:3000 for h in HYPOTHESES}
        self.last_metrics = copy.deepcopy(state['last_metrics'])
        self.set_stage('joint_stochastic')
        restored = self.native_state_tree()
        for key,value in restored.items():
            assert_tree_bitwise_equal(state[key], value, 'restored/' + key)
        self.enforce_frozen_references()
        self.imported_G_updates = 6000
        self.imported_D_updates = {h:3000 for h in HYPOTHESES}
        self.lineage = dict(origin='fix01_periodic6000_resume', base_run_id=self.config['run_id'],
            parent_run_id=FIX01_RUN_ID, source_checkpoint_sha256=expected_sha256,
            inherited_G_updates=6000, inherited_D_updates=dict(self.imported_D_updates),
            parent_lineage=copy.deepcopy(state['lineage']), model_implementation=MODEL_IMPLEMENTATION)
        self.resume_validation = dict(schema='n01/window008-resume01-import/v1', UTC=utc(), status='passed',
            source_checkpoint_sha256=expected_sha256, source_checkpoint_purpose=state['checkpoint_purpose'],
            source_schema=FIX01_SCHEMA, target_schema=SCHEMA, G_updates=6000, D_updates=dict(self.d_updates),
            new_G_updates=0, new_D_updates={h:0 for h in HYPOTHESES}, next_G_update=6001,
            first_update_requires_audit=True, first_update_requires_same_version_replay=True,
            optimizer_moments_reset=False, native_tree_bitwise_equal=True,
            state_digests={key:tree_digest(value) for key,value in restored.items()},
            G_optimizer_named_steps=g_steps,
            G_optimizer_named_slot_sha256={name:tree_digest(slot) for name,slot in
                                          optimizer_named_state(self.model, self.g_optimizer.state_dict()).items()},
            D_optimizer_named_slot_sha256={h:{name:tree_digest(slot) for name,slot in
                optimizer_named_state(self.discriminators[h], optimizer.state_dict()).items()}
                for h,optimizer in self.d_optimizers.items()},
            parameter_signature=self.parameter_signature, optimizer_sampler_streams_exact=True,
            projection_update=6001, recovery=copy.deepcopy(self.recovery_identity),
            fix01_recovery=copy.deepcopy(self.fix01_identity), engineering_fixture=self.engineering_fixture,
            metadata_only_placeholder=self.metadata_only_placeholder)
        return copy.deepcopy(self.resume_validation)

    @property
    def new_G_updates(self):
        if self.arm == 'full' and self.resume_validation is None:
            return 0
        return self.update-self.imported_G_updates

    @property
    def new_D_updates(self):
        if self.arm == 'full' and self.resume_validation is None:
            return {h:0 for h in HYPOTHESES}
        return {h:self.d_updates[h]-self.imported_D_updates[h] for h in HYPOTHESES}

    def step(self, *, audit=False, verify_replay=False):
        require(not self.metadata_only_placeholder, 'Metadata-only CPU checkpoint inspection cannot execute a model forward/update')
        require(not self._step_in_progress and not self._step_failed, 'Interrupted/failed resume01 engine cannot continue')
        require(self.arm != 'full' or self.resume_validation is not None, 'Full resume01 requires the registered fix01 periodic6000 import')
        if self.arm == 'full' and self.new_G_updates == 0:
            require(self.update == 6000 and self.stage == 'joint_stochastic' and audit and verify_replay,
                    'First resume01 full update must be joint6001 with audit and exact replay')
        self.enforce_frozen_references()
        self._step_in_progress = True
        try:
            # Skip the old fix01 first-step3001 wrapper; reuse its untouched base
            # scientific step/VJP and enrich only resume identity/accounting.
            result = Response008Engine.step(self, audit=audit, verify_replay=verify_replay)
            result.update(recovery_run_id=RUN_ID, model_implementation=MODEL_IMPLEMENTATION,
                inherited_G_updates=self.imported_G_updates, inherited_D_updates=dict(self.imported_D_updates),
                new_G_updates=self.new_G_updates, new_D_updates=self.new_D_updates)
            return result
        except BaseException:
            self._step_failed = True
            raise
        finally:
            self._step_in_progress = False

    def checkpoint(self, path, *, purpose='periodic', allow_overwrite=False):
        path = Path(path)
        require(not self.metadata_only_placeholder, 'Metadata-only CPU import cannot create a training checkpoint')
        require(purpose in ('initial', 'restored_start', 'periodic', 'stage_endpoint', 'final', 'failure'),
                'Unknown resume01 checkpoint purpose')
        require(purpose == 'failure' or (not self._step_in_progress and not self._step_failed),
                'Incomplete/failed update cannot be saved as a complete periodic checkpoint')
        if path.exists() and not allow_overwrite:
            raise FileExistsError('Preserve existing resume01 checkpoint: ' + str(path))
        require(not allow_overwrite or path.name == 'latest.pt', 'Only latest.pt may be replaced')
        if path.drive:
            require(path.resolve().drive.upper() in ('E:', 'D:'), 'Resume01 checkpoint files must reside on E or D')
        self.enforce_frozen_references()
        state = dict(schema=SCHEMA, arm=self.arm, config=self.config, model=cpu_state(self.model.state_dict()),
            discriminators=cpu_state(self.discriminators.state_dict()), g_optimizer=cpu_tree(self.g_optimizer.state_dict()),
            d_optimizers={h:cpu_tree(optimizer.state_dict()) for h,optimizer in self.d_optimizers.items()},
            source_sampler=self.source_sampler.state_dict(), real_sampler=self.real_sampler.state_dict(),
            streams=self.streams.state_dict(), G_updates=self.update, D_updates=dict(self.d_updates),
            stage=self.stage, statistics=self.statistics, statistics_sha256=statistics_digest(self.statistics),
            last_metrics=copy.deepcopy(self.last_metrics), initial_state_sha256=self.initial_state_digest,
            common_initial_state_sha256=self.common_state_digest, initial_D_sha256=self.initial_D_digest,
            noise005_state_sha256=self.noise_state_digest, saved_UTC=utc(), population_learning=False, noise_updates=0,
            recovery=copy.deepcopy(self.recovery_identity), model_implementation=MODEL_IMPLEMENTATION,
            parameter_signature=self.parameter_signature, lineage=copy.deepcopy(self.lineage),
            new_G_updates=self.new_G_updates, new_D_updates=self.new_D_updates,
            engineering_fixture=self.engineering_fixture, checkpoint_purpose=purpose,
            automatic_resume_allowed=False, failure_may_be_mid_step=purpose == 'failure',
            complete_step_boundary=purpose != 'failure', parent_fix01_recovery=copy.deepcopy(self.fix01_identity))
        atomic_torch_save(state, path)
