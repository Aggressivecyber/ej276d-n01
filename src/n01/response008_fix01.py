"""Versioned Window008 fix01 engine; legacy step/VJP, strict mean-stage import.

The sole admitted scientific import is the hash-bound original full stage3000.
Fix01 checkpoints are forensic/fixed endpoints, never automatic resume inputs.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import FullCoverageSampler, sha256
from .model008_fix01 import N01Response008Fix01
from .response008 import (HYPOTHESES, PairedStreams, Response008Engine,
    cpu_state, state_digest, statistics_digest)
from .train import atomic_torch_save, seed_all, utc

RUN_ID = 'window008_fix01_20261013'
SCHEMA = 'response008_fix01/v1'
MODEL_IMPLEMENTATION = 'n01.model008_fix01.N01Response008Fix01'
IDENTITY_KEYS = {'run_id', 'config_sha256', 'source_manifest_sha256',
                 'base_configuration_sha256', 'base_source_manifest_sha256'}
STOCHASTIC_PREFIXES = ('physical_head.', 'basis_head.', 'envelope_head.', 'local_filter.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_equal(first, second):
    return (isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
            and first.dtype == second.dtype and first.shape == second.shape
            and torch.equal(first.detach().cpu(), second.detach().cpu()))


def assert_tree_equal(first, second, label='state'):
    if isinstance(first, torch.Tensor):
        require(tensor_equal(first, second), 'Tensor differs: ' + label)
    elif isinstance(first, np.ndarray):
        require(isinstance(second, np.ndarray) and first.dtype == second.dtype
                and first.shape == second.shape and np.array_equal(first, second), 'Array differs: ' + label)
    elif isinstance(first, dict):
        require(isinstance(second, dict) and set(first) == set(second), 'Dictionary differs: ' + label)
        for key in first:
            assert_tree_equal(first[key], second[key], label + '/' + str(key))
    elif isinstance(first, (list, tuple)):
        require(type(first) is type(second) and len(first) == len(second), 'Sequence differs: ' + label)
        for index, (left, right) in enumerate(zip(first, second)):
            assert_tree_equal(left, right, label + '/' + str(index))
    else:
        require(type(first) is type(second) and first == second, 'Value differs: ' + label)


def parameter_signature(model):
    return [dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
            for name, value in model.named_parameters()]


def state_signature(model):
    return [dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
            for name, value in model.state_dict().items()]


def tree_digest(value):
    """Hash types/keys/tensor bytes, independent of torch container metadata."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b'tensor'); visit(str(item.dtype)); visit(list(item.shape)); digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b'array'); visit(str(item.dtype)); visit(list(item.shape)); digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode()); visit(len(item))
            for part in item:
                visit(part)
        else:
            digest.update(type(item).__name__.encode())
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
        digest.update(b'\0')
    visit(value)
    return digest.hexdigest()


def validate_sampler(state, expected_n, expected_draws):
    require(set(state) == {'n', 'rng', 'order', 'position', 'completed_epochs', 'visits'}, 'Sampler keys differ')
    require(type(state['n']) is int and state['n'] == expected_n, 'Sampler population differs')
    order, visits = state['order'], state['visits']
    require(isinstance(order, torch.Tensor) and order.dtype in (torch.int32, torch.int64)
            and tuple(order.shape) == (expected_n,), 'Sampler order dtype/shape differs')
    require(isinstance(visits, torch.Tensor) and visits.dtype == torch.int64 and tuple(visits.shape) == (expected_n,), 'Sampler visits dtype/shape differs')
    order, visits = order.cpu().numpy(), visits.cpu().numpy()
    require(np.array_equal(np.sort(order), np.arange(expected_n)), 'Sampler order is not a permutation')
    epochs, position = divmod(expected_draws, expected_n)
    require(type(state['completed_epochs']) is int and state['completed_epochs'] == epochs
            and type(state['position']) is int and state['position'] == position, 'Sampler draw position differs')
    expected_visits = np.full(expected_n, epochs, dtype=np.int64)
    expected_visits[order[:position]] += 1
    require(np.array_equal(visits, expected_visits), 'Sampler visit counts differ')
    require(state['rng'].get('bit_generator') == 'PCG64', 'Sampler RNG implementation differs')
    probe = FullCoverageSampler(expected_n, 0)
    probe.load_state_dict(copy.deepcopy(state))
    assert_tree_equal(probe.state_dict(), state, 'sampler_roundtrip')


def validate_streams(state):
    expected = PairedStreams(0, 1, 2)
    require(set(state) == set(expected.generators), 'Paired stream keys differ')
    for name, reference in expected.state_dict().items():
        actual = state[name]
        require(isinstance(actual, torch.Tensor) and actual.dtype == reference.dtype
                and actual.shape == reference.shape, 'Paired RNG dtype/shape differs: ' + name)
    expected.load_state_dict(state)
    assert_tree_equal(expected.state_dict(), state, 'stream_roundtrip')


def optimizer_named_state(model, optimizer_state):
    """Reject positional ambiguity before mapping AdamW slots to parameter names."""
    require(set(optimizer_state) == {'state', 'param_groups'} and len(optimizer_state['param_groups']) == 1,
            'Expected one AdamW parameter group')
    parameters = list(model.named_parameters())
    identifiers = optimizer_state['param_groups'][0]['params']
    require(len(identifiers) == len(parameters) and len(set(identifiers)) == len(identifiers)
            and identifiers == list(range(len(parameters))), 'Optimizer parameter IDs/order differ')
    require(set(optimizer_state['state']) <= set(identifiers), 'Unknown optimizer parameter slot')
    return {name: optimizer_state['state'].get(identifier)
            for identifier, (name, _) in zip(identifiers, parameters)}


def validate_optimizer(model, saved, expected_groups, *, update, stochastic_empty=False):
    assert_tree_equal(saved['param_groups'], expected_groups, 'optimizer_param_groups')
    by_name = optimizer_named_state(model, saved)
    parameters = dict(model.named_parameters())
    for name, slot in by_name.items():
        if stochastic_empty and name.startswith(STOCHASTIC_PREFIXES):
            require(slot is None, 'Inactive stochastic parameter already has optimizer history: ' + name)
        elif stochastic_empty and update:
            require(slot is not None, 'Trained mean parameter is missing its optimizer history: ' + name)
        if slot is None:
            continue
        require(set(slot) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unexpected AdamW slot keys: ' + name)
        step = slot['step']
        require(isinstance(step, torch.Tensor) and step.numel() == 1 and float(step) == update,
                'AdamW step differs: ' + name)
        parameter = parameters[name]
        for key in ('exp_avg', 'exp_avg_sq'):
            value = slot[key]
            require(isinstance(value, torch.Tensor) and value.shape == parameter.shape
                    and value.dtype == parameter.dtype and bool(torch.isfinite(value).all()),
                    'AdamW moment dtype/shape/finiteness differs: ' + name + '/' + key)
    if update:
        require(any(value is not None for value in by_name.values()), 'Missing trained optimizer slots')
    else:
        require(not saved['state'], 'Untrained optimizer contains slots')
    return by_name


class Response008Fix01Engine(Response008Engine):
    """Reuse the frozen step and VJP with an explicitly versioned signal model."""

    def __init__(self, data, statistics, config, noise_checkpoint, common, *, arm,
                 recovery_identity, device='cuda:0', engineering_fixture=False):
        require(arm in ('full', 'simple'), 'Fix01 never retrains retained deterministic arm')
        require(set(recovery_identity) == IDENTITY_KEYS and recovery_identity['run_id'] == RUN_ID,
                'Fix01 identity keys/run differ')
        require(all(isinstance(value, str) and len(value) == 64
                    for key, value in recovery_identity.items() if key != 'run_id'), 'Fix01 identity hashes differ')
        if engineering_fixture:
            require(config.get('engineering_fixture') is True and statistics.get('engineering_fixture') is True
                    and bool((np.asarray(data['source_ids']) < 0).all())
                    and bool((np.asarray(data['real_ids']) < 0).all()), 'Engineering mode requires explicitly artificial arrays')
        else:
            require(config.get('engineering_fixture') is not True and statistics.get('engineering_fixture') is not True,
                    'Artificial inputs cannot enter a scientific engine')
        self.engineering_fixture = bool(engineering_fixture)
        super().__init__(data, statistics, config, noise_checkpoint, common, arm=arm, device=device)
        old = self.model
        original_optimizer = copy.deepcopy(self.g_optimizer.state_dict())
        signature = parameter_signature(old)
        seed_all(config['seeds']['initialization'])
        revised = N01Response008Fix01(statistics['model'], arm).to(self.device)
        require(signature == parameter_signature(revised) and state_signature(old) == state_signature(revised),
                'Fix01 changed parameter/buffer names, order, shapes or dtypes')
        revised.load_state_dict(old.state_dict(), strict=True)
        assert_tree_equal(old.state_dict(), revised.state_dict(), 'model_replacement')
        self.model = revised
        self.g_optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['loss']['G_lr'],
                                             weight_decay=config['loss']['G_weight_decay'])
        assert_tree_equal(self.g_optimizer.state_dict(), original_optimizer, 'new_G_optimizer_layout')
        self.g_optimizer.load_state_dict(original_optimizer)
        assert_tree_equal(self.g_optimizer.state_dict(), original_optimizer, 'new_G_optimizer_mapping')
        require(state_digest(self.model.state_dict()) == self.initial_state_digest, 'Replacement changed common initializer')
        self.recovery_identity = copy.deepcopy(recovery_identity)
        self.parameter_signature = signature
        self.imported_G_updates = 0
        self.imported_D_updates = {h: 0 for h in HYPOTHESES}
        self.lineage = dict(origin='fresh_common_initialization', base_run_id=config['run_id'],
                            source_checkpoint_sha256=None, inherited_G_updates=0,
                            inherited_D_updates=dict(self.imported_D_updates))
        self.model_replacement_receipt = dict(parameter_signature=signature,
            state_dict_signature=state_signature(self.model),
            original_and_fix01_initial_state_equal=True, optimizer_named_mapping_exact=True,
            initial_state_sha256=self.initial_state_digest, initial_D_sha256=self.initial_D_digest)
        self.resume_validation = None
        self._legacy_import_attempted = False

    def restore(self, path):
        raise ValueError('No automatic fix01 resume; only import_legacy_full_stage3000 is registered')

    def import_legacy_full_stage3000(self, path, *, expected_sha256):
        require(self.arm == 'full' and self.update == 0 and self.resume_validation is None,
                'Legacy import is permitted once into a fresh full engine')
        self._legacy_import_attempted = True
        path = Path(path).resolve()
        require(sha256(path) == expected_sha256, 'Registered full stage3000 bytes changed')
        state = torch.load(path, map_location='cpu', weights_only=False)
        require(sha256(path) == expected_sha256, 'Legacy checkpoint changed during read')
        require(state.get('schema') == 'response008/v1' and state.get('arm') == 'full'
                and state.get('config') == self.config, 'Legacy schema/arm/config differs')
        require(state.get('G_updates') == 3000 and state.get('D_updates') == {h: 0 for h in HYPOTHESES}
                and state.get('stage') == 'mean_response', 'Only the zero-D mean stage3000 is admissible')
        require(state.get('noise_updates') == 0 and state.get('population_learning') is False,
                'Legacy noise/population history differs')
        stat_hash = statistics_digest(self.statistics)
        require(state.get('statistics_sha256') == stat_hash and statistics_digest(state['statistics']) == stat_hash,
                'Legacy statistics differ')
        require(state.get('common_initial_state_sha256') == self.common_state_digest
                and state.get('initial_state_sha256') == self.initial_state_digest
                and state.get('initial_D_sha256') == self.initial_D_digest
                and state.get('noise005_state_sha256') == self.noise_state_digest, 'Legacy frozen references differ')
        require(list(state['model']) == list(self.model.state_dict()), 'Legacy model key order differs')
        for name, reference in self.model.state_dict().items():
            value = state['model'][name]
            require(isinstance(value, torch.Tensor) and value.shape == reference.shape and value.dtype == reference.dtype
                    and bool(torch.isfinite(value).all()), 'Legacy model dtype/shape/finiteness differs: ' + name)
        validate_optimizer(self.model, state['g_optimizer'], self.g_optimizer.state_dict()['param_groups'],
                           update=3000, stochastic_empty=True)
        assert_tree_equal(state['discriminators'], cpu_state(self.discriminators.state_dict()), 'untrained_D_state')
        require(set(state['d_optimizers']) == set(HYPOTHESES), 'Legacy D optimizer keys differ')
        for h, optimizer in self.d_optimizers.items():
            validate_optimizer(self.discriminators[h], state['d_optimizers'][h], optimizer.state_dict()['param_groups'], update=0)
        groups = self.config['distribution_microbatches']
        validate_sampler(state['source_sampler'], len(self.source), 3000 * groups * self.config['source_parents_per_microbatch'])
        validate_sampler(state['real_sampler'], len(self.real), 3000 * groups * self.config['real_rows_per_microbatch'])
        validate_streams(state['streams'])
        require(state['last_metrics'].get('G_updates') == 3000 and state['last_metrics'].get('stage') == 'mean_response'
                and state['last_metrics'].get('D_updates') == {h: 0 for h in HYPOTHESES}, 'Legacy last metrics differ')
        self.model.load_state_dict(state['model'], strict=True)
        self.discriminators.load_state_dict(state['discriminators'], strict=True)
        self.g_optimizer.load_state_dict(state['g_optimizer'])
        for h, optimizer in self.d_optimizers.items():
            optimizer.load_state_dict(state['d_optimizers'][h])
        self.source_sampler.load_state_dict(state['source_sampler'])
        self.real_sampler.load_state_dict(state['real_sampler'])
        self.streams.load_state_dict(state['streams'])
        self.update = 3000; self.d_updates = {h: 0 for h in HYPOTHESES}
        self.last_metrics = copy.deepcopy(state['last_metrics'])
        self.set_stage('mean_response')
        restored = dict(model=self.model.state_dict(), discriminators=self.discriminators.state_dict(),
            g_optimizer=self.g_optimizer.state_dict(), d_optimizers={h:o.state_dict() for h,o in self.d_optimizers.items()},
            source_sampler=self.source_sampler.state_dict(), real_sampler=self.real_sampler.state_dict(), streams=self.streams.state_dict())
        for key, value in restored.items():
            assert_tree_equal(state[key], value, 'restored/' + key)
        self.imported_G_updates = 3000
        self.lineage = dict(origin='stage3000_recovery', base_run_id=self.config['run_id'],
            source_checkpoint_sha256=expected_sha256, inherited_G_updates=3000,
            inherited_D_updates=dict(self.imported_D_updates))
        self.resume_validation = dict(schema='n01/window008-fix01-resume/v1', UTC=utc(), status='passed',
            source_checkpoint_sha256=expected_sha256, G_updates=3000, D_updates=dict(self.d_updates),
            next_G_update=3001, first_update_requires_audit=True, first_update_requires_same_version_replay=True,
            state_digests={key:tree_digest(value) for key,value in restored.items()},
            parameter_signature=self.parameter_signature, optimizer_sampler_streams_exact=True,
            projection_update=3001, recovery=copy.deepcopy(self.recovery_identity))
        return copy.deepcopy(self.resume_validation)

    @property
    def new_G_updates(self):
        # A failed import has executed no new optimizer step, even if a later
        # post-load assertion interrupted installation of its inherited state.
        if self.arm == 'full' and self.resume_validation is None and (not self.engineering_fixture or self._legacy_import_attempted):
            return 0
        return self.update - self.imported_G_updates

    @property
    def new_D_updates(self):
        if self.arm == 'full' and self.resume_validation is None and (not self.engineering_fixture or self._legacy_import_attempted):
            return {h:0 for h in HYPOTHESES}
        return {h:self.d_updates[h] - self.imported_D_updates[h] for h in HYPOTHESES}

    def step(self, *, audit=False, verify_replay=False):
        require(self.arm != 'full' or self.resume_validation is not None or self.engineering_fixture,
                'Full fix01 requires the registered legacy import')
        if self.arm == 'full' and self.new_G_updates == 0 and (self.resume_validation is not None or not self.engineering_fixture):
            require(self.update == 3000 and self.stage == 'joint_stochastic' and audit and verify_replay,
                    'First fix01 full update must be joint3001 with audit and exact replay')
        result = super().step(audit=audit, verify_replay=verify_replay)
        result.update(recovery_run_id=RUN_ID, model_implementation=MODEL_IMPLEMENTATION,
                      inherited_G_updates=self.imported_G_updates, new_G_updates=self.new_G_updates,
                      new_D_updates=self.new_D_updates)
        return result

    def checkpoint(self, path, *, purpose='periodic', allow_overwrite=False):
        path = Path(path)
        require(purpose in ('initial', 'restored_start', 'periodic', 'stage_endpoint', 'final', 'failure'), 'Unknown fix01 checkpoint purpose')
        if path.exists() and not allow_overwrite:
            raise FileExistsError('Preserve existing fix01 checkpoint: ' + str(path))
        require(not allow_overwrite or path.name == 'latest.pt', 'Only latest.pt may be replaced')
        self.extension.assert_frozen()
        require(state_digest(self.extension.model005.state_dict()) == self.noise_state_digest, 'Frozen005 changed')
        state = dict(schema=SCHEMA, arm=self.arm, config=self.config, model=cpu_state(self.model.state_dict()),
            discriminators=cpu_state(self.discriminators.state_dict()), g_optimizer=self.g_optimizer.state_dict(),
            d_optimizers={h:o.state_dict() for h,o in self.d_optimizers.items()}, source_sampler=self.source_sampler.state_dict(),
            real_sampler=self.real_sampler.state_dict(), streams=self.streams.state_dict(), G_updates=self.update,
            D_updates=dict(self.d_updates), stage=self.stage, statistics=self.statistics,
            statistics_sha256=statistics_digest(self.statistics), last_metrics=self.last_metrics,
            initial_state_sha256=self.initial_state_digest, common_initial_state_sha256=self.common_state_digest,
            initial_D_sha256=self.initial_D_digest, noise005_state_sha256=self.noise_state_digest,
            saved_UTC=utc(), population_learning=False, noise_updates=0,
            recovery=copy.deepcopy(self.recovery_identity), model_implementation=MODEL_IMPLEMENTATION,
            parameter_signature=self.parameter_signature, lineage=copy.deepcopy(self.lineage),
            new_G_updates=self.new_G_updates, new_D_updates=self.new_D_updates,
            engineering_fixture=self.engineering_fixture,
            checkpoint_purpose=purpose, automatic_resume_allowed=False,
            failure_may_be_mid_step=purpose == 'failure')
        atomic_torch_save(state, path)
