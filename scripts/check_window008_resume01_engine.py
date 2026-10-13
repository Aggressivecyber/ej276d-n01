"""CPU-only resume01 QA: artificial updates and zero-forward native import.

The artificial cycle6000 parser fixture has assigned counters/moment steps;
it is not evidence that 6000 engineering/scientific updates were executed.
Exactly four artificial G and six artificial D updates are executed here.
The genuine periodic checkpoint is imported using explicitly zero CPU arrays,
with forwards, optimizer steps and science sampler/stream draws forbidden.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if os.name == 'nt':
        os.environ['TEMP'] = 'E:/GAN/runtime_tmp'
        os.environ['TMP'] = 'E:/GAN/runtime_tmp'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import torch

from check_window008_engine import fixture
from n01.data import FullCoverageSampler, sha256, write_json
from n01.losses008 import WaveformDiscriminator
from n01.model008_fix01 import N01Response008Fix01
from n01.response008 import HYPOTHESES, PairedStreams, common_initial_state, state_digest
from n01.response008_fix01 import Response008Fix01Engine, STOCHASTIC_PREFIXES, optimizer_named_state
from n01.response008_resume01 import (LEGACY_STAGE3000_SHA256, MODEL_IMPLEMENTATION,
    REGISTERED_FIX01_IDENTITY, REGISTERED_IMPORT_SHA256, RUN_ID, SCHEMA,
    Response008Resume01Engine, assert_tree_bitwise_equal, cpu_tree, tree_digest,
    validate_phase_optimizer)
from n01.train import atomic_torch_save, utc


def engineering_engine_from_bundle(bundle, *, arm, device):
    """Construct native artificial QA from its frozen initialization tensors.

    CPU initialization arithmetic can differ by one ulp across Windows/Linux.
    The explicitly artificial bundle supplies its original D initializer, just
    as it supplies common G initialization. Scientific engines never use this.
    """
    if bundle.get('schema') != 'n01/window008-resume01-native-fixture/v1' or bundle.get('artificial_data_only') is not True:
        raise ValueError('Native QA requires the explicit artificial fixture bundle')
    result = Response008Resume01Engine(bundle['data'],bundle['statistics'],bundle['config'],
        bundle['noise_checkpoint'],bundle['common_initial_state'],arm=arm,
        recovery_identity=bundle['recovery_identity'],device=device,engineering_fixture=True)
    if result.engineering_fixture is not True or result.update != 0 or result.g_optimizer.state_dict()['state']:
        raise AssertionError('Artificial initializer installation requires an unused engineering engine')
    result.discriminators.load_state_dict(bundle['initial_discriminators'],strict=True)
    actual=[state_digest(value.state_dict()) for value in result.discriminators.values()]
    if len(set(actual)) != 1 or actual[0] != bundle['initial_D_sha256']:
        raise AssertionError('Frozen artificial D initializer bytes changed')
    result.initial_D_digest=actual[0]
    return result


def adapt_engineering_fixture_to_formal_batch(bundle, parser_state):
    """Expand artificial rows/batch shapes, retaining fitted stats and weights.

    All cycle6000 counters are fabricated parser metadata. This performs no
    model forward, optimizer step, experimental decode or statistic fit.
    """
    if bundle.get('artificial_data_only') is not True or parser_state.get('artificial_parser_fixture') is not True:
        raise ValueError('Formal-batch adaptation is restricted to labelled artificial fixtures')
    revised=copy.deepcopy(bundle);state=copy.deepcopy(parser_state)
    revised['config'].update(distribution_microbatches=2,source_parents_per_microbatch=256,
        responses_per_parent=2,real_rows_per_microbatch=512,engineering_fixture=True)
    for key in ('source_carrier384','real','real_baseline100'):
        values=np.asarray(bundle['data'][key]);count=len(values)
        if 1024 % count:
            raise ValueError('Artificial fixture rows must divide1024 exactly')
        revised['data'][key]=np.tile(values,(1024//count,1))
    revised['data']['source_ids']=-100000-np.arange(1024,dtype=np.int64)
    revised['data']['real_ids']=-200000-np.arange(1024,dtype=np.int64)
    state['config']=copy.deepcopy(revised['config'])
    for key,rows,seed in (
        ('source_sampler',256,revised['config']['seeds']['source_sampler']),
        ('real_sampler',512,revised['config']['seeds']['real_sampler'])):
        sampler=FullCoverageSampler(1024,seed)
        sampler.draw(6000*2*rows)
        state[key]=sampler.state_dict()
    revised.update(config_only_fixture_batch_shape_changed=True,statistics_refit=False,
        actual_optimizer_updates_for_adaptation=0,artificial_SOURCE_rows=1024,artificial_REAL_rows=1024,
        assigned_source_sampler_draws=3072000,assigned_real_sampler_draws=6144000)
    state['formal_batch_artificial_adaptation']=True
    return revised,state


def run(output, report):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    base = json.loads((ROOT/'window008_config.json').read_text(encoding='utf8'))
    resume_cfg = ROOT/'window008_resume01_config.json'
    fixed_identity = copy.deepcopy(REGISTERED_FIX01_IDENTITY)
    identity = dict(run_id=RUN_ID, config_sha256=sha256(resume_cfg), source_manifest_sha256='c'*64,
        base_configuration_sha256=fixed_identity['base_configuration_sha256'],
        base_source_manifest_sha256=fixed_identity['base_source_manifest_sha256'],
        fix01_configuration_sha256=fixed_identity['config_sha256'],
        fix01_source_manifest_sha256=fixed_identity['source_manifest_sha256'])
    protected = {}
    for manifest in ('WINDOW008_SOURCE_MANIFEST.json', 'WINDOW008_FIX01_SOURCE_MANIFEST.json'):
        document = json.loads((ROOT/manifest).read_text(encoding='utf8'))
        for relative,expected in document['files'].items():
            actual = sha256(ROOT/relative)
            if actual != expected:
                raise AssertionError('Protected source bytes changed before QA: '+relative)
            protected[relative] = actual
    own_paths = (Path(__file__), ROOT/'src/n01/response008_resume01.py')
    own_hashes = {path.relative_to(ROOT).as_posix():sha256(path) for path in own_paths}
    noise_path = ROOT/base['noise_reference']['path']
    if sha256(noise_path) != base['noise_reference']['sha256']:
        raise AssertionError('Frozen005 file bytes changed')
    noise = torch.load(noise_path, map_location='cpu', weights_only=False)
    report.update(protected_source_files=len(protected), protected_source_sha256=protected,
        own_source_sha256=own_hashes, recovery_identity=identity,
        source_manifest_identity_is_CPU_metadata_dummy=True, engineering_G_updates=0,
        engineering_D_updates=0, actual_signal_checkpoints_read=0,
        scientific_sampler_or_paired_stream_draws=0)
    checks = report['checks']
    def passed(name, details=None):
        checks.append(dict(name=name, passed=True, details=details))
    def rejected(name, call, expected):
        try:
            call()
        except (ValueError, RuntimeError, FileExistsError, KeyError, TypeError, AssertionError) as error:
            if expected not in str(error):
                raise AssertionError('Wrong rejection for '+name+': '+str(error)) from error
            passed(name, dict(error_type=type(error).__name__, error=str(error)))
        else:
            raise AssertionError('Expected rejection: '+name)
    passed('original_68_and_fix01_114_frozen_source_hashes_match', dict(unique_files=len(protected)))
    config = copy.deepcopy(base)
    config.update(distribution_microbatches=2, source_parents_per_microbatch=4,
                  responses_per_parent=2, real_rows_per_microbatch=8, engineering_fixture=True)
    data,statistics,fixture_sha = fixture(rows=128)
    common = common_initial_state(statistics['model'], base['seeds']['initialization'])
    report.update(artificial_fixture_sha256=fixture_sha,
        artificial_cycle6000_counters_and_steps_assigned=True,
        artificial_cycle6000_not_6000_actual_updates=True,
        artificial_common_state_sha256=state_digest(common))
    def engine(arm='full'):
        return Response008Resume01Engine(data,statistics,config,noise,common,arm=arm,
            recovery_identity=identity,device='cpu',engineering_fixture=True)
    def do_step(target, stage):
        target.set_stage(stage)
        value = target.step(audit=True,verify_replay=True)
        report['engineering_G_updates'] += 1
        report['engineering_D_updates'] += 0 if stage == 'mean_response' else 3
        return value

    # One artificial full mean update supplies the actual mean-branch moments.
    preparer = Response008Fix01Engine(data,statistics,config,noise,common,arm='full',
        recovery_identity=fixed_identity,device='cpu',engineering_fixture=True)
    do_step(preparer, 'mean_response')
    one_mean = output/'one_artificial_full_mean.pt'
    preparer.checkpoint(one_mean,purpose='periodic')
    staged = torch.load(one_mean,map_location='cpu',weights_only=False)
    for index,(name,parameter) in enumerate(preparer.model.named_parameters()):
        slot = staged['g_optimizer']['state'].get(index)
        if slot is None:
            slot = dict(step=torch.tensor(3000.,dtype=torch.float32),
                exp_avg=torch.full_like(parameter, (index+1)*1.e-10),
                exp_avg_sq=torch.full_like(parameter, 1.e-7))
            staged['g_optimizer']['state'][index] = cpu_tree(slot)
        else:
            slot['step'] = torch.tensor(6000.,dtype=torch.float32)
        if float(slot['step']) != (3000 if name.startswith(STOCHASTIC_PREFIXES) else 6000):
            raise AssertionError('Artificial named optimizer fixture phase mapping differs')
    # Reinsert parameter-ID order after filling formerly inactive slots.
    staged['g_optimizer']['state'] = dict(sorted(staged['g_optimizer']['state'].items()))
    for h,discriminator in preparer.discriminators.items():
        staged['d_optimizers'][h]['state'] = {
            index:dict(step=torch.tensor(3000.,dtype=torch.float32),
                exp_avg=torch.full_like(parameter, (index+1)*1.e-10),
                exp_avg_sq=torch.full_like(parameter, 1.e-7))
            for index,(_,parameter) in enumerate(discriminator.named_parameters())}
    d6000 = {h:3000 for h in HYPOTHESES}
    staged.update(G_updates=6000,D_updates=d6000,stage='joint_stochastic',new_G_updates=3000,
        new_D_updates=dict(d6000),artificial_parser_fixture=True,
        lineage=dict(origin='stage3000_recovery',base_run_id=base['run_id'],
            source_checkpoint_sha256=LEGACY_STAGE3000_SHA256,inherited_G_updates=3000,
            inherited_D_updates={h:0 for h in HYPOTHESES}))
    staged['last_metrics'].update(arm='full',stage='joint_stochastic',G_updates=6000,D_updates=dict(d6000),
        new_G_updates=3000,new_D_updates=dict(d6000),inherited_G_updates=3000,
        recovery_run_id=fixed_identity['run_id'],model_implementation=MODEL_IMPLEMENTATION,
        projection_context=preparer.loss.create_projection_context(6000).metadata())
    for key,n,rows,seed in (
        ('source_sampler',len(data['source_ids']),config['source_parents_per_microbatch'],base['seeds']['source_sampler']),
        ('real_sampler',len(data['real_ids']),config['real_rows_per_microbatch'],base['seeds']['real_sampler'])):
        sampler = FullCoverageSampler(n,seed)
        sampler.draw(6000*config['distribution_microbatches']*rows)
        staged[key] = sampler.state_dict()
    # The explicit latents are artificial fixture state, not a claim of6000 draws.
    streams = PairedStreams(base['seeds']['train_response'],base['seeds']['train_noise005'],base['seeds']['train_extension'])
    streams.draw(19,'cpu')
    staged['streams'] = streams.state_dict()
    fixture_path = output/'fabricated_fix01_periodic6000.engineering.pt'
    atomic_torch_save(staged,fixture_path)
    fixture_hash = sha256(fixture_path)
    atomic_torch_save(dict(schema='n01/window008-resume01-native-fixture/v1',
        artificial_data_only=True,data=data,statistics=statistics,config=config,
        noise_checkpoint=noise,common_initial_state=common,
        initial_discriminators=staged['discriminators'],initial_D_sha256=staged['initial_D_sha256'],
        recovery_identity=identity,fix01_identity=fixed_identity,
        parser_checkpoint_name=fixture_path.name,parser_checkpoint_sha256=fixture_hash,
        assigned_parser_G_updates=6000,assigned_parser_D_updates_per_hypothesis=3000,
        actual_preparer_G_updates=1,actual_preparer_D_updates=0), output/'ENGINEERING_FIXTURE_INPUTS.pt')
    resumed = engine()
    receipt = resumed.import_fix01_full_periodic6000(fixture_path,
        expected_sha256=fixture_hash,fix01_identity=fixed_identity)
    for key,value in resumed.native_state_tree().items():
        assert_tree_bitwise_equal(staged[key],value,'artificial_restored/'+key)
    if resumed.new_G_updates != 0 or any(resumed.new_D_updates.values()):
        raise AssertionError('Inherited artificial counters were counted as new updates')
    passed('artificial_all_models_named_moments_samplers_streams_import_bitwise',receipt['state_digests'])
    passed('artificial_import_retains_mixed_6000_and_3000_G_steps',
        {str(step):sum(value==step for value in receipt['G_optimizer_named_steps'].values()) for step in (6000,3000)})
    rejected('full_requires_explicit_import',lambda:engine().step(audit=True,verify_replay=True),'requires the registered')
    rejected('generic_restore_rejected',lambda:resumed.restore(fixture_path),'No generic/automatic')
    rejected('legacy3000_entry_rejected',lambda:resumed.import_legacy_full_stage3000(fixture_path,expected_sha256=fixture_hash),'rejects the old stage3000')
    rejected('repeat_import_rejected',lambda:resumed.import_fix01_full_periodic6000(fixture_path,
        expected_sha256=fixture_hash,fix01_identity=fixed_identity),'permitted once')
    rejected('simple_cannot_import_full',lambda:engine('simple').import_fix01_full_periodic6000(fixture_path,
        expected_sha256=fixture_hash,fix01_identity=fixed_identity),'fresh full')
    rejected('first6001_requires_audit',lambda:resumed.step(audit=False,verify_replay=True),'joint6001')
    rejected('first6001_requires_replay',lambda:resumed.step(audit=True,verify_replay=False),'joint6001')
    resumed.set_stage('mean_response')
    rejected('first6001_requires_joint_stage',lambda:resumed.step(audit=True,verify_replay=True),'joint6001')
    resumed.set_stage('joint_stochastic')
    # Reconstruct next draws using independent clones, without advancing resumed.
    sampler_s = FullCoverageSampler(len(data['source_ids']),0)
    sampler_r = FullCoverageSampler(len(data['real_ids']),0)
    sampler_s.load_state_dict(staged['source_sampler']);sampler_r.load_state_dict(staged['real_sampler'])
    clone_streams = PairedStreams(0,1,2);clone_streams.load_state_dict(staged['streams'])
    expected_batches = []
    for _ in range(config['distribution_microbatches']):
        si = np.repeat(sampler_s.draw(config['source_parents_per_microbatch']),config['responses_per_parent'])
        ri = sampler_r.draw(config['real_rows_per_microbatch'])
        response,nz,epsilon = clone_streams.draw(len(si),'cpu')
        expected_batches.append(dict(source_rows=si,real_rows=ri,source_parent_ids=data['source_ids'][si],
            real_original_ids=data['real_ids'][ri],response=response,noise005=nz,extension=epsilon))
    expected_projection = resumed.loss.create_projection_context(6001).state_dict()
    full_result = do_step(resumed,'joint_stochastic')
    assert_tree_bitwise_equal(expected_batches,resumed.last_context['batches'],'next6001_draws')
    assert_tree_bitwise_equal(expected_projection,resumed.last_context['projection_context'],'next6001_projection')
    if (full_result['G_updates'] != 6001 or full_result['new_G_updates'] != 1
            or full_result['D_updates'] != {h:3001 for h in HYPOTHESES}
            or full_result['new_D_updates'] != {h:1 for h in HYPOTHESES}
            or full_result['projection_context']['update'] != 6001
            or not full_result['same_version_replay_verified'] or 'endpoint_gradients' not in full_result):
        raise AssertionError('First6001 counters/projection/audit/replay differ')
    passed('artificial_next6001_draw_and_projection_match_continuation',dict(
        projection=full_result['projection_context'],source_draw_sha256=tree_digest(expected_batches),
        same_version_replay_verified=True,gradient_audit_present=True))
    validate_phase_optimizer(resumed.model,resumed.g_optimizer.state_dict(),
        resumed.g_optimizer.state_dict()['param_groups'],mean_steps=6001,joint_steps=3001)
    passed('next6001_named_mean6001_joint3001_D3001_steps')
    for label,module,optimizer in [('G',resumed.model,resumed.g_optimizer)]+[
            ('D_'+h,resumed.discriminators[h],optimizer) for h,optimizer in resumed.d_optimizers.items()]:
        cloned = copy.deepcopy(module)
        roundtrip = torch.optim.AdamW(cloned.parameters(),lr=.123)
        roundtrip.load_state_dict(optimizer.state_dict())
        assert_tree_bitwise_equal(optimizer.state_dict(),roundtrip.state_dict(),'optimizer_roundtrip/'+label)
    passed('G_and_all_three_D_optimizer_native_roundtrip_bitwise')
    checkpoint_path = output/'resume01_after_artificial6001.pt'
    resumed.checkpoint(checkpoint_path,purpose='periodic')
    saved = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if (saved['schema'] != SCHEMA or saved['recovery'] != identity
            or saved['model_implementation'] != MODEL_IMPLEMENTATION or saved['new_G_updates'] != 1
            or saved['new_D_updates'] != {h:1 for h in HYPOTHESES}
            or saved['lineage']['inherited_G_updates'] != 6000 or not saved['complete_step_boundary']):
        raise AssertionError('Resume01 schema/identity/lineage/boundary differs')
    for key,value in resumed.native_state_tree().items():
        assert_tree_bitwise_equal(saved[key],value,'saved/'+key)
    passed('resume01_distinct_schema_identity_recursive_optimizer_checkpoint_exact')
    rejected('checkpoint_overwrite_rejected',lambda:resumed.checkpoint(checkpoint_path),'Preserve existing')
    rejected('nonlatest_overwrite_rejected',lambda:resumed.checkpoint(output/'other.pt',allow_overwrite=True),'Only latest')
    resumed._step_in_progress = True
    rejected('midstep_cannot_masquerade_as_periodic',lambda:resumed.checkpoint(output/'bad_boundary.pt'),'Incomplete/failed')
    resumed.checkpoint(output/'artificial_forensic_failure.pt',purpose='failure')
    failure = torch.load(output/'artificial_forensic_failure.pt',map_location='cpu',weights_only=False)
    resumed._step_in_progress = False
    if failure['checkpoint_purpose'] != 'failure' or failure['failure_may_be_mid_step'] is not True or failure['complete_step_boundary'] is not False:
        raise AssertionError('Failure checkpoint role differs')
    passed('forensic_failure_role_is_explicit_and_nonresumable')
    simple = engine('simple')
    if type(simple.model) is not N01Response008Fix01:
        raise AssertionError('Simple does not explicitly use model008_fix01')
    do_step(simple,'mean_response');simple_result=do_step(simple,'joint_stochastic')
    if simple.new_G_updates != 2 or simple.new_D_updates != {h:1 for h in HYPOTHESES}:
        raise AssertionError('Fresh simple accounting differs')
    passed('fresh_simple_fix01_mean_and_joint_execute_from_common',dict(
        model_implementation=MODEL_IMPLEMENTATION,G_updates=2,D_updates=simple_result['D_updates']))
    # Low-level parser counterexamples do not weaken the sole scientific hash.
    target = engine()
    first_mean = next(identifier for identifier,(name,_) in enumerate(target.model.named_parameters())
                      if not name.startswith(STOCHASTIC_PREFIXES))
    first_joint = next(identifier for identifier,(name,_) in enumerate(target.model.named_parameters())
                       if name.startswith(STOCHASTIC_PREFIXES))
    mutations = [
        ('wrong_purpose',lambda s:s.update(checkpoint_purpose='failure'),'complete periodic'),
        ('midstep_marker',lambda s:s.update(failure_may_be_mid_step=True),'complete periodic'),
        ('automatic_resume',lambda s:s.update(automatic_resume_allowed=True),'complete periodic'),
        ('wrong_schema',lambda s:s.update(schema=SCHEMA),'schema/arm/config'),
        ('wrong_arm',lambda s:s.update(arm='simple'),'schema/arm/config'),
        ('wrong_seed',lambda s:s['config']['seeds'].update(train_response=1),'schema/arm/config'),
        ('wrong_statistic',lambda s:s['statistics']['model'].update(q_ref=999.),'statistics'),
        ('wrong_fix01_identity',lambda s:s['recovery'].update(source_manifest_sha256='d'*64),'identity'),
        ('wrong_model_identity',lambda s:s.update(model_implementation='n01.model008.N01Response008'),'model identity'),
        ('wrong_G_count',lambda s:s.update(G_updates=6398),'complete-cycle counters'),
        ('partial_D_count',lambda s:s['D_updates'].update(fast=3001),'complete-cycle counters'),
        ('wrong_new_G_count',lambda s:s.update(new_G_updates=6000),'executed-new counters'),
        ('wrong_parent_lineage',lambda s:s['lineage'].update(inherited_G_updates=0),'parent lineage'),
        ('wrong_last_metrics',lambda s:s['last_metrics'].update(G_updates=6398),'last completed'),
        ('wrong_projection_count',lambda s:s['last_metrics']['projection_context'].update(update=6399),'projection update'),
        ('uniform_G_step6000_rejected',lambda s:s['g_optimizer']['state'][first_joint].update(step=torch.tensor(6000.)),'phase step'),
        ('wrong_mean_step',lambda s:s['g_optimizer']['state'][first_mean].update(step=torch.tensor(5999.)),'phase step'),
        ('missing_joint_history',lambda s:s['g_optimizer']['state'].pop(first_joint),'Missing named'),
        ('missing_mean_history',lambda s:s['g_optimizer']['state'].pop(first_mean),'Missing named'),
        ('changed_G_optimizer_lr',lambda s:s['g_optimizer']['param_groups'][0].update(lr=.001),'optimizer_param_groups'),
        ('changed_G_optimizer_order',lambda s:s['g_optimizer']['param_groups'][0]['params'].reverse(),'optimizer_param_groups'),
        ('wrong_D_step',lambda s:s['d_optimizers']['fast']['state'][0].update(step=torch.tensor(2999.)),'phase step'),
        ('negative_second_moment',lambda s:s['g_optimizer']['state'][first_mean]['exp_avg_sq'].fill_(-1),'Negative AdamW'),
        ('nonfinite_first_moment',lambda s:s['g_optimizer']['state'][first_mean]['exp_avg'].fill_(float('nan')),'finiteness'),
        ('wrong_source_n',lambda s:s['source_sampler'].update(n=1),'population'),
        ('wrong_source_position',lambda s:s['source_sampler'].update(position=1),'draw position'),
        ('wrong_source_visits',lambda s:s['source_sampler']['visits'].add_(1),'visit counts'),
        ('wrong_source_permutation',lambda s:s['source_sampler']['order'].zero_(),'permutation'),
        ('wrong_real_epochs',lambda s:s['real_sampler'].update(completed_epochs=0),'draw position'),
        ('missing_stream',lambda s:s['streams'].pop('response'),'stream keys'),
        ('wrong_stream_dtype',lambda s:s['streams'].update(response=s['streams']['response'].long()),'RNG dtype/shape'),
        ('wrong_noise_reference',lambda s:s.update(noise005_state_sha256='0'*64),'frozen references'),
        ('noise_refit_history',lambda s:s.update(noise_updates=1),'noise/population'),
        ('unlabelled_artificial_parser',lambda s:s.pop('artificial_parser_fixture'),'labelled artificial')]
    for name,mutate,expected in mutations:
        changed=copy.deepcopy(staged);mutate(changed)
        rejected(name,lambda:target._validate_import_state(changed,fixed_identity),expected)
    rejected('wrong_engineering_file_hash',lambda:engine().import_fix01_full_periodic6000(fixture_path,
        expected_sha256='0'*64,fix01_identity=fixed_identity),'bytes changed')
    wrong_identity=copy.deepcopy(identity);wrong_identity['run_id']='window008_fix01_20261013'
    rejected('wrong_resume01_identity_run',lambda:Response008Resume01Engine(data,statistics,config,noise,common,
        arm='full',recovery_identity=wrong_identity,device='cpu',engineering_fixture=True),'identity keys/run')
    rejected('bitwise_comparison_detects_signed_zero',lambda:assert_tree_bitwise_equal(
        torch.tensor(0.),torch.tensor(-0.),'signed_zero'),'Tensor bytes differ')

    # Genuine native state is installed with no experimental waveform decoded.
    native_path=ROOT/'runs/window008_fix01_20261013/runs/science/train/full/latest.pt'
    if sha256(native_path) != REGISTERED_IMPORT_SHA256:
        raise AssertionError('Registered native periodic6000 file bytes changed')
    native_state=torch.load(native_path,map_location='cpu',weights_only=False)
    report['actual_signal_checkpoints_read']=1
    common_path=ROOT/'provenance/window008_fix01_reference/common_initial_state.pt'
    native_common=torch.load(common_path,map_location='cpu',weights_only=False)['model']
    ns,nr=native_state['source_sampler']['n'],native_state['real_sampler']['n']
    placeholders=dict(source_carrier384=np.zeros((ns,384),np.float64),real=np.zeros((nr,256),np.float64),
        real_baseline100=np.zeros((nr,100),np.float64),source_ids=-1-np.arange(ns),real_ids=-1-np.arange(nr),
        metadata_only_placeholder=True)
    forbidden=AssertionError('Scientific metadata-only import attempted a forward/step/draw')
    with patch.object(N01Response008Fix01,'signal_forward',side_effect=forbidden), \
         patch.object(WaveformDiscriminator,'forward',side_effect=forbidden), \
         patch.object(torch.optim.AdamW,'step',side_effect=forbidden), \
         patch.object(FullCoverageSampler,'draw',side_effect=forbidden), \
         patch.object(PairedStreams,'draw',side_effect=forbidden):
        native=Response008Resume01Engine(placeholders,native_state['statistics'],base,noise,native_common,
            arm='full',recovery_identity=identity,device='cpu')
        native_receipt=native.import_fix01_full_periodic6000(native_path,
            expected_sha256=REGISTERED_IMPORT_SHA256,fix01_identity=fixed_identity)
        for key,value in native.native_state_tree().items():
            assert_tree_bitwise_equal(native_state[key],value,'genuine_native/'+key)
        rejected('metadata_only_adapter_cannot_step',lambda:native.step(audit=True,verify_replay=True),'Metadata-only CPU')
        rejected('metadata_only_adapter_cannot_write_training_checkpoint',lambda:native.checkpoint(output/'invalid_science.pt'),'Metadata-only CPU')
        hash_reject=Response008Resume01Engine(placeholders,native_state['statistics'],base,noise,native_common,
            arm='full',recovery_identity=identity,device='cpu')
        rejected('sole_scientific_hash_enforced',lambda:hash_reject.import_fix01_full_periodic6000(native_path,
            expected_sha256='0'*64,fix01_identity=fixed_identity),'Unregistered scientific checkpoint hash')
    native_steps=native_receipt['G_optimizer_named_steps']
    if ({step:sum(value==step for value in native_steps.values()) for step in (6000,3000)} != {6000:167,3000:13}
            or native.new_G_updates != 0 or native.new_D_updates != {h:0 for h in HYPOTHESES}):
        raise AssertionError('Genuine native phase slots/counters differ')
    write_json(output/'NATIVE_IMPORT_RECEIPT.json',native_receipt)
    passed('genuine_periodic6000_native_import_all_state_bitwise_zero_scientific_forward',dict(
        source_checkpoint_sha256=REGISTERED_IMPORT_SHA256,mean_parameter_steps6000=167,joint_parameter_steps3000=13,
        D_steps_each=3000,G_updates=6000,new_G_updates=0,metadata_only_placeholder=True,
        scientific_model_forwards=0,scientific_optimizer_updates=0,scientific_stream_or_sampler_draws=0))
    if sha256(native_path) != REGISTERED_IMPORT_SHA256 or sha256(fixture_path) != fixture_hash:
        raise AssertionError('Read-only checkpoint bytes changed during QA')
    for relative,expected in protected.items():
        if sha256(ROOT/relative) != expected:
            raise AssertionError('Protected source bytes changed during QA: '+relative)
    for path in own_paths:
        ast.parse(path.read_text(encoding='utf8'))
        if sha256(path) != own_hashes[path.relative_to(ROOT).as_posix()]:
            raise AssertionError('Own source changed during QA')
    if report['engineering_G_updates'] != 4 or report['engineering_D_updates'] != 6:
        raise AssertionError('Artificial update accounting exceeded the intended G4/D6')
    passed('all_original_fix01_and_new_source_bytes_preserved_during_QA')
    report.update(status='passed_CPU_artificial_and_native_metadata_import_QA',passed=len(checks),total=len(checks))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args();output=Path(args.out).resolve()
    if os.name=='nt' and output.drive.upper() not in ('E:','D:'):
        raise ValueError('CPU QA output must stay on E/D')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='n01/window008-resume01-engine-QA/v1',status='running',UTC=utc(),checks=[],
        artificial_forward_arrays_only=True,actual_SOURCE_REAL_DEV_numeric_arrays_decoded=0,
        scientific_model_forwards=0,scientific_optimizer_updates=0,GPU_used=False,
        scientific_training_dispatched=False)
    began=time.monotonic();exit_code=0
    try:
        run(output,report)
    except BaseException as error:
        report.update(status='failed_CPU_resume01_engine_QA',error=repr(error),traceback=traceback.format_exc())
        exit_code=1
    finally:
        report.update(elapsed_seconds=time.monotonic()-began,completed_UTC=utc())
        write_json(output/'ENGINEERING_RECEIPT.json',report)
    print(json.dumps(dict(status=report['status'],passed=report.get('passed'),total=report.get('total'),
        receipt=str(output/'ENGINEERING_RECEIPT.json'),error=report.get('error'))),flush=True)
    return exit_code


if __name__=='__main__':
    raise SystemExit(main())
