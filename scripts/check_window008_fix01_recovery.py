"""CPU artificial-only fix01 migration/driver QA. Never opens SCI/DEV data.

The stage3000 parser fixture is explicitly fabricated from one artificial mean
update, then assigned synthetic counter/moment-step/sampler/RNG states. It is
not evidence that 3000 engineering or scientific updates were executed.
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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import torch

from check_window008_engine import fixture, serial
from n01.data import FullCoverageSampler, sha256, write_json
from n01.response008 import HYPOTHESES, PairedStreams, Response008Engine, common_initial_state, response_stage, state_digest
from n01.response008_fix01 import (RUN_ID, Response008Fix01Engine, assert_tree_equal,
    parameter_signature, tree_digest)
from n01.train import atomic_torch_save, utc
from n01.train_window008_fix01 import (bound, update_accounting, validate_legacy_receipts,
    validate_plan, validate_training_data, save_failure, main as training_main)


def run(output, report):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    base = json.loads((ROOT/'window008_config.json').read_text(encoding='utf8'))
    fix = json.loads((ROOT/'window008_fix01_config.json').read_text(encoding='utf8'))
    validate_plan(base, fix)
    config = copy.deepcopy(base)
    config.update(distribution_microbatches=2,source_parents_per_microbatch=4,responses_per_parent=2,
                  real_rows_per_microbatch=8,engineering_fixture=True)
    data, statistics, fixture_hash = fixture(rows=128)
    noise_path = ROOT/base['noise_reference']['path']
    if sha256(noise_path) != base['noise_reference']['sha256']:
        raise ValueError('Bound fixed005 bytes changed')
    noise = torch.load(noise_path,map_location='cpu',weights_only=False)
    common = common_initial_state(statistics['model'],base['seeds']['initialization'])
    identity = dict(run_id=RUN_ID,config_sha256='e'*64,source_manifest_sha256='f'*64,
                    base_configuration_sha256='a'*64,base_source_manifest_sha256='b'*64)
    report.update(fixture_sha256=fixture_hash,noise005_file_sha256=sha256(noise_path),
        common_initial_state_sha256=state_digest(common),scientific_checkpoints_read=0,
        artificial_stage3000_is_fabricated_parser_fixture=True,engineering_G_updates=0,engineering_D_updates=0)
    checks = report['checks']
    source_paths = (ROOT/'src/n01/model008_fix01.py',ROOT/'src/n01/response008_fix01.py',
                    ROOT/'src/n01/train_window008_fix01.py',Path(__file__))
    before_sources = {str(path.relative_to(ROOT)):sha256(path) for path in source_paths}
    def passed(name, details=None):
        checks.append(dict(name=name,passed=True,details=details))
    def rejected(name, call, expected=None):
        try:
            call()
        except (ValueError, RuntimeError, FileExistsError, KeyError, TypeError) as error:
            if expected is not None and expected not in str(error):
                raise AssertionError('Wrong rejection for '+name+': '+str(error)) from error
            passed(name,dict(error_type=type(error).__name__,error=str(error)))
        else:
            raise AssertionError('Expected rejection: '+name)
    def fixed(arm):
        return Response008Fix01Engine(data,statistics,config,noise,common,arm=arm,
            recovery_identity=identity,device='cpu',engineering_fixture=True)
    def do_step(engine, stage):
        engine.set_stage(stage)
        result = engine.step(audit=True,verify_replay=True)
        report['engineering_G_updates'] += 1
        report['engineering_D_updates'] += 0 if stage == 'mean_response' else 3
        return result
    def compare_engines(first, second):
        for key, left, right in (
            ('model',first.model.state_dict(),second.model.state_dict()),
            ('G_optimizer',first.g_optimizer.state_dict(),second.g_optimizer.state_dict()),
            ('D',first.discriminators.state_dict(),second.discriminators.state_dict()),
            ('D_optimizers',{h:o.state_dict() for h,o in first.d_optimizers.items()},
                            {h:o.state_dict() for h,o in second.d_optimizers.items()}),
            ('source_sampler',first.source_sampler.state_dict(),second.source_sampler.state_dict()),
            ('real_sampler',first.real_sampler.state_dict(),second.real_sampler.state_dict()),
            ('streams',first.streams.state_dict(),second.streams.state_dict())):
            assert_tree_equal(left,right,key)

    legacy = Response008Engine(data,statistics,config,noise,common,arm='full',device='cpu')
    replacement = fixed('full')
    compare_engines(legacy,replacement)
    passed('engine_replacement_model_parameter_order_optimizer_exact',dict(parameters=len(parameter_signature(replacement.model))))
    do_step(legacy,'mean_response'); do_step(replacement,'mean_response')
    compare_engines(legacy,replacement)
    passed('full_artificial_mean_update_bitwise_equal_to_legacy')
    old_path = output/'legacy_one_mean.engineering.pt'
    legacy.checkpoint(old_path)
    staged = torch.load(old_path,map_location='cpu',weights_only=False)
    staged['G_updates']=3000;staged['last_metrics']['G_updates']=3000
    for slot in staged['g_optimizer']['state'].values():
        slot['step']=torch.full_like(slot['step'],3000)
    for key,n,rows,seed in (
        ('source_sampler',len(data['source_ids']),config['source_parents_per_microbatch'],base['seeds']['source_sampler']),
        ('real_sampler',len(data['real_ids']),config['real_rows_per_microbatch'],base['seeds']['real_sampler'])):
        sampler=FullCoverageSampler(n,seed)
        sampler.draw(3000*config['distribution_microbatches']*rows)
        staged[key]=sampler.state_dict()
    streams=PairedStreams(base['seeds']['train_response'],base['seeds']['train_noise005'],base['seeds']['train_extension'])
    for _ in range(3000*config['distribution_microbatches']):
        streams.draw(config['source_parents_per_microbatch']*config['responses_per_parent'],'cpu')
    staged['streams']=streams.state_dict()
    staged['artificial_parser_fixture']=True
    staged_path=output/'fabricated_stage3000.engineering.pt'
    atomic_torch_save(staged,staged_path)
    before=sha256(staged_path)
    imported=fixed('full')
    receipt=imported.import_legacy_full_stage3000(staged_path,expected_sha256=before)
    if not (receipt['status']=='passed' and imported.update==3000 and imported.new_G_updates==0 and all(value==0 for value in imported.d_updates.values())):
        raise AssertionError('Wrong import state/counters')
    for key, value in (('model',imported.model.state_dict()),('g_optimizer',imported.g_optimizer.state_dict()),
        ('source_sampler',imported.source_sampler.state_dict()),('real_sampler',imported.real_sampler.state_dict()),
        ('streams',imported.streams.state_dict())):
        assert_tree_equal(staged[key],value,key)
    passed('fabricated_stage3000_all_state_import_exact',receipt['state_digests'])
    incomplete=fixed('full')
    original_assert=assert_tree_equal
    def interrupt_after_state_install(left,right,label='state'):
        if label == 'restored/model':
            raise ValueError('Injected post-load verification failure')
        return original_assert(left,right,label)
    with patch('n01.response008_fix01.assert_tree_equal',side_effect=interrupt_after_state_install):
        rejected('postload_validation_failure_preserved',lambda:incomplete.import_legacy_full_stage3000(staged_path,expected_sha256=before),'Injected')
    if incomplete.new_G_updates != 0 or any(incomplete.new_D_updates.values()) or incomplete.resume_validation is not None:
        raise AssertionError('Unfinished import was counted as new training')
    passed('failed_import_after_state_install_has_zero_new_G_D')
    rejected('repeat_legacy_import_rejected',lambda: imported.import_legacy_full_stage3000(staged_path,expected_sha256=before),'once')
    rejected('unknown_generic_resume_rejected',lambda: imported.restore(staged_path),'No automatic')
    rejected('first_full_update_wrong_stage_rejected',lambda: imported.step(audit=True,verify_replay=True),'joint3001')
    imported.set_stage(response_stage(imported.update,config))
    rejected('first_full_update_missing_audit_rejected',lambda: imported.step(audit=False,verify_replay=True),'joint3001')
    rejected('first_full_update_missing_replay_rejected',lambda: imported.step(audit=True,verify_replay=False),'joint3001')
    first=do_step(imported,'joint_stochastic')
    if first['G_updates']!=3001 or first['new_G_updates']!=1 or first['projection_context']['update']!=3001 or any(v!=1 for v in imported.d_updates.values()):
        raise AssertionError('First update/counters/projection differ')
    passed('first_resumed_update_3001_projection_replay_and_new_counts',dict(G=3001,new_G=1,D=imported.d_updates,projection=first['projection_context']))
    checkpoint_path=output/'fix01_one_joint.engineering.pt'
    imported.checkpoint(checkpoint_path,purpose='periodic')
    loaded=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if loaded['schema']!='response008_fix01/v1' or loaded['recovery']!=identity or loaded['lineage']['inherited_G_updates']!=3000 or loaded['new_G_updates']!=1:
        raise AssertionError('Fix01 schema/identity/lineage differs')
    passed('fix01_checkpoint_has_new_schema_and_explicit_lineage')
    failure_dir=output/'artificial_failed_attempt';failure_dir.mkdir()
    failed_receipt=save_failure(failure_dir,ValueError('Artificial injected failure'),[],imported,
                               began=time.monotonic(),legacy_validated=True)
    saved_failure=torch.load(failure_dir/'FAILURE_CHECKPOINT.pt',map_location='cpu',weights_only=False)
    saved_context=torch.load(failure_dir/'FAILURE_CONTEXT.pt',map_location='cpu',weights_only=False)
    if (failed_receipt['new_execution_G_updates']!=1 or failed_receipt['new_execution_D_updates']!=3
            or saved_failure['checkpoint_purpose']!='failure' or saved_failure['automatic_resume_allowed'] is not False
            or saved_failure['failure_may_be_mid_step'] is not True):
        raise AssertionError('Forensic failure state/counts differ')
    assert_tree_equal(saved_context,imported.last_context,'failure_context')
    passed('driver_failure_saves_actual_counts_checkpoint_and_context')
    rejected('fixed_checkpoint_overwrite_rejected',lambda: imported.checkpoint(checkpoint_path),'Preserve existing')
    rejected('nonlatest_overwrite_rejected',lambda: imported.checkpoint(output/'notlatest.pt',allow_overwrite=True),'Only latest')
    rejected('fix01_checkpoint_cannot_be_legacy_import',lambda: fixed('full').import_legacy_full_stage3000(checkpoint_path,expected_sha256=sha256(checkpoint_path)),'schema/arm/config')
    simple=fixed('simple')
    rejected('simple_cannot_import_full',lambda: simple.import_legacy_full_stage3000(staged_path,expected_sha256=before),'fresh full')
    rejected('deterministic_cannot_be_retrained',lambda: fixed('deterministic'),'never retrains')
    old_simple=Response008Engine(data,statistics,config,noise,common,arm='simple',device='cpu')
    compare_engines(old_simple,simple)
    do_step(old_simple,'mean_response');do_step(simple,'mean_response')
    compare_engines(old_simple,simple)
    do_step(old_simple,'joint_stochastic');do_step(simple,'joint_stochastic')
    compare_engines(old_simple,simple)
    passed('fresh_simple_mean_and_joint_bitwise_equal_to_legacy')
    # Reuse one labelled temporary mutation file; the immutable valid fixture remains.
    mutation_path=output/'temporary_rejected_mutation.engineering.pt'
    target=fixed('full')
    first_parameter=next(iter(staged['g_optimizer']['state']))
    mutations=[
        ('wrong_arm',lambda s:s.update(arm='simple'),'schema/arm/config'),
        ('wrong_schema',lambda s:s.update(schema='response008_fix01/v1'),'schema/arm/config'),
        ('failure_step3269',lambda s:s.update(G_updates=3269),'stage3000'),
        ('wrong_stage',lambda s:s.update(stage='joint_stochastic'),'stage3000'),
        ('nonzero_D',lambda s:s['D_updates'].update(fast=269),'stage3000'),
        ('changed_config',lambda s:s['config']['seeds'].update(train_response=1),'schema/arm/config'),
        ('changed_statistics',lambda s:s['statistics']['model'].update(q_ref=999.),'statistics'),
        ('changed_common',lambda s:s.update(common_initial_state_sha256='0'*64),'frozen references'),
        ('changed_initial_D',lambda s:s.update(initial_D_sha256='0'*64),'frozen references'),
        ('changed_noise',lambda s:s.update(noise005_state_sha256='0'*64),'frozen references'),
        ('changed_G_optimizer_step',lambda s:s['g_optimizer']['state'][first_parameter].update(step=torch.tensor(2999.)),'AdamW step'),
        ('missing_mean_optimizer_slot',lambda s:s['g_optimizer']['state'].pop(first_parameter),'missing its optimizer'),
        ('changed_G_optimizer_order',lambda s:s['g_optimizer']['param_groups'][0]['params'].reverse(),'optimizer_param_groups'),
        ('changed_G_optimizer_lr',lambda s:s['g_optimizer']['param_groups'][0].update(lr=.001),'optimizer_param_groups'),
        ('changed_D_buffer',lambda s:next(iter(s['discriminators'].values())).add_(1),'untrained_D_state'),
        ('wrong_source_population',lambda s:s['source_sampler'].update(n=1),'population'),
        ('wrong_source_position',lambda s:s['source_sampler'].update(position=1),'draw position'),
        ('wrong_source_visits',lambda s:s['source_sampler']['visits'].add_(1),'visit counts'),
        ('wrong_source_order',lambda s:s['source_sampler']['order'].zero_(),'permutation'),
        ('wrong_real_epochs',lambda s:s['real_sampler'].update(completed_epochs=0),'draw position'),
        ('missing_stream',lambda s:s['streams'].pop('response'),'stream keys'),
        ('wrong_stream_dtype',lambda s:s['streams'].update(response=s['streams']['response'].long()),'RNG dtype/shape'),
        ('wrong_last_metrics',lambda s:s['last_metrics'].update(G_updates=3269),'last metrics'),
        ('noise_updates',lambda s:s.update(noise_updates=1),'noise/population')]
    for name,mutate,expected in mutations:
        changed=copy.deepcopy(staged);mutate(changed);atomic_torch_save(changed,mutation_path)
        rejected(name,lambda:target.import_legacy_full_stage3000(mutation_path,expected_sha256=sha256(mutation_path)),expected)
    rejected('unbound_file_hash_rejected',lambda:target.import_legacy_full_stage3000(staged_path,expected_sha256='0'*64),'bytes changed')
    if sha256(staged_path)!=before:
        raise AssertionError('Legacy fixture bytes changed')
    passed('legacy_checkpoint_bytes_unchanged')
    rejected('path_escape_rejected',lambda:bound(output,'../outside'),'in-root')
    rejected('absolute_reference_rejected',lambda:bound(output,str(output/'outside')),'in-root')
    bad_plan=copy.deepcopy(fix);bad_plan['new_execution_G_updates']=57000
    rejected('budget_mismatch_rejected',lambda:validate_plan(base,bad_plan),'budgets')
    bad_plan=copy.deepcopy(fix);bad_plan['resume']['D_updates_per_hypothesis']=269
    rejected('resume_policy_mismatch_rejected',lambda:validate_plan(base,bad_plan),'resume policy')
    rows=[dict(arm='deterministic',new_G_updates=0,new_D_updates={h:0 for h in HYPOTHESES}),
          dict(arm='full',new_G_updates=16000,new_D_updates={h:16000 for h in HYPOTHESES}),
          dict(arm='simple',new_G_updates=19000,new_D_updates={h:16000 for h in HYPOTHESES})]
    counts=update_accounting(rows)
    expected_counts=dict(new_execution_G_updates=35000,new_execution_D_updates=96000,effective_comparison_G_updates=57000,
        effective_comparison_D_updates=144000,historical_actual_G_updates=57269,historical_actual_D_updates=144807)
    if any(counts[k]!=v for k,v in expected_counts.items()):
        raise AssertionError('Final accounting mismatch')
    passed('new_effective_historical_counts_separated',counts)
    # Simulate D advancing before a failed G; it must remain in actual new counts.
    failed=type('FailureCounterFixture',(),dict(arm='full',new_G_updates=269,new_D_updates={'fast':270,'slow':269,'common_level':269}))()
    failure_counts=update_accounting(rows[:1],failed)
    if failure_counts['new_execution_G_updates']!=269 or failure_counts['new_execution_D_updates']!=808:
        raise AssertionError('Partial D failure accounting lost updates')
    passed('midstep_partial_D_updates_preserved',failure_counts)
    unknown=update_accounting([],legacy_validated=False)
    if unknown['new_execution_G_updates']!=0 or unknown['effective_comparison_G_updates'] is not None:
        raise AssertionError('Unvalidated prior history was promoted')
    passed('binding_failure_does_not_claim_valid_inherited_endpoints')
    rejected_out=output/'artificial_bind_failure'
    missing=output/'intentionally_missing.json'
    argv=[]
    for name in ('config','recovery-config','source-manifest','container','identities','dataset','statistics'):
        argv.extend(['--'+name,str(missing)])
    argv.extend(['--out',str(rejected_out),'--device','cpu'])
    try:
        training_main(argv)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Intentionally missing binding was accepted')
    binding_failure=json.loads((rejected_out/'FAILURE.json').read_text(encoding='utf8'))
    preserved_hash=sha256(rejected_out/'FAILURE.json')
    if binding_failure['new_execution_G_updates']!=0 or binding_failure['legacy_references_validated'] is not False:
        raise AssertionError('Binding failure claimed new or inherited updates')
    rejected('driver_rejects_existing_failed_output',lambda:training_main(argv),'Fresh fix01 output')
    if sha256(rejected_out/'FAILURE.json')!=preserved_hash:
        raise AssertionError('Duplicate start changed first failure')
    passed('driver_binding_failure_saved_and_first_attempt_preserved')
    # Parser-only fake native receipts: never copied from the actual job.
    stamp=lambda second:f'2026-01-01T00:00:{second:02d}+00:00'
    det=dict(UTC=stamp(2),arm='deterministic',G_updates=19000,D_updates={h:16000 for h in HYPOTHESES},
        final_sha256=fix['reference_inputs']['deterministic_final']['sha256'],coverage=dict(source_unique=6593,real_unique=21047))
    docs=dict(legacy_native_exit=dict(exit_code=1,reason=None,supervisor_pid=11,child_pid=12,started_utc=stamp(0),ended_utc=stamp(4)),
        legacy_launch=dict(supervisor_pid=11,child_pid=12,started_utc=stamp(0),command=['timeout','--signal=TERM','--kill-after=15s','28800s',
            '/artificial/venv/bin/python','-B','-m','n01.train_window008','--container',base['remote_training_container'],
            '--identities',base['remote_root']+'/data/PARENT_IDENTITIES.npz','--dataset',base['remote_root']+'/'+base['dataset'],
            '--statistics',base['remote_root']+'/'+base['statistics'],'--out',base['remote_root']+'/runs/science',
            '--config',base['remote_root']+'/window008_config.json','--source-manifest',base['remote_root']+'/WINDOW008_SOURCE_MANIFEST.json',
            '--device','cuda:0']),
        legacy_failure=dict(UTC=stamp(3),run_id=base['run_id'],status='failed_or_bounded_stop_preserved',G_updates=22269,D_updates=48807,
            current_arm='full',completed_arms=[det],DEV_waveforms_read=False,evaluation_complete=False,
            error='Invalid Window008 shape canonical charge',C2ST_updates=0,noise_updates=0,population_updates=0),
        legacy_binding=dict(UTC=stamp(1),pid=13,run_id=base['run_id'],config_sha256=fix['base_configuration_sha256'],
            source_manifest_sha256=fix['legacy_source_manifest_sha256'],dataset_sha256=base['dataset_sha256'],statistics_sha256=base['statistics_sha256'],
            raw_container_sha256=base['train_container_sha256'],identities_sha256=base['identities_sha256'],DEV_waveforms_read=False),
        legacy_training_inputs=dict(SOURCE_rows=6593,REAL_rows=21047,effective_samples=256,internal_carrier_samples=384,
            DEV_waveforms_read=False,source_count_weights='all_one',new_noise_updates=0,new_population_updates=0),
        deterministic_training_complete=det)
    validate_legacy_receipts(base,fix,docs)
    passed('native_exit1_chain_explicitly_accepted_as_failure_with_retained_arm')
    for name, mutate, expected in (
        ('forged_legacy_exit0',lambda d:d['legacy_native_exit'].update(exit_code=0),'remain exit1'),
        ('legacy_PID_mismatch',lambda d:d['legacy_native_exit'].update(child_pid=999),'PID chain'),
        ('legacy_wrong_module',lambda d:d['legacy_launch']['command'].__setitem__(7,'n01.train_other'),'bound Window008'),
        ('legacy_failure_counts_wrong',lambda d:d['legacy_failure'].update(G_updates=22000),'accounting'),
        ('legacy_DEV_read',lambda d:d['legacy_binding'].update(DEV_waveforms_read=True),'binding'),
        ('legacy_chronology_wrong',lambda d:d['legacy_native_exit'].update(ended_utc=stamp(1)),'chronology')):
        changed=copy.deepcopy(docs);mutate(changed)
        rejected(name,lambda:validate_legacy_receipts(base,fix,changed),expected)
    for path in (ROOT/'src/n01/response008_fix01.py',ROOT/'src/n01/train_window008_fix01.py',Path(__file__)):
        ast.parse(path.read_text(encoding='utf8'))
    passed('new_sources_parse')
    after_sources={str(path.relative_to(ROOT)):sha256(path) for path in source_paths}
    if before_sources!=after_sources:
        raise AssertionError('Source files changed during artificial QA')
    passed('source_bytes_unchanged_during_QA')
    report.update(status='passed_artificial_CPU_recovery_QA',passed=len(checks),total=len(checks),source_sha256=after_sources)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args();output=Path(args.out).resolve()
    if os.name=='nt' and output.drive.upper() not in ('E:','D:'):
        raise ValueError('Artificial QA output must stay on E/D')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='n01/window008-fix01-recovery-QA/v1',status='running',UTC=utc(),checks=[],
        artificial_data_only=True,actual_SCI_DEV_arrays_decoded=0,scientific_model_forwards=0,
        scientific_optimizer_updates=0,GPU_used=False,actual_native_training_dispatched=False)
    began=time.monotonic();exit_code=0
    try:
        run(output,report)
    except BaseException as error:
        report.update(status='failed_artificial_CPU_recovery_QA',error=repr(error),traceback=traceback.format_exc())
        exit_code=1
    finally:
        report.update(elapsed_seconds=time.monotonic()-began,completed_UTC=utc())
        write_json(output/'ENGINEERING_RECEIPT.json',report)
    print(json.dumps(dict(status=report['status'],passed=report.get('passed'),total=report.get('total'),
        receipt=str(output/'ENGINEERING_RECEIPT.json'),error=report.get('error'))),flush=True)
    return exit_code


if __name__=='__main__':
    raise SystemExit(main())
