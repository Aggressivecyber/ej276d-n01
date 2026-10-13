"""Bindings for registered zero-optimization proposal diagnostics."""
from pathlib import Path
import os
import window008_resume01_evaluation_common as main
from window008_resume01_evaluation_common import ROOT, TRAIN_MANIFEST, sha, read, write_new, require, bound, verify_files

EVAL_MANIFEST='WINDOW008_RESUME01_DIAGNOSTICS_SOURCE_MANIFEST.json'
POLICY='window008_resume01_diagnostics_policy.json'


def evaluation_binding(root=ROOT):
    root=Path(root).resolve()
    cfg,training,main_policy,main_manifest=main.evaluation_binding(root)
    policy=read(root/POLICY);manifest=read(root/EVAL_MANIFEST)
    require(manifest['schema']=='n01/window008-resume01-diagnostics-source-manifest/v1', 'Wrong diagnostic freeze schema')
    require(manifest['before_DEV_numeric_decode'] is True and manifest['before_scientific_model_forward'] is True,
        'Diagnostics must be registered before numerical access')
    require(manifest['policy_sha256']==sha(root/POLICY)
        and manifest['main_evaluation_source_manifest_sha256']==sha(root/main.EVAL_MANIFEST), 'Diagnostic freeze identity differs')
    required={'src/n01/diagnose_window008_resume01.py',POLICY,main.EVAL_MANIFEST,
        'scripts/window008_resume01_diagnostics_common.py','scripts/run_window008_resume01_diagnostics.py',
        'scripts/start_window008_resume01_diagnostics_remote.py','scripts/supervise_window008_resume01_diagnostics.py',
        'scripts/launch_window008_resume01_diagnostics.sh','scripts/check_window008_resume01_diagnostics.py'}
    require(required<=set(manifest['files']), 'Missing diagnostic source binding')
    verify_files(root,manifest['files'])
    require(policy['resource_limits']==main_policy['resource_limits'] and policy['CPU_threads']==12, 'Diagnostic resources differ')
    return cfg,training,policy,manifest


def paths_for(kind,root=ROOT):
    require(kind=='primary','Only one registered diagnostics execution')
    root=Path(root).resolve();job=root/'runs/diagnostic_jobs'/kind
    return dict(root=root,run=root/'runs/science',job=job,artifacts=job/'artifacts',
        supervisor=job/'supervisor',primary=root/'runs/evaluation_jobs/primary')


def training_completion(root=ROOT,cfg=None):
    training=main.training_completion(root,cfg)
    primary=main.completed_job('primary',root);replay=main.completed_job('replay01',root)
    return dict(training=training,main_summary_sha256=sha(primary['paths']['artifacts']/'SUMMARY.json'),
        main_native_sha256=sha(primary['paths']['supervisor']/'exit_receipt.json'),
        replay_native_sha256=sha(replay['paths']['supervisor']/'exit_receipt.json'),
        exact_replay_sha256=sha(replay['paths']['job']/'REPLAY_VERIFICATION.json'))


def core_command(root,executable):
    root=Path(root).resolve();cfg=read(root/'window008_config.json');p=paths_for('primary',root)
    return [executable,'-B','-m','n01.diagnose_window008_resume01','--out',str(p['run']),
        '--config',str(root/'window008_config.json'),'--resume-config',str(root/'window008_resume01_config.json'),
        '--dataset',str(root/cfg['dataset']),'--statistics',str(root/cfg['statistics']),
        '--raw-container',cfg['remote_training_container'],'--identities',str(root/'data/PARENT_IDENTITIES.npz'),
        '--source-manifest',str(root/main.EVAL_MANIFEST),'--evaluation-policy',str(root/main.POLICY),
        '--policy',str(root/POLICY),'--addon-source-manifest',str(root/EVAL_MANIFEST),
        '--primary-artifacts',str(root/'runs/evaluation_jobs/primary/artifacts'),
        '--primary-native-receipt',str(root/'runs/evaluation_jobs/primary/supervisor/exit_receipt.json'),
        '--replay-native-receipt',str(root/'runs/evaluation_jobs/replay01/supervisor/exit_receipt.json'),
        '--destination',str(p['artifacts']),'--device','cuda:0']


def artifact_integrity(destination):
    destination=Path(destination).resolve()
    require(not (destination/'FAILED.json').exists(), 'Diagnostic failure preserved')
    summary=read(destination/'SUMMARY.json')
    require(summary['schema']=='n01/window008-resume01-diagnostics/v1'
        and summary['status']=='registered_frozen_endpoint_diagnostics_complete', 'Diagnostic execution incomplete')
    require(summary['scientific_status']=='response_match_not_established', 'Unsupported scientific status')
    for key in ('optimizer_updates','G_updates','D_updates','C2ST_updates','noise_updates','population_updates'):
        require(summary[key]==0, 'Diagnostics may not optimize')
    require(summary['DEV_numeric_decoded'] is True and summary['artificial_arrays_only'] is False
        and summary['all_selected_parent_draw_rows_preserved'] is True and summary['frozen_states_rechecked'] is True
        and summary['scientific_model_forward_rows']==29952 and summary['repeat_selection_parent_count']==128
        and summary['repeat_draws_per_parent']==64 and summary['intervention_parent_count']==32
        and summary['intervention_draws_per_parent']==8, 'Incomplete diagnostic scope/freeze/counter contract')
    counts=read(destination/'ACTUAL_COUNTERS.json')
    require(counts==summary['actual_forward_counts'], 'Diagnostic counter receipt differs from summary')
    require(counts['primary']==dict(signal_forward_rows=28416,observation_rows=85248,noise005_forward_rows=8704)
        and counts['total_signal_forward_rows']==29952 and counts['total_observation_rows']==89856
        and counts['total_noise005_forward_rows']==9216, 'Diagnostic total forward accounting differs')
    expected_groups=dict(repeat=(24576,73728,8192),shape=(1536,4608,256),Q_trend=(2304,6912,256),
        identity_order_replay=(1536,4608,512))
    require(set(counts['groups'])==set(expected_groups), 'Diagnostic family inventory differs')
    for name,expected in expected_groups.items():
        group=counts['groups'][name]
        require(tuple(group[k] for k in ('signal_forward_rows','observation_rows','noise005_forward_rows'))==expected,
            'Diagnostic family forward accounting differs: '+name)
        require(group['per_arm']=={a:expected[0]//3 for a in ('deterministic','full','simple')}
            and group['per_arm_hypothesis']=={a+'_'+h:expected[1]//9 for a in ('deterministic','full','simple')
                for h in ('fast','slow','common_level')}, 'Diagnostic arm/hypothesis row accounting differs')
    for split,parents in (('TRAIN',6593),('DEV',512)):
        population=summary['population'][split]
        require(population['parents']==parents and population['rows']==parents*4 and population['response_forward_rows']==0
            and population['mathematical_waveform_mixture_exact'] is True, 'Population diagnostic scope differs')
    required_files={'FROZEN_INPUTS.json','ADDON_PREREGISTERED_INPUTS.json','ADDON_FROZEN_INPUTS.json','SELECTIONS.npz',
        'repeat_latents.npz','intervention_latents.npz','LATENT_BANKS.json','REPLAY_IDENTITY_ORDER.json',
        'ACTUAL_COUNTERS.json','GENERATION.json','DIAGNOSTIC_STATISTICS.json'}
    required_files|={family+'_'+a+'_'+h+'.npz' for family in ('REPEAT','SHAPE','Q_TREND')
        for a in ('deterministic','full','simple') for h in ('fast','slow','common_level')}
    required_files|={family+'_COMMON.npz' for family in ('REPEAT','SHAPE','Q_TREND')}
    required_files|={'POPULATION_'+split+'.'+suffix for split in ('TRAIN','DEV') for suffix in ('json','npz')}
    require(set(summary['artifact_sha256'])==required_files, 'Diagnostic declared artifact inventory differs')
    expected=set(summary['artifact_sha256'])|{'SUMMARY.json'}
    require({p.name for p in destination.iterdir()}==expected,'Diagnostic artifact membership differs')
    require(all(p.is_file() and not p.is_symlink() for p in destination.iterdir()),'Diagnostic artifact is not regular')
    verify_files(destination,summary['artifact_sha256'])
    return summary


def completed_job(kind,root=ROOT):
    root=Path(root).resolve();evaluation_binding(root)
    paths=paths_for(kind,root);job=paths['job']
    require(not (job/'JOB_FAILED.json').exists() and not (job/'LAUNCH_FAILURE.json').exists(), 'Diagnostic failed attempt preserved')
    native=read(paths['supervisor']/'exit_receipt.json');launch=read(paths['supervisor']/'launch.json')
    require(native['exit_code']==0 and native.get('reason') is None,'Diagnostic native success missing')
    require(all(native[k]==launch[k] and type(native[k]) is int for k in ('supervisor_pid','child_pid')), 'Diagnostic native PID chain differs')
    command=launch['command']
    require(len(command)==5 and command[1:]==['-B',str(root/'scripts/run_window008_resume01_diagnostics.py'),'--kind','primary'], 'Diagnostic native command differs')
    core=read(job/'EVALUATOR_NATIVE.json');complete=read(job/'JOB_COMPLETE.json')
    require(core['exit_code']==0 and core['module']=='n01.diagnose_window008_resume01' and core['kind']=='primary'
        and core['command']==core_command(root,command[0]) and core['wrapper_pid']==native['child_pid']
        and core['supervisor_pid']==native['supervisor_pid']
        and core['diagnostics_source_manifest_sha256']==sha(root/EVAL_MANIFEST), 'Diagnostic core native binding differs')
    require(complete['status']=='diagnostics_job_complete' and complete['kind']=='primary'
        and complete['diagnostics_source_manifest_sha256']==sha(root/EVAL_MANIFEST)
        and complete['training_source_manifest_sha256']==sha(root/TRAIN_MANIFEST)
        and complete['optimizer_updates']==0 and complete['evaluator_native_sha256']==sha(job/'EVALUATOR_NATIVE.json'),
        'Diagnostic job receipt differs')
    summary=artifact_integrity(paths['artifacts'])
    require(complete['summary_sha256']==sha(paths['artifacts']/'SUMMARY.json'),'Diagnostic summary changed')
    return dict(paths=paths,summary=summary,complete=complete,native=native,core=core)
