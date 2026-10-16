"""CPU-only completion of the frozen022 contrast from its saved generated arrays."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

from wp1_ops import Store, atomic_json, clean, now, read_json, require, sha, unchanged
from run_wp4_022 import HYPOTHESES, GROUPS, WEIGHTS

RUN_ID = 'wp4_023_20261016'
REMOTE = '/home/rylan/projects/ej276d-n01/wp4_023_20261016'
SAVED022 = '/home/rylan/projects/ej276d-n01/wp4_022_20261016'
REUSED_SOURCES023 = {
    'scripts/run_wp4_022.py': (23831, '93b7f09fe44b0aa3ccc0689cf107fdb8e796c1faec597604dc960ee9f2c17a8d'),
    'scripts/run_wp4_019.py': (31058, 'd76661d3004a326cbff06798318000be441e10734cdba94302bc5e767e5290f9'),
    'scripts/run_wp4_018.py': (28128, '4903798a66adef3e664103d94b604191c8180e9ca6d52cd56088da9e7259a0bb'),
    'scripts/run_wp4_017.py': (63149, '34e8604b7be53359f8b50ffc4de882845d561f49190063931614e25ec6bc5e62'),
    'scripts/wp1_ops.py': (12872, 'f8ddbbdb2f2f6df033e964f27bf03dcbda07055cc7378775591ef69b9caf85b5'),
    'src/n01/stats017.py': (21447, '9dc33275b66257397e2382cd2989f74701026a897e847af76fa459c72859dd0a'),
    'src/n01/stats018.py': (12775, '7a5e31ef5c8c7074a2cd826849c76ebbf485168f5cecc0d64e16fba343767cc8'),
    'src/n01/saved_array_kernels017.py': (3691, 'fe1366f35f36e97974f1a4037a36ad858c03dd8453957c1b45b65080f0fee3ac')}
SCIENCE_SOURCES023 = ('scripts/run_wp4_023.py', *REUSED_SOURCES023)


def settings023():
    return dict(schema='n01/wp4-023-config/v1', run_id=RUN_ID, status='draft', remote_root=REMOTE,
        remote_output=REMOTE+'/runs/primary/artifacts', saved_root022=SAVED022,
        saved_input_manifest='WP4_023_SAVED_INPUTS.json',
        saved_input_manifest_sha256='0153109069d04cd76737f0178f36db91519fdd04adcf087eeac7d5f54a9a4a57',
        saved_input_file_count=23, saved_input_bytes=1000564887,
        parents=6593, fit_parents=5274, heldout_parents=1319, banks=2, draws_per_bank=32,
        hypotheses=list(HYPOTHESES), new_distance_hypotheses=['slow', 'common_level'],
        groups=list(GROUPS), weights=list(WEIGHTS), required_primary_cells=3, reported_cells=36,
        new_observable_energy_tolerance=.1, new_observable_MC_half_width_limit=.05,
        teacher32_null_abs_upper95_limit=.05, time_scale_software_ns=1.,
        scales=dict(fast=dict(Q_Vns=.14006821393078955,peak_voltage_V=.005499862543299458,FWHM_ns=.5129398878764326),
            slow=dict(Q_Vns=.1400669323527055,peak_voltage_V=.005499869994334829,FWHM_ns=.5149414509583063),
            common_level=dict(Q_Vns=.14006769980934886,peak_voltage_V=.00550009352539599,FWHM_ns=.5147888093676196)),
        CPU_threads=12, resource_limits=dict(wall_seconds=7200, RSS_GiB=8, output_MiB=1536),
        resource_caps=dict(CPU_threads=12,address_space_GiB=8,wall_seconds=7200,
            CPU_seconds_soft=21590,CPU_seconds_hard=21600,output_MiB=1536,selected_collection_MiB=128),
        allowed_input_roots=[REMOTE,SAVED022,'/home/rylan/projects/ej276d-n01/wp4_018_20261016',
            '/home/rylan/projects/ej276d-n01/wp4_017_20261015'],
        contracts={
            'qualification018':dict(path=REMOTE+'/receipts/wp4_018_20261016/FINAL_RESULT_RECEIPT.json',bytes=6437,
                sha256='bbbed5f554f6de4548a5131d6df68445c160d01c001a36c3d9d2d918be748bfe'),
            'result022':dict(path=REMOTE+'/receipts/wp4_022_20261016/FINAL_RESULT_RECEIPT.json',bytes=7462,
                sha256='f83ab370385b6e6946aaf6e98bd01bfb9d27653798c337d9d85018f6ccab3452'),
            'user_authorization':dict(path=REMOTE+'/receipts/wp4_020_20261016/authorization_01/USER_AUTHORIZATION.json',bytes=1874,
                sha256='d1dcdd2002cf52c792aa307e14cc11fb3b1ad25d035ff0cd46a8bb84cc0040fa'),
            'prospective_scope':dict(path=REMOTE+'/receipts/wp4_023_20261016/registration_01/PROSPECTIVE_SCOPE.json',bytes=6175,
                sha256='d547c1d6b0f926c51d8f29188329c6ee7c4da2351e8049277df733e62cf272a7')},
        fast_distance_recomputation=0, CPU_only=True, T2_training_started=False,
        full257_descriptive_only=True, automatic_restart=False,
        autonomous_registered_engineering_recovery_authorized=True)


def budget_caps023(p):
    caps = dict.fromkeys(('new_teacher_kernels','NN_forward_rows','noise005_rows','checkpoint_loads',
        'new_AD_queries','G_updates','optimizer_steps','new_RNG_requests','new_normal_requests',
        'new_sign_uniform_requests','new_PIT_uniform_requests','new_hard_ADC_or_argmax_rows',
        'original_SOURCE_REAL_DEV_waveform_reads','parameter_fits','training_trial_slots'), 0)
    caps.update(reused_fast_parent_score_rows=6593,new_parent_score_rows=13186,
        reported_group_weight_hypothesis_cells=36,required_primary_cells=3,
        saved_T2_hard_rows_decoded=843904,saved_T1_hard_rows_decoded=843904,total_saved_hard_rows_decoded=1687808,
        evaluation_energy_pair_distances=54009856,evaluation_observable_pair_distances=94517248,
        new_pair_distances_total=148527104)
    return caps


def validate_settings023(p, *, frozen=True, driver_source=None):
    require(not frozen or p['status']=='frozen', '023 draft cannot execute')
    require(all(p.get(k)==v for k,v in settings023().items() if k!='status'), '023 frozen settings differ')
    require(p['budget_caps']==budget_caps023(p), '023 exact additive budget differs')


def gate023(config, expected_sha, output):
    cp=Path(config).resolve()
    require(sha(cp)==expected_sha, '023 frozen config SHA differs')
    p=read_json(cp);validate_settings023(p)
    require(set(SCIENCE_SOURCES023)<=set(p['source_files']), '023 mandatory CPU source missing')
    for name in SCIENCE_SOURCES023:
        row=p['source_files'][name]
        require(Path(row['path'])==Path(REMOTE)/name, '023 imported source must be owned/bound')
        if name in REUSED_SOURCES023:
            require((row['bytes'],row['sha256'])==REUSED_SOURCES023[name], '023 reused source changed: '+name)
    require(Path(output).absolute()==Path(p['remote_output']) and not Path(output).exists(), '023 exact fresh output required')
    mf=cp.parent/p['saved_input_manifest']
    require(sha(mf)==p['saved_input_manifest_sha256'], '023 input manifest differs')
    manifest=read_json(mf)
    require(manifest['schema']=='n01/wp4-023-saved-inputs/v1' and manifest['run_id']==RUN_ID and
        manifest['file_count']==23 and manifest['total_frozen_input_bytes']==1000564887 and
        p['inputs']==manifest['entries'], '023 exact saved input whitelist differs')
    records={**p['source_files'],**{'input:'+k:v for k,v in p['inputs'].items()},
        **{'contract:'+k:v for k,v in p['contracts'].items()}}
    checked={str(cp):expected_sha,str(mf):sha(mf)};roots=[Path(x).resolve() for x in p['allowed_input_roots']]
    for name,row in records.items():
        path=Path(row['path'])
        require(not path.is_symlink() and any(path.resolve().is_relative_to(r) for r in roots), '023 input path escaped: '+name)
        require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], '023 bound bytes differ: '+name)
        checked[str(path)]=row['sha256']
    q=read_json(p['contracts']['qualification018']['path'])
    require(q['schema']=='n01/wp4-018-final-result/v1' and q['qualified'] is True and
        q['native_exit_code']==0 and q['independent_review_pass'] is True, '023 accepted018 required')
    failed=read_json(p['contracts']['result022']['path'])
    require(failed['schema']=='n01/wp4-022-final-result/v1' and failed['native_exit_code']==1 and
        failed['T2_kernel_generation_complete'] is True and failed['remaining_total_distances']==148527104 and
        failed['fast_parent_scores_saved']['sha256']==p['inputs']['fast_scores022']['sha256'], '023 saved022 completion facts differ')
    auth=read_json(p['contracts']['user_authorization']['path'])
    require(auth['persistent_for_N01_master_plan'] is True and auth['resume_authorized'] is True and
        auth['necessary_additional_NN_AD_G_RNG_computation_authorized'] is True, '023 persistent authorization required')
    scope=read_json(p['contracts']['prospective_scope']['path'])
    require(scope['schema']=='n01/wp4-023-prospective-scope/v1' and scope['run_id']==RUN_ID and
        scope['budget_caps']==p['budget_caps'], '023 registered scope differs')
    require(read_json(p['inputs']['config018']['path'])['scales']==p['scales'], '023 frozen018 scales differ')
    old=read_json(p['inputs']['source_config022']['path'])
    require(old['run_id']=='wp4_022_20261016' and old['scales']==p['scales'], '023 original022 contract differs')
    evaluation=read_json(p['inputs']['T2_evaluation022']['path'])
    require(evaluation['run_id']=='wp4_022_20261016' and evaluation['teacher']=='T2', '023 original T2 evaluation differs')
    return p,{k:Path(v['path']) for k,v in p['inputs'].items()},checked


def contrast_cell023(scores, T1_null, group, weights):
    import numpy as np
    from n01.stats017 import support_cell017, weighted017
    from n01.stats018 import Z95
    support=support_cell017(group,weights,np.ones(len(T1_null),bool))
    ix=support.pop('indices')
    if support['status']=='pending':
        return dict(support,status='MC_unresolved',supported_at_registered_tolerance=False,supported_parent_indices=ix)
    cross=weighted017(scores['cross_parent'][ix],weights[ix])
    null1=weighted017(T1_null[ix],weights[ix]);null2=weighted017(scores['T2_null'][ix],weights[ix])
    wave=weighted017(scores['full257_excess'][ix],weights[ix])
    half=Z95*cross['SE'];lower=cross['mean']-half;upper=cross['mean']+half
    null1upper=abs(null1['mean'])+Z95*null1['SE'];null2upper=abs(null2['mean'])+Z95*null2['SE']
    reasons=[]
    if half>.05:reasons.append('cross_MC_halfwidth_above005')
    if null1upper>.05:reasons.append('T1_teacher32_null_unresolved')
    if null2upper>.05:reasons.append('T2_teacher32_null_unresolved')
    if reasons:status='MC_unresolved'
    elif lower>.1:status='supported_at_registered_tolerance'
    elif upper<=.1:status='not_established_at_registered_tolerance'
    else:status='tolerance_straddling_unresolved'
    return dict(support,status=status,reasons=reasons,supported_at_registered_tolerance=status=='supported_at_registered_tolerance',
        cross_parent_cluster=cross,lower95=lower,upper95=upper,half_width95=half,
        T1_teacher32_null=null1,T2_teacher32_null=null2,T1_null_abs_upper95=null1upper,T2_null_abs_upper95=null2upper,
        full257_descriptive=dict(wave,lower95=wave['mean']-Z95*wave['SE'],upper95=wave['mean']+Z95*wave['SE'],
            no_O19_threshold_or_training_gate=True),supported_parent_indices=ix,
        original_draws_per_bank=32,directions_averaged_before_parent_SE=True,independent_direction_replicates=False,
        O19_tolerance=.1,MC_half_width_limit=.05,null_abs_upper_limit=.05)


def run023(config, config_sha256, output):
    require(os.name=='posix', '023 native CPU run requires Linux')
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(21590,21600))
    p,paths,checked=gate023(config,config_sha256,output)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]=str(p['CPU_threads'])
    # This original018 import creates the numerical n01 namespace without n01/__init__.
    from run_wp4_018 import load_kernel018
    import numpy as np
    from n01.stats018 import DISTANCE_COUNTS018
    from run_wp4_022 import crossbank_scores022, teacher_data022
    from run_wp4_017 import compact_report017
    require('torch' not in sys.modules, '023 pure NumPy path imported torch')
    DISTANCE_COUNTS018.clear()
    store=Store(output,p,checked);store.status(dict(native_status='STARTED'))
    try:
        store.json('STARTED.json',dict(store.base(),schema='n01/wp4-023-started/v1',native_status='STARTED',
            config_sha256=config_sha256,manifest_verified=True,input_files=23,NN_CP_AD_G_RNG_ADC=0,
            source_run_id='wp4_022_20261016',original022_native_exit_code=1,fast_distances_recomputed=0))
        with np.load(paths['parent_axes018'],allow_pickle=False) as ar:
            ids,fit,test=ar['source_ids'],ar['fit_rows'],ar['test_rows']
            groups={name:ar['group_'+name] for name in GROUPS}
            weights={name:ar['comparison_weight_'+name] for name in WEIGHTS}
            store.npz('PARENT_AXES.npz',**{name:ar[name] for name in ar.files})
        require(ids.shape==(6593,) and fit.shape==(5274,) and test.shape==(1319,), '023 original parent axes differ')
        cells=[]
        for h in HYPOTHESES:
            store.phase='reuse_fast_scores' if h=='fast' else 'decode_saved_'+h;store.status()
            if h=='fast':
                with np.load(paths['fast_scores022'],allow_pickle=False) as ar:
                    saved=store.call('reuse_saved_fast_parent_scores',dict(reused_fast_parent_score_rows=6593),
                        lambda:{name:ar[name] for name in ar.files})
                require(np.array_equal(saved['source_ids'],ids) and np.array_equal(saved['heldout_source_rows'],test),
                    '023 fast parent identity differs')
                T1_null=saved['reused_T1_null']
                scores={name:value for name,value in saved.items() if name not in
                    ('source_ids','heldout_source_rows','reused_T1_null','reused_T1_full257_reference')}
            else:
                S=store.call('decode_saved_T2_'+h,dict(saved_T2_hard_rows_decoded=421952,total_saved_hard_rows_decoded=421952),
                    lambda:load_kernel018(Path(SAVED022),'T2',h,ids,p,store.root,store.started,'T2'))
                T_wave,T_observable,T1_null,reference=store.call('decode_saved_T1_'+h,
                    dict(saved_T1_hard_rows_decoded=421952,total_saved_hard_rows_decoded=421952),lambda:teacher_data022(paths,h))
                store.phase='score_saved_'+h;store.status()
                scores=store.call('remaining_crossbank_scores_'+h,dict(new_parent_score_rows=6593,
                    evaluation_energy_pair_distances=27004928,evaluation_observable_pair_distances=47258624,
                    new_pair_distances_total=74263552),
                    lambda:crossbank_scores022(S['waves'],T_wave,S['embedding'],T_observable,reference))
                saved=dict(source_ids=ids,heldout_source_rows=test,reused_T1_null=T1_null,
                    reused_T1_full257_reference=reference,**scores)
                del S,T_wave,T_observable,reference
            require(T1_null.shape==(6593,) and all(v.shape==(6593,) and np.isfinite(v).all() for v in scores.values()),
                '023 finite all-parent score vectors required')
            store.npz(f'PARENT_SCORES023_{h}.npz',**saved)
            selected={name:value[test] for name,value in scores.items()}
            cells.extend(store.call('classify_'+h,dict(reported_group_weight_hypothesis_cells=12,required_primary_cells=1),
                lambda:[dict(contrast_cell023(selected,T1_null[test],groups[g],weights[w]),
                    hypothesis=h,group=g,weights=w,primary=g=='all_heldout' and w=='nominal') for g in GROUPS for w in WEIGHTS]))
            store.phase='scored_'+h;store.status()
            usage=resource.getrusage(resource.RUSAGE_SELF)
            atomic_json(store.root/'PHASE.json',dict(store.base(),schema='n01/wp4-023-phase/v1',
                completed_distance_counts=dict(DISTANCE_COUNTS018),fast_distances_recomputed=0,
                elapsed_seconds=time.monotonic()-store.started,peak_RSS_bytes=usage.ru_maxrss*1024,
                user_CPU_seconds=usage.ru_utime,system_CPU_seconds=usage.ru_stime,
                output_bytes=sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file())))
            del scores,selected,saved,T1_null
        require(DISTANCE_COUNTS018=={257:54009856,19:94517248}, '023 exact new distance counts differ')
        require({(c['hypothesis'],c['group'],c['weights']) for c in cells}==
            {(h,g,w) for h in HYPOTHESES for g in GROUPS for w in WEIGHTS} and len(cells)==36, '023 full36 signature differs')
        primary=[c for c in cells if c['primary']]
        statuses=[c['status'] for c in primary];supported=all(c['supported_at_registered_tolerance'] for c in primary)
        if supported:decision='supported_at_registered_tolerance_all3'
        elif 'MC_unresolved' in statuses:decision='MC_or_support_unresolved'
        elif 'tolerance_straddling_unresolved' in statuses:decision='tolerance_straddling_unresolved'
        elif 'supported_at_registered_tolerance' in statuses:decision='electronics_hypothesis_sensitive'
        else:decision='not_established_at_registered_tolerance'
        compact_report017(store.root/'CONTRAST023.json',dict(schema='n01/wp4-023-hard-contrast/v1',run_id=RUN_ID,
            cells=cells,primary_cells=primary,decision=decision,all3_primary_supported=supported,
            parent_ids=ids,heldout_source_rows=test,groups=groups,comparison_weights=weights,
            source_run_id='wp4_022_20261016',original022_failure_preserved=True,
            fast_parent_scores_reused=True,fast_distances_recomputed=0,parent_average_before_SE=True,
            pooled64_used=False,bank_directions_are_not_independent_replicates=True,O19_U_negative_values_preserved=True,
            full257_descriptive_only=True,statistical_power_or_simultaneous_test_calibration=False,
            non_detection_does_not_establish_mechanism_absence=True,T2_training_started=False,
            old017_pending_preserved=True,old020_nonpass_preserved=True,scientific_acceptance=False))
        unchanged(checked)
        require('torch' not in sys.modules and not store.inflight and store.counts==store.attempted==p['budget_caps'],
            '023 pure CPU exact additive counters differ')
        usage=resource.getrusage(resource.RUSAGE_SELF)
        resources=dict(elapsed_seconds=time.monotonic()-store.started,peak_RSS_bytes=usage.ru_maxrss*1024,
            user_CPU_seconds=usage.ru_utime,system_CPU_seconds=usage.ru_stime,
            output_bytes=sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file()),snapshot_before_terminal_receipts=True)
        require(resources['elapsed_seconds']<=7200 and resources['peak_RSS_bytes']<=8*1024**3 and
            resources['output_bytes']<=1536*1024**2, '023 terminal resource cap exceeded')
        files={q.relative_to(store.root).as_posix():dict(bytes=q.stat().st_size,sha256=sha(q))
            for q in sorted(store.root.rglob('*')) if q.is_file() and q.name!='status.json'}
        store.json('ARTIFACT_INVENTORY.json',dict(schema='n01/wp4-023-artifact-inventory/v1',run_id=RUN_ID,files=files))
        result=dict(store.base(),schema='n01/wp4-023-complete/v1',native_status='COMPLETE',decision=decision,
            all3_primary_supported=supported,reported_cells=36,required_primary_cells=3,resources=resources,
            new_distance_counts=dict(DISTANCE_COUNTS018),old022_completed_distances=74263552,
            old022_plus023_distances=222790656,original022_native_exit_code=1,NN_CP_AD_G_RNG_ADC=0,
            T2_training_started=False,scientific_acceptance=False)
        store.json('COMPLETE.json',result);store.status(dict(native_status='COMPLETE'))
        require(sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file())<=1536*1024**2, '023 final output cap exceeded')
        return result
    except BaseException as error:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        failure=dict(store.base(),schema='n01/wp4-023-failure/v1',native_status='FAILED',error_type=type(error).__name__,
            error=str(error),elapsed_seconds=time.monotonic()-store.started,peak_RSS_bytes=usage.ru_maxrss*1024,
            user_CPU_seconds=usage.ru_utime,system_CPU_seconds=usage.ru_stime,
            output_bytes=sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file()),
            entered_kernel_distance_counts_completion_unknown=dict(DISTANCE_COUNTS018),
            automatic_restart=False,original022_failure_preserved=True,T2_training_started=False)
        store.json('FAILED.json',failure);store.status(failure)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    def terminate(signum, frame):
        raise InterruptedError('Owned023 received signal '+str(signum))
    signal.signal(signal.SIGTERM,terminate)
    result=run023(args.config,args.config_sha256,args.output)
    print(json.dumps(dict(native_status=result['native_status'],decision=result['decision'],new_G=0),allow_nan=False),flush=True)


if __name__=='__main__':
    main()
