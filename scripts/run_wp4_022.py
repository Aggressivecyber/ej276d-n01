"""One fixedT2 hard-observation contrast, with independent cross-bank U statistics."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from wp1_ops import clean, read_json, require, sha, unchanged, write_new
from run_wp4_017 import seed017
from run_wp4_019 import settings019, budget_caps019, SCIENCE_SOURCES019, verify019 as verify022

RUN_ID = 'wp4_022_20261016'
REMOTE = '/home/rylan/projects/ej276d-n01/wp4_022_20261016'
HYPOTHESES = ('fast', 'slow', 'common_level')
GROUPS = ('all_heldout', 'fit_Q_tertile0_heldout', 'fit_Q_tertile1_heldout', 'fit_Q_tertile2_heldout')
WEIGHTS = ('nominal', 'low', 'high')
SCIENCE_SOURCES022 = SCIENCE_SOURCES019 + ('scripts/run_wp4_022.py', 'scripts/mechanism012_losses.py')


def settings022():
    p = settings019()
    p.update(schema='n01/wp4-022-config/v1', run_id=RUN_ID, remote_root=REMOTE, status='draft',
        trainable_parameters=0, candidates=[], seed_sets=[], trial_order=[], G_updates=0,
        teachers=dict(T1=dict(a_fraction_qref=.01,b=.0025), T2=dict(m=.9,component=0)),
        shared_output_MiB=2048, failure_checkpoint_job_cap=0,
        teacher_and_initial_evaluation_reused=False, T1_saved_evaluation_reused=True,
        initial_evaluation_used=False, final_evaluation_kernels=0, new_teacher_kernels=1,
        T2_enabled=True, training_enabled=False, T2_training_authorized_by022=False,
        autonomous_engineering_recovery_authorized=True,
        required_primary_cells=3, reported_cells=36, all3_primary_hypotheses_required=True,
        new_observable_energy_tolerance=.10, new_observable_MC_half_width_limit=.05,
        teacher32_null_abs_upper95_limit=.05,
        cross_pairs=['T2_bank0_vs_T1_bank1', 'T2_bank1_vs_T1_bank0'],
        cross_directions_averaged_per_parent_before_SE=True, pooled64_forbidden=True,
        Gaussian_namespace_shared_with_T1_same_bank=True,
        stream_addresses=dict(evaluation_teacher={str(bank): dict(namespace=dict(kind='evaluation',role='teacher',bank=bank),
            seeds={stream: seed017(dict(kind='evaluation',role='teacher',bank=bank,stream=stream))
                for stream in ('shape','charge','noise005','extension284','sign','PIT')}) for bank in (0,1)},
            incidental_coverage=dict(namespace=dict(kind='coverage',teacher='T2'),
                PIT_seed=seed017(dict(kind='coverage',teacher='T2',stream='PIT')))),
        full257_descriptive_only=True, statistical_power_or_simultaneous_calibration=False)
    for key in ('required_cells_per_trial','required_cells_total'):
        del p[key]
    return p


def budget_caps022(p):
    n, r = 6593, 32
    rows = n*2*r
    caps = {name: 0 for name in budget_caps019(settings019())}
    caps.update(checkpoint_loads=1, noise005_checkpoint_loads=1, cached_array_mmap_files=3,
        teacher_charge_module_rows=rows, shape_cached_basis_assembly_rows=rows,
        noise005_rows=rows, all_three_extensions_rows=3*rows, hard_ADC_rows=3*rows,
        hard_first_argmax_gather_rows=3*rows, evaluation_signal_rows=rows,
        evaluation_kernel_snapshots=1, new_teacher_kernels=1,
        normal_shape_scalar_requests=8*rows, normal_charge_scalar_requests=rows,
        normal_noise005_scalar_requests=101*rows, normal_extension284_scalar_requests=284*rows,
        total_scalar_normal_requests=394*rows, T2_teacher_sign_uniform_requests=rows,
        coverage_PIT_uniform_requests=n*2*r*6*3,
        reused_teacher_kernel_sets=1, evaluation_energy_candidate_comparisons=1,
        evaluation_energy_pair_distances=4*n*r*r*3,
        evaluation_observable_pair_distances=7*n*r*r*3, new_pair_distances_total=11*n*r*r*3)
    return caps


def validate_settings022(p, *, frozen=True, driver_source=None):
    reference = settings022()
    require(not frozen or p['status']=='frozen', '022 draft cannot execute')
    require(all(p.get(k)==v for k,v in reference.items() if k!='status'), '022 frozen settings differ')
    require(p['budget_caps']==budget_caps022(p), '022 exact additive budget differs')


def gate022(config, expected_sha, output):
    cp=Path(config).resolve()
    require(sha(cp)==expected_sha, '022 frozen config SHA differs')
    p=read_json(cp);validate_settings022(p)
    require(set(SCIENCE_SOURCES022)<=set(p['source_files']), '022 actual mandatory source path missing')
    require(all(Path(p['source_files'][name]['path'])==Path(REMOTE)/name for name in SCIENCE_SOURCES022),
        '022 actual imported source is not owned/bound')
    require(Path(output).absolute()==Path(REMOTE)/'runs/primary/artifacts' and not Path(output).exists(),
        '022 exact fresh output required')
    mf=cp.parent/p['saved_input_manifest']
    require(sha(mf)==p['saved_input_manifest_sha256'], '022 saved input manifest SHA differs')
    manifest=read_json(mf)
    require(manifest['file_count']==27 and p['inputs']==manifest['entries'], '022 exact27 input whitelist differs')
    records={**p['source_files'],**{'input:'+k:v for k,v in p['inputs'].items()},
        **{'contract:'+k:v for k,v in p['contracts'].items()},**{'protected:'+k:v for k,v in p['protection_files'].items()}}
    checked={str(cp):expected_sha,str(mf):sha(mf)};roots=[Path(x).resolve() for x in p['allowed_input_roots']]
    for label,row in records.items():
        path=Path(row['path'])
        require(not path.is_symlink() and any(path.resolve().is_relative_to(r) for r in roots), '022 bound path escaped: '+label)
        require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], '022 bound bytes differ: '+label)
        checked[str(path)]=row['sha256']
    contract018=read_json(p['contracts']['qualification018']['path'])
    require(contract018['schema']=='n01/wp4-018-final-result/v1' and contract018['qualified'] is True and
        contract018['native_exit_code']==0 and contract018['independent_review_pass'] is True and
        contract018['old017_pending_preserved'] is True, '022 accepted018 contract required')
    contract021=read_json(p['contracts']['result021']['path'])
    require(contract021['schema']=='n01/wp5-021-final-result/v1' and contract021['native_exit_code']==0 and
        contract021['analytic_diagnostic_complete'] is True and contract021['independent_review_pass'] is True,
        '022 independently completed021 prerequisite required')
    auth=read_json(p['contracts']['user_authorization']['path'])
    require(auth['persistent_for_N01_master_plan'] is True and auth['resume_authorized'] is True and
        auth['necessary_additional_NN_AD_G_RNG_computation_authorized'] is True, '022 direct persistent authorization required')
    scope=read_json(p['contracts']['prospective_scope']['path'])
    require(scope['schema']=='n01/wp4-022-prospective-scope/v1' and scope['run_id']==RUN_ID, '022 registered scope differs')
    original018=read_json(p['inputs']['config018']['path'])
    require(original018['scales']==p['scales'] and original018['new_observable_energy_tolerance']==.1 and
        original018['new_observable_MC_half_width_limit']==.05 and original018['time_scale_software_ns']==1.,
        '022 original018 observable scales/tolerances differ')
    return p,{k:Path(v['path']) for k,v in p['inputs'].items()},checked



def write_generated022(path, value, *, output, store):
    """Version only newly generated metadata; numerical algorithms/namespaces stay017."""
    require(Path(path).resolve().is_relative_to(output.resolve()), '022 metadata write escaped new output')
    schema = value['schema'] if 'schema' in value else None
    if schema is not None and 'wp4-017' in schema:
        value['implementation_schema_origin'] = schema
        value['schema'] = schema.replace('wp4-017', 'wp4-022')
    value['run_id'] = RUN_ID
    if Path(path).name == 'EVALUATION_AXES.json':
        value['source_index_root'] = store.config['reused_evaluation_root017']
    write_new(path, value)



def crossbank_scores022(T2_wave, T1_wave, T2_observable, T1_observable, reference257):
    """Two32-bank contrasts paired by parent; never pool64 or reverse parents."""
    import numpy as np
    from run_wp4_018 import wave_energy018
    from n01.stats018 import self_mean018, pair_distances018
    require(T2_wave.shape==T1_wave.shape and T2_wave.ndim==4 and T2_wave.shape[1]==2,
        '022 parent/bank/draw/wave axes')
    require(T2_observable.shape==T1_observable.shape and T2_observable.shape[:3]==T2_wave.shape[:3]
        and T2_observable.shape[-1]==19, '022 matching observable axes')
    ss=[self_mean018(T2_observable[:,b]) for b in (0,1)]
    tt=[self_mean018(T1_observable[:,b]) for b in (0,1)]
    cross01=pair_distances018(T2_observable[:,0],T1_observable[:,1]).mean((1,2))
    cross10=pair_distances018(T2_observable[:,1],T1_observable[:,0]).mean((1,2))
    d01=cross01-.5*(ss[0]+tt[1]);d10=cross10-.5*(ss[1]+tt[0])
    T2_null_cross=pair_distances018(T2_observable[:,0],T2_observable[:,1]).mean((1,2))
    return dict(cross_parent=.5*(d01+d10), cross_direction01=d01, cross_direction10=d10,
        T2_null=T2_null_cross-.5*(ss[0]+ss[1]),
        T2_self0=ss[0],T2_self1=ss[1],T1_self0=tt[0],T1_self1=tt[1],
        cross_distance01=cross01,cross_distance10=cross10,T2_null_cross_distance=T2_null_cross,
        full257_excess=wave_energy018(T2_wave[:,::-1],T1_wave)-reference257)


def contrast_cell022(scores, T1_null, group, weights):
    import numpy as np
    from n01.stats017 import support_cell017, weighted017, Z95
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


def teacher_data022(paths,h):
    import numpy as np
    from run_wp4_017 import decode_hard017
    waves=np.stack([decode_hard017(paths[f'teacher_{h}_hard{b}'])[0].reshape(6593,32,256) for b in (0,1)],axis=1)
    with np.load(paths[f'observable018_{h}'],allow_pickle=False) as ar:embedding=ar['embedding']
    with np.load(paths[f'control018_{h}'],allow_pickle=False) as ar:
        null,reference=ar['teacher_null'],ar['fullwave_teacher_reference']
    require(embedding.shape==(6593,2,32,19) and null.shape==reference.shape==(6593,), '022 savedT1 axes differ')
    return waves,embedding,null,reference



def run022(config, config_sha256, output, device):
    p, paths, checked = gate022(config, config_sha256, output)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(p['CPU_threads'])
    import functools
    import numpy as np
    # The018 import exposes its existing pure numerical helpers and namespace;
    # no private loading occurs merely by importing it.
    from run_wp4_018 import load_kernel018
    import run_wp4_017 as legacy
    from n01.stats018 import DISTANCE_COUNTS018
    from n01.stats017 import weighted017
    import torch
    from n01.model017 import deterministic_loading017
    from n01.response008 import make_extension, state_digest
    from n01.evaluate_window008 import _memory_limit
    torch.set_num_threads(p['CPU_threads']); torch.use_deterministic_algorithms(True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    _memory_limit(dict(gpu_memory_limit_GiB=p['resource_limits']['GPU_GiB']), device)
    store = legacy.Store017(output, p, checked)
    legacy.write_new = functools.partial(write_generated022, output=store.root, store=store)
    store.status()
    try:
        store.json('BINDING.json', dict(store.base(), schema='n01/wp4-022-binding/v1',
            configuration_sha256=config_sha256, immutable_files=checked, budget_caps=p['budget_caps'], before_numeric_decode=True))
        cache_manifest = read_json(paths['cache_manifest'])
        prior = read_json(paths['prior_frozen'])['frozen_state_sha256']
        with np.load(paths['parent_split'], allow_pickle=False) as ar:
            cache = dict(fit=ar['fit_rows'], test=ar['test_rows'], ids=ar['source_ids'],
                source_features={k: ar[k] for k in ('q_source', 'source_peak_index', 'source_baseline_mean_V')},
                directory=paths['cache_manifest'].parent, manifest=cache_manifest)
        for name in ('A', 'B', 'Q'):
            cache[name] = store.call('read_only_cached_'+name, dict(cached_array_mmap_files=1),
                lambda name=name: np.load(paths['cache_'+name], mmap_mode='r', allow_pickle=False))
            require(not cache[name].flags.writeable, '022 requires read-only017 cache')
        require(len(cache['Q']) == 6593 and len(cache['fit']) == 5274 and len(cache['test']) == 1319,
                '022 original parent axes differ')
        with np.load(paths['physical_weights'], allow_pickle=False) as ar:
            np.testing.assert_array_equal(ar['source_ids'], cache['ids'])
            np.savez_compressed(store.root/'ORIGINAL_PHYSICAL_WEIGHTS.npz', weights=ar['weights'], source_ids=cache['ids'])
        with np.load(paths['parent_axes018'], allow_pickle=False) as ar:
            for key, value in (('source_ids', cache['ids']), ('fit_rows', cache['fit']), ('test_rows', cache['test']), ('fixed_A_Q_Vns', cache['Q'])):
                np.testing.assert_array_equal(ar[key], value)
        stats = read_json(paths['statistics'])
        noise = store.call('only_frozen005_checkpoint', dict(checkpoint_loads=1, noise005_checkpoint_loads=1),
            lambda: torch.load(paths['noise005'], map_location='cpu', weights_only=False))
        require(noise['schema_version'] == 5 and noise['update'] == 5000 and noise['model_type'] == 'ObservedBaselineNoiseV3'
            and noise['engineering'] is False, '022 original005 checkpoint metadata')
        with deterministic_loading017():
            extension = make_extension(noise, device)
        del noise
        require(state_digest(extension.model005.state_dict()) == prior['noise005'] and
                state_digest(extension.state_dict()) == prior['extension'], '022 original005 frozen state differs')
        rt = legacy.Runtime017(p, cache, extension, stats, prior, store, device)
        rt.verify = functools.partial(verify022, rt)
        store.call('reuse_T1_saved_evaluation', dict(reused_teacher_kernel_sets=1), lambda: None)
        rt.guard=lambda directory,started,**unused:legacy.Runtime017.guard(rt,directory,started,shared=False)
        teacher_meta=read_json(paths['teacher_eval'])
        require(teacher_meta['teacher']=='T1' and teacher_meta['a_Vns']==.01*p['qref'] and teacher_meta['b']==.0025,
            '022 originalT1 coefficients differ')
        B0=np.ascontiguousarray(cache['B'][:,0,:])
        B0_sha256=hashlib.sha256(B0.tobytes()).hexdigest();del B0
        store.json('INIT_PROOF.json',dict(store.base(),schema='n01/wp4-022-initialized/v1',run_id=RUN_ID,
            native_initialized=True,checkpoint_loads=1,noise005_checkpoint_loads=1,A_checkpoint_loads=0,M0_checkpoint_loads=0,
            trainable_parameters=0,training_enabled=False,new_teacher='T2',
            mechanism=dict(eta0='.9*sign+sqrt(1-.9**2)*z0',mixture_m=.9,component=0,other7_standard_normal=True),
            a_Vns=.01*p['qref'],b=.0025,cache_manifest_sha256=sha(paths['cache_manifest']),
            cached_B_file_sha256=p['inputs']['cache_B']['sha256'],B0_ordered_float64_sha256=B0_sha256,
            B0_shape=[6593,384],B0_scope=cache_manifest['T2_fixed_B0_scope'],frozen_state_sha256=prior,
            stream_addresses=p['stream_addresses'],same_bank_T1_T2_Gaussian_prefix_shared=True,
            cross_bank_pairing=p['cross_pairs'],PIT_draws_incidental_not_used_for_contrast=7595136,
            new_SOURCE_REAL_DEV_original_waveform_reads=0,T2_training_started=False))
        kernel_started=time.monotonic()
        legacy.evaluate_kernel017(rt,store.root/'T2',teacher='T2')
        store.call('one_registered_T2_teacher_kernel',dict(new_teacher_kernels=1),lambda:None)
        kernel_resources=rt.guard(store.root/'T2',kernel_started)
        rt.verify(files=True)
        DISTANCE_COUNTS018.clear()
        with np.load(paths['parent_axes018'],allow_pickle=False) as ar:
            groups={name:ar['group_'+name] for name in GROUPS}
            weights={name:ar['comparison_weight_'+name] for name in WEIGHTS}
        cells=[];test=cache['test'];ids=cache['ids']
        for h in HYPOTHESES:
            S=load_kernel018(Path(REMOTE),'T2',h,ids,p,store.root,store.started,'T2')
            T_wave,T_observable,T1_null,reference=teacher_data022(paths,h)
            scores=store.call('independent_crossbank_T2_T1_scores',
                dict(evaluation_energy_pair_distances=4*6593*32**2,
                    evaluation_observable_pair_distances=7*6593*32**2,new_pair_distances_total=11*6593*32**2),
                lambda:crossbank_scores022(S['waves'],T_wave,S['embedding'],T_observable,reference))
            np.savez_compressed(store.root/f'PARENT_SCORES022_{h}.npz',source_ids=ids,heldout_source_rows=test,
                reused_T1_null=T1_null,reused_T1_full257_reference=reference,**scores)
            selected={name:value[test] for name,value in scores.items()}
            for group in GROUPS:
                for weight in WEIGHTS:
                    cells.append(dict(contrast_cell022(selected,T1_null[test],groups[group],weights[weight]),
                        hypothesis=h,group=group,weights=weight,
                        primary=group=='all_heldout' and weight=='nominal'))
            del S,T_wave,T_observable,scores,selected
            store.status(dict(phase='scored_'+h))
        require(DISTANCE_COUNTS018=={257:81014784,19:141775872}, '022 exact distance kernel counts differ')
        store.call('one_T2_T1_complete_descriptive_comparison',dict(evaluation_energy_candidate_comparisons=1),lambda:None)
        primary=[c for c in cells if c['primary']]
        require(len(cells)==36 and len(primary)==3 and {c['hypothesis'] for c in primary}==set(HYPOTHESES),
            '022 full36/primary3 signature differs')
        statuses=[c['status'] for c in primary]
        supported=all(c['supported_at_registered_tolerance'] for c in primary)
        if supported:decision='supported_at_registered_tolerance_all3'
        elif 'MC_unresolved' in statuses:decision='MC_or_support_unresolved'
        elif 'tolerance_straddling_unresolved' in statuses:decision='tolerance_straddling_unresolved'
        elif 'supported_at_registered_tolerance' in statuses:decision='electronics_hypothesis_sensitive'
        else:decision='not_established_at_registered_tolerance'
        report=dict(schema='n01/wp4-022-hard-contrast/v1',run_id=RUN_ID,cells=cells,primary_cells=primary,
            decision=decision,all3_primary_supported=supported,
            parent_ids=ids,heldout_source_rows=test,groups=groups,comparison_weights=weights,
            T1_T2_same_bank_Gaussian_prefix_shared=True,independent_crossbank_pairs=p['cross_pairs'],
            pooled64_used=False,bank_directions_are_not_independent_replicates=True,
            O19_U_negative_values_preserved=True,full257_descriptive_only=True,
            statistical_power_or_simultaneous_test_calibration=False,
            non_detection_does_not_establish_mechanism_absence=True,T2_training_started=False,
            old017_pending_preserved=True,old020_nonpass_preserved=True,scientific_acceptance=False)
        legacy.compact_report017(store.root/'CONTRAST022.json',report)
        kernel_resources=rt.guard(store.root/'T2',kernel_started)
        rt.verify(files=True)
        require(not store.inflight and store.counts==store.attempted==p['budget_caps'], '022 exact additive counters differ')
        shared_bytes=sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file() and q.relative_to(store.root).parts[0]!='T2')
        require(shared_bytes<=2048*1024**2, '022 shared output exceeds2048MiB')
        files={q.relative_to(store.root).as_posix():dict(bytes=q.stat().st_size,sha256=sha(q))
            for q in sorted(store.root.rglob('*')) if q.is_file() and q.name!='status.json'}
        store.json('ARTIFACT_INVENTORY.json',dict(schema='n01/wp4-022-artifact-inventory/v1',run_id=RUN_ID,files=files,
            full_arrays_retained_remote=True))
        result=dict(store.base(),schema='n01/wp4-022-complete/v1',native_status='COMPLETE',decision=decision,
            all3_primary_supported=supported,reported_cells=36,required_primary_cells=3,
            kernel_resources=kernel_resources,generated_teacher_kernels=1,T2_training_started=False,
            new_G_AD=0,old017_pending_preserved=True,old020_nonpass_preserved=True,
            full257_descriptive_only=True,scientific_acceptance=False)
        store.json('COMPLETE.json',result);store.status(dict(native_status='COMPLETE'))
        require(sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file())<=5120*1024**2, '022 whole output cap')
        return result
    except BaseException as error:
        failure=dict(store.base(),schema='n01/wp4-022-failure/v1',native_status='FAILED',
            error_type=type(error).__name__,error=str(error),automatic_restart=False,
            T2_training_started=False,checkpoint_save_attempts=0)
        store.json('FAILED.json',failure);store.status(failure)
        raise



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    def terminate(signum, frame):
        raise InterruptedError('Owned022 received signal '+str(signum))
    signal.signal(signal.SIGTERM, terminate)
    result = run022(args.config, args.config_sha256, args.output, args.device)
    print(json.dumps(dict(native_status=result['native_status'], G_updates=result['counters']['G_updates'],
        contrast_decision=result['decision']), allow_nan=False), flush=True)



if __name__ == '__main__':
    main()
