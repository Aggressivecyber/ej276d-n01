"""Independent deterministic fixtures for the same WP4-017 source review.

Only source/JSON and manufactured NumPy arrays are read. No torch, source
dataset, checkpoint, scientific job or server operation is used here.
"""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import types

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
TMP = HERE/'tmp'
TMP.mkdir(exist_ok=True)
for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR'):
    os.environ[key] = str(TMP)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.dont_write_bytecode = True
import numpy as np


def record(path):
    value = path.read_bytes()
    return dict(sha256=hashlib.sha256(value).hexdigest(),bytes=len(value))


def load_stats():
    package = types.ModuleType('_wp4_review_final_numpy')
    package.__path__ = [str(ROOT/'src/n01')]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__+'.stats017',ROOT/'src/n01/stats017.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pure_factories():
    tree = ast.parse((ROOT/'scripts/run_wp4_017.py').read_bytes())
    names = {'settings017','budget_caps017','validate_settings017'}
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for node in nodes for n in ast.walk(node))
    def require(value,message):
        if not value:
            raise ValueError(message)
    scope = dict(RUN_ID='wp4_017_20261015',REMOTE='/home/rylan/projects/ej276d-n01/wp4_017_20261015',
        PROPOSAL_SHA='b2db64dfd3549f6cd22cb5253a1e374738248012e22554308ec26885d2d42e8e',
        HYPOTHESES=('fast','slow','common_level'),TRIALS=('T1_paired_seed0','T1_paired_seed1',
        'T1_unpaired_seed0','T1_unpaired_seed1','T2_paired_seed0','T2_paired_seed1','T2_unpaired_seed0','T2_unpaired_seed1'),
        require=require)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<independent-factories>','exec'),scope)
    return scope


def primary_and_missing_fixture():
    # Literal accepted axes and causes, deliberately independent of the module
    # constants so an accidental omitted cause is not reproduced in this oracle.
    groups=('all_heldout','fit_Q_tertile0_heldout','fit_Q_tertile1_heldout','fit_Q_tertile2_heldout')
    features=('Q','tail','PSD_raw','baseline_mean','baseline_RMS','peak_voltage','rise10_90','FWHM','peak_index')
    required=tuple('undefined_'+f for f in features)+(
        'flag__PSD_undefined','flag__rise_undefined','flag__FWHM_undefined','flag__temporal_nonpositive_peak',
        'flag__rise_prepeak_censored','flag__rise_other_undefined','flag__FWHM_prepeak_censored',
        'flag__FWHM_postpeak_censored','flag__FWHM_other_undefined')
    cells=[]
    for g in groups:
        for w in ('nominal','low','high'):
            cells.append(dict(metric='analytic_charge_variance',group=g,weights=w,status='pass',reasons=[]))
            for h in ('fast','slow','common_level'):
                tag=dict(group=g,weights=w,hypothesis=h,status='pass',reasons=[])
                for m in ('hard_wave_mean','hard_wave_variance','conditional_cov9','conditional_energy',
                          'time_peak_index','time_rise10_90','time_FWHM'):
                    cells.append(dict(tag,metric=m))
                for f in ('Q','tail','peak_voltage','peak_index','rise10_90','FWHM'):
                    for level in (.5,.9):
                        cells.append(dict(tag,metric='PIT_coverage',feature=f,level=level))
                for f in required:
                    cells.append(dict(tag,metric='missingness_or_censor',feature=f,required_for_gate=True))
    return [dict(trial=f'T1_paired_seed{s}',cells=copy.deepcopy(cells)) for s in (0,1)],required


def main():
    stats=load_stats();checks=[]
    def check(name,condition,**detail):
        assert bool(condition),name
        checks.append(dict(name=name,passed=True,**detail))
    def reject(name,callback):
        try:
            callback()
        except ValueError:
            check(name,True)
        else:
            raise AssertionError(name)

    trials,required=primary_and_missing_fixture()
    check('exact_1392_primary_plus_1296_mandatory_causes',
          len(required)==18 and sum(c['metric']!='missingness_or_censor' for t in trials for c in t['cells'])==1392
          and sum(len(t['cells']) for t in trials)==2688)
    gate=stats.stage_gate017(trials)
    check('actual_stage_gate_opens_only_complete_two_trial_matrix',gate['stage2_allowed'] and gate['required_cell_count']==2688)
    for metric in ('conditional_cov9','time_peak_index','PIT_coverage','analytic_charge_variance'):
        bad=copy.deepcopy(trials)
        cell=next(c for c in bad[1]['cells'] if c['metric']==metric)
        cell.update(status='pending',reasons=['MC_unresolved'])
        check('actual_stage_gate_'+metric+'_pending_closes_stage2',not stats.stage_gate017(bad)['stage2_allowed'])
    bad=copy.deepcopy(trials);bad[0]['cells'][0]['status']='fail'
    check('one_primary_fail_closes_stage2',not stats.stage_gate017(bad)['stage2_allowed'])
    bad=copy.deepcopy(trials);bad[0]['cells'].pop()
    reject('one_required_missingness_cause_cannot_be_substituted_by_any_flag',lambda:stats.stage_gate017(bad))
    bad=copy.deepcopy(trials);bad[1]['cells'].append(copy.deepcopy(bad[1]['cells'][0]))
    reject('duplicate_primary_cell_rejected',lambda:stats.stage_gate017(bad))
    diagnostic=dict(metric='other_retained_flag_fraction',feature='flag__ADC_saturated',status='diagnostic',
                    absolute_difference=1.,scientific_threshold=None,required_for_gate=False)
    good=copy.deepcopy(trials)
    for trial in good:
        trial['other_retained_flag_diagnostics']=[copy.deepcopy(diagnostic)]
    check('diagnostic_ADC_difference_does_not_add_scientific_gate',stats.stage_gate017(good)['stage2_allowed'])

    n=100;group=np.ones(n,bool);weights=np.ones(n)
    moments=dict(full_slot_undefined_fraction=np.zeros((n,9)))
    causes=required[9:]
    teacher_flags={f:np.zeros(n) for f in causes}
    teacher_flags.update({'flag__ADC_saturated':np.zeros(n),'flag__peak_at_search_boundary':np.zeros(n),
                          'flag__Q_nonpositive':np.zeros(n)})
    student_flags={k:v.copy() for k,v in teacher_flags.items()}
    for k in ('flag__ADC_saturated','flag__peak_at_search_boundary','flag__Q_nonpositive'):
        student_flags[k][:]=1.
    rows=stats.missingness_cells017(moments,moments,student_flags,teacher_flags,group,weights)
    check('only_literal18_undefined_censor_causes_are_required',
          {r['feature'] for r in rows if r['required_for_gate']}==set(required)
          and all(r['status']=='pass' for r in rows if r['required_for_gate'])
          and all(r['status']=='diagnostic' and r['scientific_threshold'] is None for r in rows if not r['required_for_gate']))
    check('all_flag_fractions_preserve_original64_draw_denominator',
          all(r['original_draw_slots_per_parent']==64 for r in rows)
          and all(r['absolute_difference']==1. for r in rows if not r['required_for_gate']))
    student_flags['flag__rise_prepeak_censored'][:]=.02
    rows=stats.missingness_cells017(moments,moments,student_flags,teacher_flags,group,weights)
    check('required_undefined_censor_point02_fails_existing_point01_gate',
          next(r for r in rows if r['feature']=='flag__rise_prepeak_censored')['status']=='fail')
    missing=dict(student_flags);missing.pop('flag__FWHM_postpeak_censored')
    reject('missing_named_censor_cause_rejected',lambda:stats.missingness_cells017(moments,moments,missing,
          {k:v for k,v in teacher_flags.items() if k in missing},group,weights))

    qref=3.7103998046755806
    fit=qref*np.linspace(.25,2.,100);test=qref*np.linspace(.4,1.8,100)
    ident=stats.charge_identifiability017(fit,test,.01*qref,.0025)
    correct=True;details={}
    for role,q in (('fit',fit),('test',test)):
        # Independent scalar summations for the four fixed Gram entries.
        x=[float(qv/qref) for qv in q]
        gram=np.asarray([[math.fsum(v**2 for v in x),math.fsum(v**3 for v in x)],
                         [math.fsum(v**3 for v in x),math.fsum(v**4 for v in x)]])/len(x)
        actual=ident['design_by_role'][role]
        correct &= np.allclose(actual['Gram'],gram,rtol=8e-16,atol=0)
        eigenvalues=np.linalg.eigvalsh(gram);ratio=float(eigenvalues[0]/eigenvalues[-1])
        correct &= math.isclose(actual['eigenvalue_ratio'],ratio,rel_tol=2e-14,abs_tol=0)
        details[role]=dict(Gram=gram.tolist(),eigenvalue_ratio=ratio)
    check('fit_AND_test_fixed_Q_over_qref_Gram_matches_scalar_oracle',correct,actual=details)
    flat=stats.charge_identifiability017(np.full(100,qref),test,.01*qref,.0025)
    check('unidentifiable_fit_cannot_be_rescued_by_broad_test',not flat['individually_identifiable']
          and flat['secondary_25percent_pass'] is None and flat['secondary_only_not_primary_gate'])

    good=stats.variance_curve_cell017(test,.01*qref,.0025,group,weights)
    check('teacher_exact_analytic_law_passes_original_all_parent_gate',good['status']=='pass' and good['relative_max']==0.)
    reject('invalid_a_cannot_be_silently_dropped_by_empirical_support_rule',
           lambda:stats.variance_curve_cell017(test,float('nan'),.0025,group,weights))
    overflow_Q=np.ones(100);overflow_Q[0]=1e160
    bad=stats.variance_curve_cell017(overflow_Q,1.,.1,group,weights)
    check('one_invalid_original_analytic_parent_pending_not_99percent_PASS',bad['status']=='pending'
          and bad['invalid_parent_indices'].tolist()==[0] and not bad['invalid_parents_deleted'])
    Qmax=np.ones(100);Qmax[0]=1e5
    # Most parents are near a matched variance but one remains over the max gate.
    bad=stats.variance_curve_cell017(Qmax,.01*qref,.0031,group,weights)
    check('one_finite_original_parent_above_point20_max_cannot_be_removed',bad['status']=='fail' and bad['relative_max']>.20)

    factories=pure_factories();p=factories['settings017']()
    p.update(budget_caps=factories['budget_caps017'](p),stage1_budget_caps=factories['budget_caps017'](p,True))
    factories['validate_settings017'](p,frozen=False)
    check('preflight_binds_existing40GiB10GiB_GPUempty',p['preflight']==dict(minimum_free_disk_GiB=40,
           minimum_available_RAM_GiB=10,GPU_must_be_empty=True))
    for label,mutate in (
        ('missing',lambda x:x.pop('preflight')),
        ('disk8',lambda x:x['preflight'].__setitem__('minimum_free_disk_GiB',8)),
        ('RAM8',lambda x:x['preflight'].__setitem__('minimum_available_RAM_GiB',8)),
        ('GPU_nonempty_allowed',lambda x:x['preflight'].__setitem__('GPU_must_be_empty',False))):
        bad=copy.deepcopy(p);mutate(bad)
        reject('preflight_'+label+'_rejected_by_driver_factory_validator',lambda b=bad:factories['validate_settings017'](b,frozen=False))
    literal_full=dict(trial_slots=8,G_updates=16000,new_AD_queries=16000,checkpoint_loads=3,
        A_original_forward_rows=6593,student_charge_module_rows=20181568,teacher_charge_module_rows=17227904,
        shape_cached_basis_assembly_rows=37409472,hard_ADC_rows=112228416,total_scalar_normal_requests=14739331968,
        T2_teacher_sign_uniform_requests=8613952,coverage_PIT_uniform_requests=15190272,
        paired_training_energy_pair_distances=61440000,unpaired_training_energy_pair_distances=50331648000,
        evaluation_energy_pair_distances=931670016,checkpoint_saves_complete=17)
    literal_stage1=dict(trial_slots=2,G_updates=4000,new_AD_queries=4000,checkpoint_loads=3,
        A_original_forward_rows=6593,student_charge_module_rows=5361856,teacher_charge_module_rows=4517952,
        shape_cached_basis_assembly_rows=9879808,hard_ADC_rows=29639424,total_scalar_normal_requests=3892644352,
        T2_teacher_sign_uniform_requests=0,coverage_PIT_uniform_requests=7595136,
        paired_training_energy_pair_distances=30720000,unpaired_training_energy_pair_distances=0,
        evaluation_energy_pair_distances=303805440,checkpoint_saves_complete=5)
    check('full_and_stage1_fixed_scientific_costs_unchanged',
          all(p['budget_caps'][k]==v for k,v in literal_full.items())
          and all(p['stage1_budget_caps'][k]==v for k,v in literal_stage1.items())
          and p['trainable_parameters']==2 and p['failure_checkpoint_job_cap']==1)

    zero=stats.T2_fixed_basis_scope017(np.zeros((4,384)))
    nonzero=np.zeros((4,384));nonzero[:,100]=.125
    scope=stats.T2_fixed_basis_scope017(nonzero)
    check('zero_fixed_B0_recorded_untestable_without_new_gate_or_component',
          zero['status']=='untestable' and zero['exact_zero_parents']==4
          and not zero['component_or_mechanism_replacement_allowed'])
    check('nonzero_B0_preserves_unestablished_observed_mechanism_boundary',scope['status']=='inconclusive'
          and scope['theoretical_eta_fourth_moment']==1.6878 and not scope['hard_T1_T2_distinguishability_established']
          and not scope['separate_T1_T2_observed_contrast_computed'] and scope['extra_contrast_distances']==0
          and scope['additional_random_draws_or_forwards']==0 and not scope['conditional_charge_gate_pass_implies_mixture_recovery'])
    check('this_delta_check_never_imports_torch','torch' not in sys.modules)

    own=record(Path(__file__))
    payload=dict(schema='n01/wp4-017-independent-final-artificial-delta/v1',passed=True,
        source_review_layer='same_requested_SOURCE_REVIEW',checks=checks,check_count=len(checks),
        script=dict(path=str(Path(__file__).relative_to(ROOT)).replace('\\','/'),**own),
        sources={s:record(ROOT/s) for s in ('scripts/run_wp4_017.py','scripts/wp4_017_controller.py',
            'src/n01/model017.py','src/n01/stats017.py','src/n01/saved_array_kernels017.py')},
        fixture_scope='deterministic manufactured NumPy arrays and literal source factories only',
        actual_CP_SOURCE_REAL_DEV_reads=0,actual_NN_AD_G_RNG_GPU=0,server_operations=0,
        complete_scientific_acceptance=False)
    output=HERE/'FINAL_SEMANTIC_DELTA_CHECKS.json'
    assert not output.exists(),'Preserve prior independent result'
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,checks=len(checks),output=str(output),**record(output))))


if __name__=='__main__':
    main()
