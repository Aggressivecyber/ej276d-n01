from pathlib import Path
import json,sys
import numpy as np
from scipy.stats import spearmanr,wasserstein_distance
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;o=w/'collected/run_01';sys.path[:0]=[str(r/'scripts'),str(r/'src')]
from wp1_evaluation import summary
from run_response039 import HYPOTHESES,MORPH
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(w/'ARTIFACT_VERIFICATION.json')['status']=='PASS'
old=r/'receipts/response056_20261018/wake_20261018_1325_01/collected/run_01'
real=dict(np.load(old/'REAL_ROWS.npz'));inputs=dict(np.load(old/'INPUT_CONTENT.npz'));q=inputs['A0_canonical_Q_Vns']
T=read(r/'receipts/response039_20261018/preparation_01/CALIBRATED_TOLERANCES_READONLY.json')['T_by_hypothesis']
solution=read(o/'LINEARIZATION_AND_SOLUTION.json');cases=('BASE054','PROBE_0_neg','PROBE_0_pos','PROBE_1_neg','PROBE_1_pos','ENDPOINT')
probe_rows=[];example_flips=[];cells=[];variance_errors=[]
for seed in (0,1):
    m0=read(o/f'seed{seed}_M0.json');base_report=read(o/f'seed{seed}_BASE054.json')
    for case in cases:
        rec=read(o/f'seed{seed}_{case}.json');g=dict(np.load(o/f'seed{seed}_{case}_GEOMETRY_ROWS.npz'));ex=dict(np.load(o/f'seed{seed}_{case}_EXAMPLES.npz'))
        probe_rows.append(dict(seed=seed,case=case,coefficients=rec['geometry']['coefficients'],feasible=rec['decision']['feasible'],violations=rec['decision']['violations'],
            mean_peak_changed_rows=int((g['new_mean_peak_index']!=g['old_mean_peak_index']).sum()),maximum_correction_V=rec['geometry']['maximum_absolute_correction_V'],
            maximum_absolute_log_tilt=rec['geometry']['maximum_absolute_log_tilt'],
            core_peak_changed_rows=rec['geometry']['core_peak_changed_rows'],boundary_delta_max_absolute_V=rec['geometry']['boundary_delta_max_absolute_V'],
            core_rank_minimum_adjacent_difference_V=float(g['core_rank_minimum_adjacent_difference_V'].min()),
            by_hypothesis={h:dict(wave_gain=(m0[h]['values']['wave']-rec['metrics'][h]['values']['wave'])/m0[h]['values']['wave'],
                joint_gain=(m0[h]['values']['joint_morphology']-rec['metrics'][h]['values']['joint_morphology'])/m0[h]['values']['joint_morphology'],
                morphology_gain_T=float(np.mean([(m0[h]['values'][k]-rec['metrics'][h]['values'][k])/T[h][k] for k in MORPH])),
                tail_excess_T=(rec['metrics'][h]['values']['tail_Vns']-m0[h]['values']['tail_Vns'])/T[h]['tail_Vns'],
                FWHM_excess_T=(rec['metrics'][h]['values']['FWHM_ns']-m0[h]['values']['FWHM_ns'])/T[h]['FWHM_ns'],
                FWHM_undefined_change=rec['metrics'][h]['fake_flags']['FWHM_undefined']['count']-m0[h]['fake_flags']['FWHM_undefined']['count'],
                rise_undefined_change=rec['metrics'][h]['fake_flags']['rise_undefined']['count']-m0[h]['fake_flags']['rise_undefined']['count']) for h in HYPOTHESES}))
        for i in range(32):
            p0,p1=int(g['old_mean_peak_index'][i]),int(g['new_mean_peak_index'][i])
            if p0==p1:continue
            before,after,delta=ex['fixed_mean_pre_V'][i],ex['adapted_mean_pre_V'][i],ex['correction_pre_V'][i]
            gap=float(before[p0]-before[p1]);contrast=float(delta[p1]-delta[p0]);assert gap>=0 and contrast>=gap-1e-14
            example_flips.append(dict(seed=seed,case=case,parent_row=i,Q=float(q[i]),old_peak=p0,new_peak=p1,
                old_voltage_advantage_V=gap,correction_contrast_toward_new_peak_V=contrast,old_peak_voltage_V=float(before[p0]),
                competing_height_fraction=float(before[p1]/before[p0]),maximum_correction_V=float(np.max(np.abs(delta)))))
    g=dict(np.load(o/f'seed{seed}_ENDPOINT_GEOMETRY_ROWS.npz'))
    for h in HYPOTHESES:
        b=dict(np.load(o/f'seed{seed}_BASE054_{h}_FEATURE_ROWS.npz'));e=dict(np.load(o/f'seed{seed}_ENDPOINT_{h}_FEATURE_ROWS.npz'))
        peak_delta=e['peak_index']-b['peak_index'];changed=peak_delta!=0;tail_delta=e['tail_Vns']-b['tail_Vns'];q_delta=e['Q_Vns']-b['Q_Vns']
        assert np.array_equal(tail_delta[~changed],np.zeros(int((~changed).sum())))
        tail_positive=tail_delta>0;tail_negative=tail_delta<0
        group_rows=[];associations=[]
        for group,bin_id in [('input_Q',inputs['input_Q_bin']),('fixed_BASE054_output_Q',b['native_output_Q_bin']),('own_ENDPOINT_output_Q',e['native_output_Q_bin'])]:
            for j in range(8):
                mask=bin_id==j;rmask=real['output_Q_bin']==j
                row=dict(group=group,bin=j,SOURCE_rows=int(mask.sum()),input_Q=summary(q[mask]),mean_peak_changed=int((g['old_mean_peak_index'][mask]!=g['new_mean_peak_index'][mask]).sum()),
                    native_peak_changed=int(changed[mask].sum()),log_tilt=summary(g['log_tilt'][mask]),maximum_correction=summary(g['maximum_absolute_delta_V'][mask]),
                    boundary_delta_110_V=summary(g['boundary_delta_110_V'][mask]),boundary_delta_141_V=summary(g['boundary_delta_141_V'][mask]),
                    native_peak_shift=summary(peak_delta[mask]),native_tail_delta=summary(tail_delta[mask]),native_Q_delta=summary(q_delta[mask]),features={})
                for k in MORPH:
                    defined=mask&np.isfinite(b[k])&np.isfinite(e[k]);before=b[k][defined];after=e[k][defined];delta=after-before
                    covariance=float(np.mean((before-before.mean())*(delta-delta.mean())))
                    residual=abs(float(after.var()-before.var()-delta.var()-2*covariance));variance_errors.append(residual);assert residual<1e-12
                    row['features'][k]=dict(BASE054=summary(b[k][mask]),ENDPOINT=summary(e[k][mask]),paired_delta=summary(delta),common_defined_rows=len(delta),
                        variance_delta=float(after.var()-before.var()),variance_of_paired_delta=float(delta.var()),twice_covariance=2*covariance)
                    if group!='input_Q':row['features'][k]['REAL_same_observed_Q_bin']=summary(real[k][rmask])
                group_rows.append(row)
        for k in MORPH:
            for name,source in [('A0_canonical_Q',q)]+[(a,inputs[a+'_A0_fixed110_366']) for a in MORPH]:
                finite=np.isfinite(source)&np.isfinite(e[k]);rho=float(spearmanr(source[finite],e[k][finite]).statistic)
                associations.append(dict(input=name,output=k,rows=int(finite.sum()),spearman=rho))
        values,counts=np.unique(peak_delta,return_counts=True)
        cells.append(dict(seed=seed,hypothesis=h,native_peak_shift_histogram={str(int(k)):int(n) for k,n in zip(values,counts)},
            native_peak_changed_rows=int(changed.sum()),unchanged_peak_tail_delta_exact_zero=True,
            all_native_tail_change_confined_to_peak_changed_rows=True,tail_delta_positive_rows=int(tail_positive.sum()),tail_delta_negative_rows=int(tail_negative.sum()),
            mean_tail_change_Vns=float(tail_delta.mean()),total_signed_tail_change_Vns=float(tail_delta.sum()),tail_delta_min_Vns=float(tail_delta.min()),tail_delta_max_Vns=float(tail_delta.max()),
            tail_change_from_positive_peak_shifts_Vns=float(tail_delta[peak_delta>0].sum()),tail_change_from_negative_peak_shifts_Vns=float(tail_delta[peak_delta<0].sum()),
            same_peak_Q_delta=summary(q_delta[~changed]),changed_peak_Q_delta=summary(q_delta[changed]),
            input_Q_native_width_rho=float(spearmanr(q,e['FWHM_ns']).statistic),own_native_Q_width_rho=float(spearmanr(e['Q_Vns'],e['FWHM_ns']).statistic),
            REAL_own_Q_width_rho=float(spearmanr(real['Q_Vns'][np.isfinite(real['FWHM_ns'])],real['FWHM_ns'][np.isfinite(real['FWHM_ns'])]).statistic),
            grouping_rows=group_rows,input_associations=associations))
record=dict(cells=cells,probe_rows=probe_rows,first32_example_mean_peak_flips=example_flips,maximum_paired_variance_closure_error=max(variance_errors),
    endpoint_negative_u_rows=int((g['log_tilt']<0).sum()),endpoint_positive_u_rows=int((g['log_tilt']>0).sum()),input_rows=6593,
    LP_active_constraints=read(w/'ARTIFACT_VERIFICATION.json')['LP_active_nonzero_marginals'],
    conclusion='058 preserves deterministic core ordering but endpoint fails seed1 FWHM protection and all six morphology usefulness targets. All native tail-feature changes remain restricted to native reselected-peak rows; conditional width-Q trend remains reversed. No claim of global infeasibility.',
    no_new_scientific_evaluations=True,no_new_optimization=True,groups_are_descriptive_not_paired_energy_truth=True)
(w/'CONSTRAINT_AND_CONDITIONAL_ANALYSIS.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(endpoint_negative_u_rows=record['endpoint_negative_u_rows'],endpoint_positive_u_rows=record['endpoint_positive_u_rows'],
    cells=[{k:v for k,v in c.items() if k not in ('grouping_rows','input_associations')} for c in cells],
    probes=[{k:v for k,v in a.items() if k!='violations'} for a in probe_rows[:6]],endpoint_example_peak_flips=[a for a in example_flips if a['case']=='ENDPOINT' and a['seed']==0])))
