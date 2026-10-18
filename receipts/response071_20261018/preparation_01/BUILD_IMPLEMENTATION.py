from pathlib import Path
import ast
r=Path('E:/GAN/research/N01');s=(r/'scripts/run_response070.py').read_text()
a=s.index('from run_response069 import level_warp069');b=s.index('@torch.no_grad()',a)
s=s[:a]+'''from run_response069 import level_warp069
from run_response070 import split_warp070
from response071_diagnostics import native_decomposition071


'''+s[b:]
s=s.replace('evaluate070','evaluate071').replace('run070','run071').replace('response070','response071').replace("'070 ","'071 ")
# Keep the reused frozen kernel as an import from the immutable 070 implementation.
s=s.replace('from run_response071 import split_warp070','from run_response070 import split_warp070')
s=s.replace('"""One crossing-level warp with separate front/tail integral constraints."""','"""Frozen070 replay with all-population native peak and tail decomposition."""')
needle="        progress['feature_records']+=1;progress['feature_rows']+=len(q)"
assert s.count(needle)==1
s=s.replace(needle,needle+'''
        if enabled:
            baseline_signal=r._ExactZeroMeanAddition015.apply(bank['M0'],fixed_delta)
            diagnostic,baseline_observation=native_decomposition071(fixed_mean,mean,random,random_delta,baseline_signal,base_signal,signal,bank['noise'][h],obs,float(setup['loss'].q_floor))
            progress['diagnostic_BASE054_observation_calls']+=1;progress['diagnostic_BASE054_observation_rows']+=len(q)
            diagnostic['BASE054_native_output_Q_bin']=output_q_bins(diagnostic['BASE054_Q_Vns'],setup['loss'])
            numeric=sum(value.dtype!=bool for value in diagnostic.values());boolean=sum(value.dtype==bool for value in diagnostic.values())
            if numeric>52 or boolean>10:raise RuntimeError('071 diagnostic output allocation exceeds contract')
            for name in ('Q','tail'):
                if np.max(np.abs(diagnostic[name+'_closure_Vns']))>1e-12:raise RuntimeError('071 paired decomposition does not close')
                if np.max(np.abs(diagnostic[name+'_native_delta_Vns']-(features[name+'_Vns']-diagnostic['BASE054_'+name+'_Vns'])))>1e-12:raise RuntimeError('071 decomposition differs from native integral')
            components=sum(diagnostic['peak_advantage_'+name+'_V'] for name in ('BASE054_mean','mean_change','original_random','random_change','electronic_noise','candidate_ADC_error','candidate_reassociation'))
            if np.max(np.abs(components-diagnostic['peak_advantage_candidate_ADC_V']))>1e-12:raise RuntimeError('071 peak competition does not close')
            np.savez_compressed(out/(label+'_'+h+'_DECOMPOSITION_ROWS.npz'),**diagnostic)
            progress['diagnostic_decomposition_records']+=1;progress['diagnostic_decomposition_rows']+=len(q)
            r.write_json(out/(label+'_'+h+'_DECOMPOSITION.json'),dict(rows=len(q),numeric_columns=numeric,boolean_columns=boolean,
                row_summaries={key:summary(value) for key,value in diagnostic.items() if key!='parent_row'},
                BASE054_observation_flags={key:dict(count=int(value.sum()),denominator=len(q)) for key,value in baseline_observation['flags'].items()},
                native_feature_equality_to_prior_BASE054_pending_offline_verification=True,
                decomposition='Native delta = same-new-window mean+random+ADC+floating reassociation changes + BASE054 window shift. Peak advantage is at candidate native peak minus BASE054 native peak. Ordered algebraic attribution, not unique causal effects.'))
            del baseline_observation,diagnostic,baseline_signal
''')
s=s.replace("            if index==0 and enabled:progress['new_candidate_settings']+=1\n",'')
s=s.replace("            if not enabled:\n                if json.loads(path.read_text())['metrics']!=replay[str(index)]['LEVELWARP']['metrics']:raise RuntimeError('071 LEVEL069 full report differs')\n                progress['fixed_case_replays']+=1\n            else:results.append(record);progress['candidate_head_evaluations']+=1", "            if json.loads(path.read_text())!=replay[str(index)][label]:raise RuntimeError('071 frozen070 full record differs '+label)\n            if not enabled:progress['fixed_case_replays']+=1\n            else:results.append(record);progress['frozen_endpoint_head_replays']+=1")
s=s.replace("stage_outcome='FIT_USEFUL_MEAN_PENDING_CONFIRMATION' if passed else 'CLOSED_NEGATIVE_SPLIT_AREA_CROSSING_LEVEL_WARP'", "stage_outcome='CLOSED_DIAGNOSTIC_COMPLETE',frozen070_negative_outcome_retained=True,new_response_candidates=0")
s=s.replace("    result=dict(schema=", "    if passed:raise RuntimeError('071 frozen070 unexpectedly changed acceptance outcome')\n    result=dict(schema=")
s=s.replace('response071-split-area-crossing-level-warp-result','response071-native-peak-tail-decomposition-result')
old="uncertainty_scope='Conditional50/90% nearest-peak voltage crossing warp anchored at BASE054 mean peak and support ends; separate positive-area scales before/from reference mean peak+30. The split may introduce a seam; reference integrals are not native PSD guarantees. The target observed profile is not preADC truth; its crossings need not equal mean per-event crossings. Residual uses the same interpolation and deterministic linear front/tail integral projections. FIT only; covariance changes; conditional zero expectation is preserved only if it held in the original residual. Multiple crests and input event detail are retained, not replaced by a template.'"
new="uncertainty_scope='No changed response or new endpoint. All070 FIT reports must replay exactly. Six additional BASE054 observations support paired native Q/tail and peak decompositions. Ordered algebraic terms include nonlinear ADC and readout selection; they are not uniquely identified physical noise sources. No training, confirmation or DEV.'"
assert old in s;s=s.replace(old,new)
assert 'candidate_head_evaluations' not in s
ast.parse(s);path=r/'scripts/run_response071.py';assert not path.exists();path.write_text(s)
print('071 frozen runner prepared')
