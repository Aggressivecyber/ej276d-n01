from pathlib import Path
from datetime import datetime,timezone
import base64,copy,hashlib,json,math,sys
import numpy as np
from scipy.stats import wasserstein_distance
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;sys.path[:0]=[str(r/'scripts'),str(r/'src')]
from run_response064 import readouts064,verify_arrays064,STAGES064,FIELDS064
from run_response044 import FIELDS044
from run_response053 import CROSSINGS053
from run_response036 import WAVE_WEIGHTS
from run_response041 import ordered_contrasts
from run_response039 import morphology018
from wp1_evaluation import summary
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=read(w/'OVERFLOW_INVENTORY.json');p=read(r/'configs/response064_20261018.json');out=w/'collected/run_01';files={'run_01/'+x.name:read(x) for x in out.glob('*.json')}
collection=read(w/'COLLECTION_VERIFICATION.json');exit_receipt=read(w/'collected/control_01/exit_receipt.json')
assert exit_receipt['exit_code']==1 and files['run_01/FAILED.json']['error']=='039 output cap exceeded'
assert not (out/'RESULT.json').exists() and not (out/'COMPLETE.json').exists()
assert not any(x['present'] for x in v['identities'].values())
assert files['run_01/FAILED.json']['execution_counts']==p['operation_bounds']
arrays={x.name:dict(np.load(x,allow_pickle=False)) for x in out.glob('*.npz')};arrays['REAL_ROWS.npz']=dict(np.load(collection['REAL_reference'],allow_pickle=False));assert len(arrays)==53
reference=read(r/p['diagnostic_reference_file']);examples=read(r/p['example_selection_file'])['rows'];definitions=files['run_01/DEFINITIONS.json'];qfloor=definitions['q_floor']
inputs=dict(np.load(r/p['input_reference_file']));q=inputs['A0_canonical_Q_Vns'];ib=inputs['input_Q_bin'];real=arrays['REAL_ROWS.npz'];verify_arrays064('REAL_ROWS.npz',real,reference['arrays'])
rf={k:real[k] for k in FIELDS044};rf['PSD_raw']=rf['tail_Vns']/rf['Q_Vns'];rf['PSD_guarded']=rf['tail_Vns']/(np.where(rf['Q_Vns']<0.,-1.,1.)*np.maximum(np.abs(rf['Q_Vns']),qfloor))
template=read(r/p['template_file']);sk=np.log(template['source_Q_centers_Vns']);tk=np.log(template['target_Q_centers_Vns']);sp=np.array(template['source_unit_Q_profiles']);tp=np.array(template['target_unit_Q_profiles'])
moment_error=0.;linear_error=0.;example_error=0.;cross_error=0.;formula_error=0.;scale_error=0.;normalization_error=0.;raw_limit_error=0.;native_cross_reassociation_error=0.;native_row_checks=0;W1audit=[];stage_audit=[]
for seed,h in [(seed,h) for seed in range(2) for h in ('fast','slow','common_level')]:
    for case in ('BASE054','RESIDUAL'):
        label=f'seed{seed}_'+case;full=label+'_'+h;metric=files['run_01/'+full+'_METRICS.json'];assert metric==reference['reports'][str(seed)][case]['metrics'][h]
        for suffix in ('GEOMETRY_ROWS','EXAMPLES','INTRINSIC_MEAN_ROWS'):verify_arrays064(label+'_'+suffix+'.npz',arrays[label+'_'+suffix+'.npz'],reference['arrays'])
        geo=arrays[label+'_GEOMETRY_ROWS.npz'];limits=arrays[label+'_POSITIVITY_ROWS.npz'];rows=arrays[full+'_STAGE_ROWS.npz'];obs=arrays[full+'_OBSERVATION_FLAGS.npz'];ex=arrays[full+'_DIAGNOSTIC_EXAMPLES.npz'];report=files['run_01/'+full+'_STAGE_REPORT.json']['stages']
        native_rows={k:(rows[k] if k in rows else rows[k+'_by_stage'][-1]) for k in reference['arrays'][full+'_NATIVE_ROWS.npz']}
        cross_keys={k+'_absolute_ns' for k in CROSSINGS053};native_reference=reference['arrays'][full+'_NATIVE_ROWS.npz']
        verify_arrays064(full+'_NATIVE_ROWS.npz',native_rows,{full+'_NATIVE_ROWS.npz':{k:meta for k,meta in native_reference.items() if k not in cross_keys}})
        old_native=dict(np.load(r/'receipts/response063_20261018/wake_20261018_1733_01/collected/run_01'/(full+'_NATIVE_ROWS.npz')))
        verify_arrays064(full+'_NATIVE_ROWS.npz',old_native,reference['arrays'])
        for k in cross_keys:
            x,y=native_rows[k],old_native[k];assert np.array_equal(np.isnan(x),np.isnan(y));error=np.abs(x-y);bound=4*np.finfo(float).eps*np.maximum(np.abs(y),1.)
            assert np.all(error[np.isfinite(error)]<=bound[np.isfinite(error)]);native_cross_reassociation_error=max(native_cross_reassociation_error,float(np.nanmax(error)))
        verify_arrays064(full+'_OBSERVATION_FLAGS.npz',obs,reference['arrays']);native_row_checks+=6593
        assert np.array_equal(rows['parent_row'],np.arange(6593)) and np.array_equal(rows['input_Q_Vns'],q) and np.array_equal(rows['input_Q_bin'],ib)
        assert np.array_equal(limits['parent_row'],np.arange(6593)) and np.array_equal(limits['input_Q_Vns'],q)
        assert np.array_equal(np.minimum(1.,limits['raw_positivity_limit']),geo['feasible_scale']) and np.array_equal(limits['raw_positivity_limit']<=1.,geo['positivity_boundary_reached'])
        bounded=limits['has_negative_direction'];assert np.array_equal(bounded,np.isfinite(limits['raw_positivity_limit'])) and np.all(limits['limiting_absolute_index'][~bounded]==-1)
        assert np.all(np.isnan(limits['limiting_old_amplitude_V'][~bounded])) and np.all(np.isnan(limits['limiting_raw_direction_V'][~bounded]))
        assert np.all((limits['limiting_absolute_index'][bounded]>=110)&(limits['limiting_absolute_index'][bounded]<176)) and np.all(limits['limiting_raw_direction_V'][bounded]<0.)
        raw_limit_error=max(raw_limit_error,float(np.max(np.abs(limits['raw_positivity_limit'][bounded]-limits['limiting_old_amplitude_V'][bounded]/-limits['limiting_raw_direction_V'][bounded]))))
        assert np.array_equal(rows['group_counts'],np.array([[np.sum(bins==b) for b in range(8)] for bins in (ib,rows['BASE054_output_Q_bin'])]))
        for stage_index,stage in enumerate(STAGES064):
            saved=report[stage];view=saved['wave_views'];parts=dict(swd=float(np.mean(list(view['swd_by_view'].values()))),coordinate=float(np.mean([x for z in view['coordinate_by_16sample_block'].values() for x in z])),short_lag=float(np.mean(list(view['short_lag_by_view_and_lag'].values()))))
            assert max(abs(parts[k]-view['aggregate_parts'][k]) for k in parts)<1e-12 and abs(sum(WAVE_WEIGHTS[k]*value for k,value in parts.items())-view['aggregate_wave'])<1e-12
            mean=rows['population_wave_mean_V'][stage_index];SD=rows['population_wave_SD_V'][stage_index]
            values={k:rows[k+'_by_stage'][stage_index] for k in FIELDS064}
            for k,x in values.items():
                assert summary(x)==saved['features'][k]
                y=rf[k];a=x[np.isfinite(x)];b=y[np.isfinite(y)];dist=float(wasserstein_distance(a,b));support=np.r_[a,b];order=np.argsort(support,kind='stable');gap=np.diff(support[order]);mass=np.r_[np.full(len(a),len(b),np.int64),np.full(len(b),-len(a),np.int64)];cdf=np.cumsum(mass[order],dtype=np.int64)[:-1]
                independent=math.fsum((gap*np.abs(cdf)/(len(a)*len(b))).tolist());span=float(support.max()-support.min());eps=np.finfo(float).eps;saved_distance=saved['W1_to_REAL'][k]
                assert abs(dist-independent)<=8*eps*span and abs(saved_distance['value']-independent)<=8*eps*len(support)*span
                assert saved_distance['fake_rows']==6593 and saved_distance['real_rows']==21047 and saved_distance['fake_defined_rows']==len(a) and saved_distance['real_defined_rows']==len(b)
                W1audit.append(dict(seed=seed,hypothesis=h,case=case,stage=stage,field=k,error=abs(saved_distance['value']-independent)))
            for key,record in saved['flags'].items():
                x=rows[key+'_flag_by_stage'][stage_index];assert record==dict(count=int(x.sum()),denominator=6593)
            for field,left,right in [('rise10_90_ns','left_crossing10_ns','left_crossing90_ns'),('FWHM_ns','left_crossing50_ns','right_crossing50_ns')]:
                expected=rows[right+'_absolute_ns_by_stage'][stage_index]-rows[left+'_absolute_ns_by_stage'][stage_index];assert np.allclose(expected,values[field],rtol=0,atol=1e-12,equal_nan=True)
            linear_error=max(linear_error,abs(float(mean.sum()-values['Q_Vns'].mean())),abs(float(mean[:46].sum()-values['front_Vns'].mean())),abs(float(mean[46:].sum()-values['tail_Vns'].mean())),abs(float(mean[16]-values['peak_voltage_V'].mean())))
            for group,(name,bins) in enumerate([('input_Q',ib),('fixed_BASE054_output_Q',rows['BASE054_output_Q_bin'])]):
                gm=rows['group_wave_mean_V'][stage_index,group];gs=rows['group_wave_SD_V'][stage_index,group];weight=rows['group_counts'][group]/6593
                moment_error=max(moment_error,float(np.max(np.abs(weight@gm-mean))),float(np.max(np.abs(weight@(gs**2+(gm-mean)**2)-SD**2))))
                for b in range(8):
                    mask=bins==b;cell=saved['groups'][name][b];assert cell['rows']==int(mask.sum()) and cell['bin']==b
                    for k,x in values.items():assert summary(x[mask])==cell['features'][k]
                    for k,value in cell['flags'].items():assert value==int(rows[k+'_flag_by_stage'][stage_index,mask].sum())
            stage_audit.append(dict(seed=seed,hypothesis=h,case=case,stage=stage,rows=6593,all_row_features_flags_and_groups_verified=True))
        assert abs(report[STAGES064[-1]]['wave_views']['aggregate_wave']-metric['values']['wave'])<1e-12 and abs(report[STAGES064[-1]]['QPSD']-metric['values']['QPSD'])<1e-12
        selected=ex['parent_row'];assert selected.tolist()==examples[full]
        carriers=ex['carriers'];wave=ex['waves'];anchor=ex['BASE054_crop_indices'];own=ex['native_crop_indices'];assert np.array_equal(anchor,rows['BASE054_peak_index'][selected,None]-16+np.arange(256)) and np.array_equal(own,rows['native_peak_index'][selected,None]-16+np.arange(256))
        codes=np.round((np.clip(carriers[2],-.5,.5)+.5)*16383);assert np.array_equal(codes,np.round((carriers[3]+.5)*16383));example_error=max(example_error,float(np.max(np.abs(codes/16383-.5-carriers[3]))))
        for j in range(5):
            value=carriers[min(j,3)];centered=value-value[:,:100].mean(1,keepdims=True);expected=np.take_along_axis(centered,own if j==4 else anchor,axis=1);example_error=max(example_error,float(np.max(np.abs(expected-wave[j]))))
            features,flags=readouts064(wave[j],qfloor);morph=morphology018(wave[j]);start=(rows['native_peak_index'] if j==4 else rows['BASE054_peak_index'])[selected]-16
            for k,x in features.items():
                target=rows[k+'_by_stage'][j,selected];assert np.array_equal(np.isnan(x),np.isnan(target));example_error=max(example_error,float(np.nanmax(np.abs(x-target))))
            for k,x in flags.items():assert np.array_equal(x,rows[k+'_flag_by_stage'][j,selected])
            for k in CROSSINGS053:
                actual=morph[k]+start;target=rows[k+'_absolute_ns_by_stage'][j,selected];assert np.array_equal(np.isnan(actual),np.isnan(target));cross_error=max(cross_error,float(np.nanmax(np.abs(actual-target))))
            if j<4:
                peak=np.argmax(centered[:,112:144],axis=1)+112;globalpeak=np.argmax(centered,axis=1)
                assert np.array_equal(peak,rows['selected_peak_index_by_carrier_stage'][j,selected]) and np.array_equal(globalpeak,rows['global_peak_index_by_carrier_stage'][j,selected])
        oldex=arrays[f'seed{seed}_BASE054_{h}_DIAGNOSTIC_EXAMPLES.npz'];assert np.array_equal(selected,oldex['parent_row'])
        for ii,parent in enumerate(selected):
            old=oldex['carriers'][0,ii];peak=int(np.argmax(old[112:144]))+112;loc=np.arange(110,176)-peak+16;mask=(loc>=2)&(old[110:176]>0.);area=math.fsum(np.where(mask,old[110:176],0.))
            ss=np.array([np.interp(np.log(q[parent]),sk,sp[:,j]) for j in range(256)])[np.maximum(loc,2)]*mask;tt=np.array([np.interp(np.log(q[parent]),tk,tp[:,j]) for j in range(256)])[np.maximum(loc,2)]*mask
            sn=area/math.fsum(ss);tn=area/math.fsum(tt);direction=tn*tt-sn*ss;bound=np.full(66,np.inf);negative=direction<0.;bound[negative]=old[110:176][negative]/-direction[negative];raw=float(bound.min());scale=min(1.,raw)
            if raw<=1.:scale=np.nextafter(scale,0.)
            applied=scale if case=='RESIDUAL' else 0.;expected=old.copy();expected[110:176]+=applied*direction
            formula_error=max(formula_error,float(np.max(np.abs(direction-ex['raw_direction110_176'][ii]))),float(np.max(np.abs(expected-carriers[0,ii]))));scale_error=max(scale_error,abs(applied-ex['applied_scale'][ii]));normalization_error=max(normalization_error,abs(sn-geo['source_normalization'][parent]),abs(tn-geo['target_normalization'][parent]))
            if np.isfinite(raw):assert int(np.argmin(bound))+110==limits['limiting_absolute_index'][parent]
            else:assert limits['limiting_absolute_index'][parent]==-1
    pair=files[f'run_01/seed{seed}_{h}_PAIRED_CONTRASTS.json'];base=arrays[f'seed{seed}_BASE054_{h}_STAGE_ROWS.npz'];new=arrays[f'seed{seed}_RESIDUAL_{h}_STAGE_ROWS.npz']
    for k in ('Q_Vns','front_Vns','tail_Vns'):
        actual=ordered_contrasts(new[k+'_by_stage'],base[k+'_by_stage']);saved=pair['row_contrasts'][k];assert actual['closure_max_error']==saved['closure_max_error']
        for part in ('stage_delta','ordered_contributions'):assert [summary(x) for x in actual[part]]==saved[part]
    for k,z in pair['score_contrasts'].items():
        expected=ordered_contrasts(z['RESIDUAL'],z['BASE'])
        for name in ('stage_delta','ordered_contributions'):assert np.array_equal(expected[name],np.array(z[name]))
        assert expected['closure_max_error']==z['closure_max_error']
    for key,fields in pair['wave_component_contrasts'].items():
        for name,z in fields.items():assert np.max(np.abs(np.sum(z['ordered_contributions'],axis=0)-np.array(z['stage_delta'])[-1]))<1e-12
    for k,z in pair['flag_transitions'].items():
        b=base[k+'_flag_by_stage'];n=new[k+'_flag_by_stage'];expected=dict(BASE_counts=b.sum(1),RESIDUAL_counts=n.sum(1),new_relative_to_BASE=(n&~b).sum(1),resolved_relative_to_BASE=(~n&b).sum(1),RESIDUAL_adjacent_on=(n[1:]&~n[:-1]).sum(1),RESIDUAL_adjacent_off=(~n[1:]&n[:-1]).sum(1))
        for name,value in expected.items():assert np.array_equal(value,np.array(z[name]))
    for key,fields in pair['wave_component_contrasts'].items():
        for name,z in fields.items():
            old=[files[f'run_01/seed{seed}_BASE054_{h}_STAGE_REPORT.json']['stages'][stage]['wave_views'][key][name] for stage in STAGES064];actual=[files[f'run_01/seed{seed}_RESIDUAL_{h}_STAGE_REPORT.json']['stages'][stage]['wave_views'][key][name] for stage in STAGES064];expected=ordered_contrasts(actual,old)
            for part in ('stage_delta','ordered_contributions'):assert np.array_equal(np.array(z[part]),expected[part])
assert len(W1audit)==540 and len(stage_audit)==60 and native_row_checks==79116 and max(moment_error,linear_error,example_error,cross_error,formula_error,scale_error,normalization_error,raw_limit_error)<1e-10
for name,digest in p['source_pins'].items():assert sha(r/name)==digest,name
manifest=read(w/'collected/SOURCE_MANIFEST.json')
for name,meta in manifest['files'].items():assert (r/name).stat().st_size==meta['bytes'] and sha(r/name)==meta['sha256'],name
assert len(files['run_01/REPLAY_CHECKS.json'])==12 and all(x['full_report_exact063'] for x in files['run_01/REPLAY_CHECKS.json'])
record=dict(status='FAILED_OUTPUT_CAP_WITH_COMPLETE_DIAGNOSTIC_TABLES_VERIFIED',UTC=datetime.now(timezone.utc).isoformat(),original_exit_code=1,original_RESULT_and_COMPLETE_absent=True,original_final_deep_frozen_assertions_not_reached=True,original_exact_operation_count_guard_reached=True,failed_counts_match_contract=True,source_pins_verified=len(p['source_pins']),source_manifest_files_verified=len(manifest['files']),native_reports_exact063=12,native_rows_verified=native_row_checks,raw_reference_arrays_verified=37,geometry_rows_verified=26372,positivity_rows_verified=26372,intrinsic_mean_rows_verified=26372,stage_summaries_verified=60,stage_rows_verified=395580,carrier_stage_rows=316464,diagnostic_example_rows_verified=sum(len(v['parent_row']) for k,v in arrays.items() if k.endswith('_DIAGNOSTIC_EXAMPLES.npz')),native_stage_crossing_reassociation_max_error_ns=native_cross_reassociation_error,NaN_masks_preserved=True,formula_max_error_V=formula_error,scale_max_error=scale_error,normalization_max_error=normalization_error,raw_limit_internal_max_error=raw_limit_error,ADC_integer_codes_exact=True,example_stage_and_feature_max_error=example_error,example_crossing_max_error_ns=cross_error,group_moment_max_error=moment_error,linear_scalar_max_error=linear_error,W1_integer_CDF_and_scipy_checks=len(W1audit),W1_max_error=max(x['error'] for x in W1audit),W1_audit=W1audit,ordered_pair_contrasts_and_flag_transitions_verified=6,stages=stage_audit,uncertainty_scope='Full-row scalar and flag tables, saved group moments and selected full-wave examples verified; unsaved full-population wave and counterfactual QPSD not independently re-evaluated offline. Final original in-memory deep freeze postconditions were not reached after cap failure.',original_source_code_unchanged=True,scientific_goal_achieved=False)
assert record['diagnostic_example_rows_verified']==428
(w/'ARTIFACT_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:x for k,x in record.items() if k not in ('W1_audit','stages')}))
