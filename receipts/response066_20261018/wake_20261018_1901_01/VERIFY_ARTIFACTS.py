from pathlib import Path
from datetime import datetime,timezone
import base64,copy,hashlib,json,math,sys
import numpy as np
from scipy.stats import wasserstein_distance
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;sys.path[:0]=[str(r/'scripts'),str(r/'src')]
from run_response066 import readouts066,verify_arrays066,STAGES066,FIELDS066
from run_response044 import FIELDS044
from run_response053 import CROSSINGS053
from run_response036 import WAVE_WEIGHTS
from run_response041 import ordered_contrasts
from run_response039 import morphology018
from wp1_evaluation import summary
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=read(w/'REMOTE_STATUS.json');p=read(r/'configs/response066_20261018.json');run=w/'collected/run_01';files={'run_01/'+x.name:read(x) for x in run.glob('*.json')}
collection=read(w/'COLLECTION_VERIFICATION.json');exit_receipt=read(w/'collected/control_01/exit_receipt.json');result=files['run_01/RESULT.json'];complete=files['run_01/COMPLETE.json']
assert exit_receipt['exit_code']==0 and exit_receipt['reason'] is None and not (run/'FAILED.json').exists()
assert complete['result_sha256']==sha(run/'RESULT.json') and result['run_id']==p['run_id'] and result['execution_counts']==p['operation_bounds']
assert result['stage_outcome']=='DIAGNOSTIC_COMPLETE_PENDING_ROOT_REVIEW' and result['frozen_A0_M0_005_verified']
assert result['all_TRAIN_SOURCE_parents']==6593 and result['all_TRAIN_REAL_rows']==21047 and result['all_parent_count_weights_preserved']
assert result['confirmation_banks_opened']==0 and not result['DEV_opened'] and result['new_candidate_settings']==0 and not result['mean_stage_accepted'] and not result['full_random_response_acceptance']
assert not any(x['present'] for x in v['identities'].values())
assert result['elapsed_seconds']<780 and exit_receipt['peak_rss_bytes']<12*1024**3 and collection['run_bytes']<64*1024**2
assert p['contract']==read(r/'configs/response039_20261018.json')['contract']
code=(r/'scripts/run_response066.py').read_text();assert code.index("if dict(progress)!=p['operation_bounds']:")<code.index('model.assert_frozen_unchanged(deep=True)')<code.index("r.write_json(out/'RESULT.json'")
arrays={x.name:dict(np.load(x,allow_pickle=False)) for x in run.glob('*.npz')};assert len(arrays)==24
reference=read(r/p['diagnostic_reference_file']);examples=read(r/p['example_selection_file'])['rows'];definitions=files['run_01/DEFINITIONS.json'];qfloor=definitions['q_floor']
prior=r/'receipts/response065_20261018/wake_20261018_1842_01/collected/run_01'
for name in reference['arrays']:arrays[name]=dict(np.load(prior/name,allow_pickle=False))
inputs=dict(np.load(r/p['input_reference_file']));q=inputs['A0_canonical_Q_Vns'];ib=inputs['input_Q_bin'];real=arrays['REAL_ROWS.npz'];verify_arrays066('REAL_ROWS.npz',real,reference['arrays'])
rf={k:real[k] for k in FIELDS044};rf['PSD_raw']=rf['tail_Vns']/rf['Q_Vns'];rf['PSD_guarded']=rf['tail_Vns']/(np.where(rf['Q_Vns']<0.,-1.,1.)*np.maximum(np.abs(rf['Q_Vns']),qfloor))
template=read(r/p['template_file']);sk=np.log(template['source_Q_centers_Vns']);tk=np.log(template['target_Q_centers_Vns']);sp=np.array(template['source_unit_Q_profiles']);tp=np.array(template['target_unit_Q_profiles'])
moment_error=0.;linear_error=0.;example_error=0.;cross_error=0.;formula_error=0.;column_error=0.;template_error=0.;peak_pair_error=0.;peak_closure_error=0.;normalization_error=0.;native_cross_reassociation_error=0.;native_row_checks=0;W1audit=[];stage_audit=[]
for name,meta in reference['arrays'].items():
    assert sha(prior/name)==reference['original_files'][name],name
    verify_arrays066(name,arrays[name],reference['arrays'])
for seed,h in [(seed,h) for seed in range(2) for h in ('fast','slow','common_level')]:
    for case in ('BASE054','MASS'):
        label=f'seed{seed}_'+case;full=label+'_'+h;metric=files['run_01/'+full+'_METRICS.json'];assert metric==reference['reports'][str(seed)][case]['metrics'][h]
        for suffix in ('GEOMETRY_ROWS','EXAMPLES','INTRINSIC_MEAN_ROWS'):verify_arrays066(label+'_'+suffix+'.npz',arrays[label+'_'+suffix+'.npz'],reference['arrays'])
        geo=arrays[label+'_GEOMETRY_ROWS.npz'];rows=arrays[full+'_STAGE_ROWS.npz'];obs=arrays[full+'_OBSERVATION_FLAGS.npz'];ex=arrays[full+'_DIAGNOSTIC_EXAMPLES.npz'];report=files['run_01/'+full+'_STAGE_REPORT.json']['stages']
        qq,tt=rows['Q_Vns_by_stage'],rows['tail_Vns_by_stage'];raw=np.full(qq.shape,np.nan);np.divide(tt,qq,out=raw,where=qq!=0.)
        rows['PSD_raw_by_stage']=raw;rows['PSD_guarded_by_stage']=tt/(np.where(qq<0.,-1.,1.)*np.maximum(np.abs(qq),qfloor))
        pair_voltages=rows['native_minus_mean_peak_advantage_V_by_carrier'];increments=np.concatenate((pair_voltages[:1],np.diff(pair_voltages,axis=0)),axis=0)
        assert pair_voltages.shape==(4,6593) and np.array_equal(rows['intrinsic_mean_peak_index'],arrays[label+'_INTRINSIC_MEAN_ROWS.npz']['intrinsic_peak_index'])
        assert pair_voltages[0].max()<1e-12 and pair_voltages[-1].min()>-1e-12
        same=rows['intrinsic_mean_peak_index']==rows['native_peak_index'];assert np.array_equal(pair_voltages[:,same],np.zeros((4,int(same.sum()))))
        pair_report=files['run_01/'+full+'_STAGE_REPORT.json'];peak_closure_error=max(peak_closure_error,float(np.max(np.abs(increments.sum(0)-pair_voltages[-1]))));assert pair_report['peak_competition_closure_max_error_V']<1e-12
        assert {name:summary(x) for name,x in zip(('mean','M0_random','electronic','quantization'),increments)}==pair_report['peak_competition_summaries']
        native_rows={k:(rows[k] if k in rows else rows[k+'_by_stage'][-1]) for k in reference['arrays'][full+'_NATIVE_ROWS.npz']}
        cross_keys={k+'_absolute_ns' for k in CROSSINGS053};native_reference=reference['arrays'][full+'_NATIVE_ROWS.npz']
        verify_arrays066(full+'_NATIVE_ROWS.npz',native_rows,{full+'_NATIVE_ROWS.npz':{k:meta for k,meta in native_reference.items() if k not in cross_keys}})
        old_native=dict(np.load(r/'receipts/response065_20261018/wake_20261018_1842_01/collected/run_01'/(full+'_NATIVE_ROWS.npz')))
        verify_arrays066(full+'_NATIVE_ROWS.npz',old_native,reference['arrays'])
        for k in cross_keys:
            x,y=native_rows[k],old_native[k];assert np.array_equal(np.isnan(x),np.isnan(y));error=np.abs(x-y);bound=4*np.finfo(float).eps*np.maximum(np.abs(y),1.)
            assert np.all(error[np.isfinite(error)]<=bound[np.isfinite(error)]);native_cross_reassociation_error=max(native_cross_reassociation_error,float(np.nanmax(error)))
        verify_arrays066(full+'_OBSERVATION_FLAGS.npz',obs,reference['arrays']);native_row_checks+=6593
        assert np.array_equal(rows['parent_row'],np.arange(6593)) and np.array_equal(rows['input_Q_Vns'],q) and np.array_equal(rows['input_Q_bin'],ib)
        assert np.array_equal(rows['group_counts'],np.array([[np.sum(bins==b) for b in range(8)] for bins in (ib,rows['BASE054_output_Q_bin'])]))
        for stage_index,stage in enumerate(STAGES066):
            saved=report[stage];view=saved['wave_views'];parts=dict(swd=float(np.mean(list(view['swd_by_view'].values()))),coordinate=float(np.mean([x for z in view['coordinate_by_16sample_block'].values() for x in z])),short_lag=float(np.mean(list(view['short_lag_by_view_and_lag'].values()))))
            assert max(abs(parts[k]-view['aggregate_parts'][k]) for k in parts)<1e-12 and abs(sum(WAVE_WEIGHTS[k]*value for k,value in parts.items())-view['aggregate_wave'])<1e-12
            mean=rows['population_wave_mean_V'][stage_index];SD=rows['population_wave_SD_V'][stage_index]
            values={k:rows[k+'_by_stage'][stage_index] for k in FIELDS066}
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
        assert abs(report[STAGES066[-1]]['wave_views']['aggregate_wave']-metric['values']['wave'])<1e-12 and abs(report[STAGES066[-1]]['QPSD']-metric['values']['QPSD'])<1e-12
        selected=ex['parent_row'];assert selected.tolist()==examples[full]
        carriers=ex['carriers'];wave=ex['waves'];anchor=ex['BASE054_crop_indices'];own=ex['native_crop_indices'];assert np.array_equal(anchor,rows['BASE054_peak_index'][selected,None]-16+np.arange(256)) and np.array_equal(own,rows['native_peak_index'][selected,None]-16+np.arange(256))
        codes=np.round((np.clip(carriers[2],-.5,.5)+.5)*16383);assert np.array_equal(codes,np.round((carriers[3]+.5)*16383));example_error=max(example_error,float(np.max(np.abs(codes/16383-.5-carriers[3]))))
        for j in range(5):
            value=carriers[min(j,3)];centered=value-value[:,:100].mean(1,keepdims=True);expected=np.take_along_axis(centered,own if j==4 else anchor,axis=1);example_error=max(example_error,float(np.max(np.abs(expected-wave[j]))))
            features,flags=readouts066(wave[j],qfloor);morph=morphology018(wave[j]);start=(rows['native_peak_index'] if j==4 else rows['BASE054_peak_index'])[selected]-16
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
            sn=area/math.fsum(ss);tn=area/math.fsum(tt);positive=np.where(mask,old[110:176],0.)
            sc=np.cumsum(ss);sc=np.r_[0.,sc/sc[-1]];tc=np.cumsum(tt);tc=np.r_[0.,tc/tc[-1]];sw=np.diff(sc);tw=np.diff(tc)
            overlap=np.maximum(0.,np.minimum(tc[1:,None],sc[None,1:])-np.maximum(tc[:-1,None],sc[None,:-1]));K=overlap/np.where(sw>0.,sw,1.)[None,:]
            expected=old.copy()
            if case=='MASS':expected[110:176]+=K@positive-positive
            formula_error=max(formula_error,float(np.max(np.abs(expected-carriers[0,ii]))));normalization_error=max(normalization_error,abs(sn-geo['source_normalization'][parent]),abs(tn-geo['target_normalization'][parent]))
            column_error=max(column_error,float(np.max(np.abs(K.sum(0)[mask]-1.))));template_error=max(template_error,float(np.max(np.abs(K@sw-tw))))
        k=rows['intrinsic_mean_peak_index'][selected];n=rows['native_peak_index'][selected];assert np.array_equal(k,np.argmax(carriers[0,:,112:144],axis=1)+112)
        independent_pair=np.array([(np.take_along_axis(value,n[:,None],axis=1)-np.take_along_axis(value,k[:,None],axis=1)).ravel() for value in carriers])
        peak_pair_error=max(peak_pair_error,float(np.max(np.abs(independent_pair-pair_voltages[:,selected]))))
    pair=files[f'run_01/seed{seed}_{h}_PAIRED_CONTRASTS.json'];base=arrays[f'seed{seed}_BASE054_{h}_STAGE_ROWS.npz'];new=arrays[f'seed{seed}_MASS_{h}_STAGE_ROWS.npz']
    for k in ('Q_Vns','front_Vns','tail_Vns'):
        actual=ordered_contrasts(new[k+'_by_stage'],base[k+'_by_stage']);saved=pair['row_contrasts'][k];assert actual['closure_max_error']==saved['closure_max_error']
        for part in ('stage_delta','ordered_contributions'):assert [summary(x) for x in actual[part]]==saved[part]
    for k,z in pair['score_contrasts'].items():
        expected=ordered_contrasts(z['MASS'],z['BASE'])
        for name in ('stage_delta','ordered_contributions'):assert np.array_equal(expected[name],np.array(z[name]))
        assert expected['closure_max_error']==z['closure_max_error']
    for key,fields in pair['wave_component_contrasts'].items():
        for name,z in fields.items():assert np.max(np.abs(np.sum(z['ordered_contributions'],axis=0)-np.array(z['stage_delta'])[-1]))<1e-12
    for k,z in pair['flag_transitions'].items():
        b=base[k+'_flag_by_stage'];n=new[k+'_flag_by_stage'];expected=dict(BASE_counts=b.sum(1),MASS_counts=n.sum(1),new_relative_to_BASE=(n&~b).sum(1),resolved_relative_to_BASE=(~n&b).sum(1),MASS_adjacent_on=(n[1:]&~n[:-1]).sum(1),MASS_adjacent_off=(~n[1:]&n[:-1]).sum(1))
        for name,value in expected.items():assert np.array_equal(value,np.array(z[name]))
    for key,fields in pair['wave_component_contrasts'].items():
        for name,z in fields.items():
            old=[files[f'run_01/seed{seed}_BASE054_{h}_STAGE_REPORT.json']['stages'][stage]['wave_views'][key][name] for stage in STAGES066];actual=[files[f'run_01/seed{seed}_MASS_{h}_STAGE_REPORT.json']['stages'][stage]['wave_views'][key][name] for stage in STAGES066];expected=ordered_contrasts(actual,old)
            for part in ('stage_delta','ordered_contributions'):assert np.array_equal(np.array(z[part]),expected[part])
assert len(W1audit)==540 and len(stage_audit)==60 and native_row_checks==79116 and max(moment_error,linear_error,example_error,cross_error,formula_error,column_error,template_error,normalization_error,peak_pair_error,peak_closure_error)<1e-10
for name,digest in p['source_pins'].items():assert sha(r/name)==digest,name
manifest=read(w/'collected/SOURCE_MANIFEST.json');assert manifest['config_sha256']==sha(r/'configs/response066_20261018.json')
for name,meta in manifest['files'].items():assert (r/name).stat().st_size==meta['bytes'] and sha(r/name)==meta['sha256'],name
assert len(manifest['files'])==418 and v['terminal_source_verification']['source_files']==418 and v['terminal_source_verification']['all_hashes_verified']
assert len(files['run_01/REPLAY_CHECKS.json'])==12 and all(x['full_report_exact065'] for x in files['run_01/REPLAY_CHECKS.json'])
record=dict(status='PASS',UTC=datetime.now(timezone.utc).isoformat(),native_exit_code=0,source_pins_verified=len(p['source_pins']),source_manifest_files_verified=len(manifest['files']),frozen_A0_M0_005_verified=True,execution_counts_verified=result['execution_counts'],native_reports_exact065=12,native_rows_verified=native_row_checks,raw_reference_arrays_verified=37,reused_geometry_rows_verified=26372,reused_intrinsic_rows_verified=26372,stage_summaries_verified=60,stage_rows_verified=395580,carrier_stage_rows=316464,peak_competition_records=12,peak_competition_rows_verified=79116,compact_PSD_arrays_reconstructed=True,diagnostic_example_rows_verified=sum(len(v['parent_row']) for k,v in arrays.items() if k.endswith('_DIAGNOSTIC_EXAMPLES.npz')),native_stage_crossing_reassociation_max_error_ns=native_cross_reassociation_error,NaN_masks_preserved=True,formula_max_error_V=formula_error,column_mass_max_error=column_error,template_mapping_max_error=template_error,normalization_max_error=normalization_error,peak_pair_example_max_error_V=peak_pair_error,peak_pair_closure_max_error_V=peak_closure_error,ADC_integer_codes_exact=True,example_stage_and_feature_max_error=example_error,example_crossing_max_error_ns=cross_error,group_moment_max_error=moment_error,linear_scalar_max_error=linear_error,W1_integer_CDF_and_scipy_checks=len(W1audit),W1_max_error=max(x['error'] for x in W1audit),W1_audit=W1audit,ordered_pair_contrasts_flag_transitions_wave_blocks_verified=6,stages=stage_audit,uncertainty_scope='Full-row scalar and flag tables, saved group moments, selected full-wave examples and peak-pair differences verified; unsaved full-population waves and counterfactual QPSD not independently re-evaluated offline.',scientific_goal_achieved=False)
assert record['diagnostic_example_rows_verified']==288
(w/'ARTIFACT_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:x for k,x in record.items() if k not in ('W1_audit','stages','execution_counts_verified')}))
