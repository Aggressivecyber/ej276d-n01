from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,math
import numpy as np
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;sys.path[:0]=[str(r/'scripts'),str(r/'src')]
from run_response039 import objective039,morphology018
from run_response044 import readouts044,FIELDS044
from run_response053 import CROSSINGS053
from wp1_evaluation import summary
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=read(w/'REMOTE_STATUS.json');p=read(r/'configs/response068_20261018.json');run=w/'collected/run_01';files={'run_01/'+x.name:read(x) for x in run.glob('*.json')}
collection=read(w/'COLLECTION_VERIFICATION.json');exit_receipt=read(w/'collected/control_01/exit_receipt.json');result=files['run_01/RESULT.json'];complete=files['run_01/COMPLETE.json']
assert exit_receipt['exit_code']==0 and exit_receipt['reason'] is None and not (run/'FAILED.json').exists()
assert complete['result_sha256']==sha(run/'RESULT.json') and result['run_id']==p['run_id'] and result['execution_counts']==p['operation_bounds']
assert result['frozen_A0_M0_005_verified'] and result['all_TRAIN_SOURCE_parents']==6593 and result['all_TRAIN_REAL_rows']==21047 and result['all_parent_count_weights_preserved']
assert result['confirmation_banks_opened']==0 and not result['DEV_opened'] and not result['mean_stage_accepted'] and not result['full_random_response_acceptance']
assert not any(x['present'] for x in v['identities'].values())
assert result['elapsed_seconds']<780 and exit_receipt['peak_rss_bytes']<12*1024**3 and collection['run_bytes']<64*1024**2
assert p['contract']==read(r/'configs/response039_20261018.json')['contract']
code=(r/'scripts/run_response068.py').read_text();assert code.index("if dict(progress)!=p['operation_bounds']:")<code.index('model.assert_frozen_unchanged(deep=True)')<code.index("r.write_json(out/'RESULT.json'")
arrays={x.name:dict(np.load(x,allow_pickle=False)) for x in run.glob('*.npz')};assert len(arrays)==72
expected=read(r/p['replay_file'])['reports']
coupling_formula_error=0.;coupling_scalar_error=0.;coupling_feature_error=0.;coupling_examples_verified=0;formula_error=0.;normalization_error=0.;adc_voltage_error=0.;window_error=0.;feature_error=0.;crossing_error=0.;column_error=0.;template_error=0.;displacement_error=0.
query_error=0.;amplitude_scale_error=0.
for seed in range(2):
    m0=files[f"run_01/seed{seed}_M0.json"];assert m0==expected[str(seed)]["M0"]
    assert files[f"run_01/seed{seed}_COUPLED067.json"]["metrics"]==expected[str(seed)]["COUPLED"]["metrics"]
    prior=r/'receipts/response067_20261018/wake_20261018_1921_01/collected/run_01'
    exact=['REAL_ROWS.npz','REAL_PROFILES.npz',f'seed{seed}_COUPLED067_EXAMPLES.npz',f'seed{seed}_COUPLED067_GEOMETRY_ROWS.npz',f'seed{seed}_COUPLED067_INTRINSIC_MEAN_ROWS.npz',f'seed{seed}_COUPLED067_fast_CARRIER_EXAMPLES.npz',f'seed{seed}_COUPLED067_fast_PROFILES.npz']+[f'seed{seed}_COUPLED067_{h}_{kind}.npz' for h in ('fast','slow','common_level') for kind in ('NATIVE_EXAMPLES','OBSERVATION_FLAGS')]
    for name in exact:
        new=arrays[name];before=dict(np.load(prior/name.replace('COUPLED067','COUPLED')));assert set(new)==set(before)
        for key in new:assert np.array_equal(new[key],before[key],equal_nan=True),(name,key)
    for h in ('fast','slow','common_level'):
        new=arrays[f'seed{seed}_COUPLED067_{h}_NATIVE_ROWS.npz'];before=dict(np.load(prior/f'seed{seed}_COUPLED_{h}_NATIVE_ROWS.npz'))
        assert {k for k in new if not k.startswith('COUPLED067_')}=={k for k in before if not k.startswith('MASS065_')}
        for key in new:
            if not key.startswith('COUPLED067_'):assert np.array_equal(new[key],before[key],equal_nan=True),(h,key)
    input_ref=dict(np.load(r/p['input_reference_file']));q=input_ref['A0_canonical_Q_Vns'];T=files['run_01/DEFINITIONS.json']['T_by_hypothesis']
    assert T==read(r/'receipts/response039_20261018/preparation_01/CALIBRATED_TOLERANCES_READONLY.json')['T_by_hypothesis']
    template=read(r/p['template_file']);sk=np.log(template['source_Q_centers_Vns']);tk=np.log(template['target_Q_centers_Vns']);sp=np.array(template['source_unit_Q_profiles']);tp=np.array(template['target_unit_Q_profiles'])
    definitions=read(r/'receipts/response056_20261018/wake_20261018_1325_01/collected/run_01/INPUT_AND_REAL_DEFINITIONS.json')
    qref=definitions['output_Q_reference_Vns'];qfloor=definitions['output_Q_floor_Vns'];bounds=definitions['output_Q_boundaries']
    for case,enabled in [('COUPLED067',False),('VOLTAGEWARP',True)]:
        label=f'seed{seed}_'+case;rec=files['run_01/'+label+'.json'];rows=arrays[label+'_GEOMETRY_ROWS.npz'];ex=arrays[label+'_EXAMPLES.npz'];intrinsic=arrays[label+'_INTRINSIC_MEAN_ROWS.npz']
        assert rec['decision']==objective039(rec['metrics'],m0,T,p['contract'])
        for h in ('fast','slow','common_level'):
            u=rec['point_usefulness'][h];a=rec['metrics'][h]['values'];b=m0[h]['values']
            assert u['wave_relative_gain']==(b['wave']-a['wave'])/b['wave'] and u['joint_relative_gain']==(b['joint_morphology']-a['joint_morphology'])/b['joint_morphology']
            gain=float(np.mean([(b[k]-a[k])/T[h][k] for k in ('peak_voltage_V','rise10_90_ns','FWHM_ns')]))
            assert u['morphology_mean_T_gain']==gain
            assert u['passes_point_targets']==(u['wave_relative_gain']>=.02 and u['joint_relative_gain']>=.02 and gain>=1.)
        assert np.array_equal(rows['parent_row'],np.arange(6593)) and np.array_equal(rows['input_Q_Vns'],q)
        assert np.max(np.abs(rows['full_integral_residual_Vns']))<1e-12 and np.max(np.abs(rows['canonical_Q_residual_Vns']))<1e-12
        assert np.array_equal(rows['negative_area_change_Vns'],np.zeros(6593))
        for prefix,knots in [('source',sk),('target',tk)]:
            lookup=np.clip(np.log(q),knots[0],knots[-1]);segment=np.clip(np.searchsorted(knots,lookup,side='right')-1,0,6);weight=(lookup-knots[segment])/(knots[segment+1]-knots[segment])
            assert np.max(np.abs(lookup-rows[prefix+'_lookup_logQ']))<1e-14 and np.array_equal(segment,rows[prefix+'_interval']) and np.max(np.abs(weight-rows[prefix+'_right_weight']))<1e-14
            assert np.array_equal(rows[prefix+'_clamped_low'],np.log(q)<knots[0]) and np.array_equal(rows[prefix+'_clamped_high'],np.log(q)>knots[-1])
        assert np.sum(rows['old_mean_peak_index']!=rows['new_mean_peak_index'])==rec['geometry']['mean_peak_changed_rows']
        for key,value in rows.items():
            if key!='parent_row':assert summary(value)==rec['geometry']['row_summaries'][key],(case,key)
        assert rec['geometry']['mean_addition_reassociation_max_error_V']<1e-12
        assert np.all(rows['minimum_mapped_active_V']>=0.)
        coupling=arrays[label+'_COUPLING_ROWS.npz'];ce=arrays[label+'_COUPLING_EXAMPLES.npz'];ids=ce['parent_row']
        assert np.array_equal(coupling['parent_row'],np.arange(6593)) and np.array_equal(coupling['input_Q_Vns'],q)
        assert np.array_equal(ids,p['coupling_example_parent_rows'][str(seed)]) and np.array_equal(ce['input_Q_Vns'],q[ids])
        assert rec['geometry']['random_coupling_enabled'] is True and rec['geometry']['peak_anchored_voltage_warp_enabled']==enabled
        assert max(np.abs(coupling['full_integral_residual_Vns']).max(),np.abs(coupling['canonical_Q_residual_Vns']).max())<1e-12
        for key,val in coupling.items():
            if key not in ('parent_row','input_Q_Vns'):assert summary(val)==rec['geometry']['random_row_summaries'][key]
        for key in ('parent_row','input_Q_Vns','fixed_mean_pre_V','old_random_V','noise_fast_V'):
            assert np.array_equal(ce[key],arrays[f'seed{seed}_COUPLED067_COUPLING_EXAMPLES.npz'][key]),key
        for key,key2 in [('fixed_mean_pre_V','fixed_mean_pre_V'),('adapted_mean_pre_V','adapted_mean_pre_V')]:assert np.array_equal(ce[key][:32],ex[key2])
        if not enabled:
            before=dict(np.load(prior/f'seed{seed}_COUPLED_COUPLING_EXAMPLES.npz'))
            assert set(ce)==set(before)|{'adapted_mean_pre_V'}
            for key in before:assert np.array_equal(ce[key],before[key],equal_nan=True),key
            before=dict(np.load(prior/f'seed{seed}_COUPLED_COUPLING_ROWS.npz'))
            for key in before:assert np.array_equal(coupling[key],before[key],equal_nan=True),key
            assert np.max(rows['column_mass_max_error'])<1e-12 and np.max(rows['template_probability_max_error'])<1e-12
        else:
            queries=arrays[label+'_WARP_QUERIES.npz'];assert np.array_equal(queries['parent_row'],np.arange(6593))
            saved_query=queries['inverse_query_absolute_sample'];assert saved_query.shape==(6593,66) and saved_query.min()>=110. and saved_query.max()<=175. and np.diff(saved_query,axis=1).min()>=0.
            expected_queries=[]
            for row in range(6593):
                peak=int(rows['old_mean_peak_index'][row]);loc=np.arange(110,176)-peak+16;a=max(0,peak-124);pidx=peak-110
                ss=(1-rows['source_right_weight'][row])*sp[rows['source_interval'][row]]+rows['source_right_weight'][row]*sp[rows['source_interval'][row]+1]
                tt=(1-rows['target_right_weight'][row])*tp[rows['target_interval'][row]]+rows['target_right_weight'][row]*tp[rows['target_interval'][row]+1]
                ss=ss[np.maximum(loc,2)];tt=tt[np.maximum(loc,2)];sc=np.r_[0.,np.cumsum(.5*(ss[:-1]+ss[1:]))];tc=np.r_[0.,np.cumsum(.5*(tt[:-1]+tt[1:]))]
                query=np.arange(66,dtype=float)
                for left,right in ((a,pidx),(pidx,65)):
                    f=(tc[left:right+1]-tc[left])/(tc[right]-tc[left]);query[left:right+1]=np.interp(sc[left]+f*(sc[right]-sc[left]),sc,np.arange(66))
                query[[a,pidx,65]]=[a,pidx,65];expected_queries.append(query)
                assert saved_query[row,pidx]==peak and saved_query[row,a]==a+110 and saved_query[row,65]==175.
            expected_queries=np.array(expected_queries);query_error=max(query_error,float(np.abs(expected_queries+110-saved_query).max()));assert query_error<1e-10
            assert np.max(rows['peak_query_error_samples'])==0. and np.min(rows['query_minimum_spacing_samples'])>=0.
            assert np.max(np.abs(rows['query_maximum_displacement_samples']-np.abs(saved_query-np.arange(110,176)).max(1)))<1e-12
            assert np.max(np.abs(rows['query_minimum_spacing_samples']-np.diff(saved_query,axis=1).min(1)))<1e-12
        for i,row in enumerate(ids):
            old=ce['fixed_mean_pre_V'][i];rnd=ce['old_random_V'][i];delta=ce['random_correction_V'][i];new=ce['adapted_mean_pre_V'][i];peak=np.argmax(old[112:144])+112;loc=np.arange(110,176)-peak+16;mask=(loc>=2)&(old[110:176]>0.)
            positive=np.where(mask,old[110:176],0.);active=np.where(mask,rnd[110:176],0.);area=math.fsum(positive)
            if enabled:
                query=expected_queries[row];vp=np.interp(query,np.arange(66),positive)*mask;alpha=area/math.fsum(vp);moved=alpha*vp
                provisional=alpha*np.interp(query,np.arange(66),active)*mask;moved_random=provisional-(math.fsum(provisional)-math.fsum(active))*moved/area
                amplitude_scale_error=max(amplitude_scale_error,abs(alpha-rows['amplitude_scale'][row]))
            else:
                ss=np.array([np.interp(float(np.log(q[row])),sk,sp[:,t]) for t in range(256)])[np.maximum(loc,2)]*mask;tt=np.array([np.interp(float(np.log(q[row])),tk,tp[:,t]) for t in range(256)])[np.maximum(loc,2)]*mask
                sn=area/math.fsum(ss);tn=area/math.fsum(tt);sc=np.cumsum(ss);sc=np.r_[0.,sc/sc[-1]];tc=np.cumsum(tt);tc=np.r_[0.,tc/tc[-1]];width=np.diff(sc);target=np.diff(tc);K=np.zeros((66,66))
                for dest in range(66):
                    for source in range(66):
                        if width[source]>0.:K[dest,source]=max(0.,min(tc[dest+1],sc[source+1])-max(tc[dest],sc[source]))/width[source]
                moved=K@positive;moved_random=K@active
                normalization_error=max(normalization_error,abs(sn-rows['source_normalization'][row]),abs(tn-rows['target_normalization'][row]))
                column_error=max(column_error,float(np.max(np.abs(K.sum(0)[mask]-1.))));template_error=max(template_error,float(np.max(np.abs(K@width-target))))
                displacement=float(np.sum(K*positive[None,:]*np.abs(np.arange(66)[:,None]-np.arange(66)[None,:]))/area);displacement_error=max(displacement_error,abs(displacement-rows['mean_absolute_mass_displacement_samples'][row]))
            expected_mean_delta=np.zeros(384);expected_mean_delta[110:176]=moved-positive
            expected_delta=np.zeros(384);expected_delta[110:176]=moved_random-active
            formula_error=max(formula_error,float(np.max(np.abs(old+expected_mean_delta-new))))
            coupling_formula_error=max(coupling_formula_error,float(np.max(np.abs(expected_delta-delta))))
            assert np.array_equal(new[:110],old[:110]) and np.array_equal(new[176:],old[176:]) and np.array_equal(new[old<=0.],old[old<=0.])
            assert np.array_equal(new[110:176][loc<2],old[110:176][loc<2]) and np.all(new[old>0.]>=0.)
            assert abs(math.fsum(new[110:366])-q[row])<1e-10 and abs(math.fsum(new-old))<1e-12
            assert np.argmax(new[112:144])+112==rows['new_mean_peak_index'][row]
            assert abs(area-rows['positive_area_Vns'][row])<1e-12 and np.sum(mask)==rows['active_positive_samples'][row]
            assert np.array_equal(delta[:110],np.zeros(110)) and np.array_equal(delta[176:],np.zeros(208)) and np.array_equal(delta[110:176][~mask],np.zeros((~mask).sum()))
            assert np.max(np.abs(ce['base_signal_V'][i]+delta-ce['coupled_signal_V'][i]))<1e-12
            assert np.max(np.abs(new+rnd-ce['base_signal_V'][i]))<1e-12
            for key,val in [('full_integral_residual_Vns',math.fsum(delta)),('canonical_Q_residual_Vns',math.fsum(delta[110:366])),('maximum_absolute_delta_V',np.abs(delta).max()),('old_random_L2_V',np.sqrt(np.sum(rnd*rnd))),('new_random_L2_V',np.sqrt(np.sum((rnd+delta)**2)))]:
                coupling_scalar_error=max(coupling_scalar_error,abs(float(val-coupling[key][row])))
            assert np.argmax(ce['base_signal_V'][i,112:144])+112==coupling['old_signal_peak_index'][row]
            assert np.argmax(ce['coupled_signal_V'][i,112:144])+112==coupling['new_signal_peak_index'][row]
        assert np.max(np.abs(ex['adapted_mean_pre_V']-ex['fixed_mean_pre_V']-ex['correction_pre_V']))<1e-12
        pre=ce['coupled_signal_V']+ce['noise_fast_V'];codes=np.round((np.clip(pre,-.5,.5)+.5)*16383);adc=codes/16383-.5;centered=adc-adc[:,:100].mean(1,keepdims=True);peaks=np.argmax(centered[:,112:144],1)+112
        wave=np.take_along_axis(centered,peaks[:,None]-16+np.arange(256)[None,:],1);f,g=readouts044(wave);native_rows=arrays[label+'_fast_NATIVE_ROWS.npz']
        assert np.array_equal(peaks,native_rows['native_peak_index'][ids])
        for key,val in f.items():
            assert np.array_equal(np.isnan(val),np.isnan(native_rows[key][ids]));coupling_feature_error=max(coupling_feature_error,float(np.nanmax(np.abs(val-native_rows[key][ids]))))
        for key,val in g.items():assert np.array_equal(val,native_rows[key+'_flag'][ids])
        coupling_examples_verified+=len(ids)
        carrier=arrays[label+'_fast_CARRIER_EXAMPLES.npz'];obs=arrays[label+'_fast_NATIVE_EXAMPLES.npz'];assert np.array_equal(carrier['mean_pre_V'],ex['adapted_mean_pre_V'])
        codes=np.round((np.clip(carrier['pre_ADC_V'],-.5,.5)+.5)*16383);adc=codes/16383-.5
        assert np.array_equal(codes,np.round((carrier['ADC_pre_V']+.5)*16383));adc_voltage_error=max(adc_voltage_error,float(np.max(np.abs(adc-carrier['ADC_pre_V']))))
        centered=adc-adc[:,:100].mean(1,keepdims=True);peaks=np.argmax(centered[:,112:144],1)+112;indices=peaks[:,None]-16+np.arange(256)[None,:]
        assert np.array_equal(indices,carrier['native_crop_indices']) and np.array_equal(peaks,obs['peak_index'])
        window_error=max(window_error,float(np.max(np.abs(np.take_along_axis(centered,indices,axis=1)-obs['wave']))))
        assert np.max(np.abs(obs['baseline100']-adc[:,:100]))<1e-12
        for h in ('fast','slow','common_level'):
            native_rows=arrays[label+'_'+h+'_NATIVE_ROWS.npz'];example=arrays[label+'_'+h+'_NATIVE_EXAMPLES.npz'];report=rec['metrics'][h]
            assert np.array_equal(native_rows['parent_row'],np.arange(6593)) and np.array_equal(native_rows['input_Q_Vns'],q) and np.array_equal(native_rows['input_Q_bin'],input_ref['input_Q_bin'])
            flags=arrays[label+'_'+h+'_OBSERVATION_FLAGS.npz'];assert np.array_equal(flags['parent_row'],np.arange(6593))
            expectedflags=dict(ADC_rail_hit=(flags['ADC_minimum_V']==-.5)|(flags['ADC_maximum_V']==.5),pre_ADC_out_of_range=(flags['pre_ADC_minimum_V']<-.5)|(flags['pre_ADC_maximum_V']>.5),
                peak_at_search_boundary=(native_rows['native_peak_index']==112)|(native_rows['native_peak_index']==143),global_peak_outside_search=(flags['global_peak_index']<112)|(flags['global_peak_index']>=144),
                Q_nonpositive=native_rows['Q_Vns']<=0.,PSD_undefined=native_rows['Q_Vns']==0.,PSD_guarded=np.abs(native_rows['Q_Vns'])<qfloor)
            for k,val in expectedflags.items():assert np.array_equal(val,flags[k]) and report['observation_flags'][k]==dict(count=int(val.sum()),denominator=6593),(h,k)
            for which in ('minimum','maximum'):
                codes=np.round((np.clip(flags['pre_ADC_'+which+'_V'],-.5,.5)+.5)*16383);assert np.max(np.abs(codes/16383-.5-flags['ADC_'+which+'_V']))<1e-15
            if h=='fast':
                for which,fun in [('minimum',np.min),('maximum',np.max)]:
                    assert np.array_equal(flags['pre_ADC_'+which+'_V'][:32],fun(carrier['pre_ADC_V'],axis=1))
                    assert np.array_equal(flags['ADC_'+which+'_V'][:32],fun(carrier['ADC_pre_V'],axis=1))
                centered=carrier['ADC_pre_V']-carrier['ADC_pre_V'][:,:100].mean(1,keepdims=True);assert np.array_equal(flags['global_peak_index'][:32],centered.argmax(1))
            base=arrays[f'seed{seed}_COUPLED067_'+h+'_NATIVE_ROWS.npz']
            assert np.array_equal(native_rows['COUPLED067_peak_index'],base['native_peak_index']) and np.array_equal(native_rows['COUPLED067_output_Q_bin'],base['native_output_Q_bin'])
            qq=native_rows['Q_Vns'];coord=np.where(qq>=qfloor,np.log(np.maximum(qq,qfloor)/qref),np.log(qfloor/qref)+np.arcsinh((qq-qfloor)/qfloor))
            assert np.array_equal(native_rows['native_output_Q_bin'],np.searchsorted(bounds,coord,side='left'))
            for field,right,left in [('FWHM_ns','right_crossing50_ns','left_crossing50_ns'),('rise10_90_ns','left_crossing90_ns','left_crossing10_ns')]:
                value=native_rows[right+'_absolute_ns']-native_rows[left+'_absolute_ns'];assert np.allclose(value,native_rows[field],rtol=0,atol=1e-12,equal_nan=True)
            for key in native_rows:
                if key.endswith('_flag'):assert report['fake_flags'][key[:-5]]['count']==int(native_rows[key].sum())
            for b,group in enumerate(report['output_Q_conditions']['bins']):
                mask=native_rows['native_output_Q_bin']==b;assert group['fake_rows']==int(mask.sum())
                for key in FIELDS044:
                    if key!='negative_lobe_area_Vns':assert summary(native_rows[key][mask])==group['fake'][key]
            f,g=readouts044(example['wave']);m=morphology018(example['wave'])
            for key,val in f.items():feature_error=max(feature_error,float(np.nanmax(np.abs(val-native_rows[key][:32]))))
            for key,val in g.items():assert np.array_equal(val,native_rows[key+'_flag'][:32])
            for key in CROSSINGS053:
                saved=native_rows[key+'_absolute_ns'][:32];value=m[key]+example['peak_index']-16;assert np.array_equal(np.isnan(saved),np.isnan(value));crossing_error=max(crossing_error,float(np.nanmax(np.abs(saved-value))))
            for key in ('noise','baseline_mean_V','baseline_RMS_V'):assert report['values'][key]==m0[h]['values'][key]
        ownpeak=np.argmax(ex['adapted_mean_pre_V'][:,112:144],1)+112;own=np.take_along_axis(ex['adapted_mean_pre_V'],ownpeak[:,None]-16+np.arange(256)[None,:],1);f,g=readouts044(own)
        for key,val in f.items():feature_error=max(feature_error,float(np.nanmax(np.abs(val-intrinsic[key][:32]))))
        for key,val in g.items():assert np.array_equal(val,intrinsic[key+'_flag'][:32])
    assert max(formula_error,normalization_error,adc_voltage_error,window_error,feature_error,crossing_error)<1e-12
    
    assert max(column_error,template_error,displacement_error)<1e-12
inputs=dict(np.load(r/p['input_reference_file']));qfloor=files['run_01/REAL_PROFILE_DEFINITION.json']['q_floor']
profile_audit=[];max_moment_error=0.;max_linear_error=0.;max_cov_error=0.;max_parity_error=0.
labels=['REAL']+[f'seed{seed}_{case}_{kind}' for seed in range(2) for case in ('COUPLED067','VOLTAGEWARP') for kind in ('fast','intrinsic')]
for label in labels:
    rows=arrays[label+'_PROFILES.npz'];definition=files['run_01/'+label+'_PROFILE_DEFINITION.json'];N=21047 if label=='REAL' else 6593
    assert np.array_equal(rows['parent_row'],np.arange(N)) and rows['mean'].shape==(len(rows['group_bin']),2,8,256)
    assert definition['parent_rows']==N and definition['group_cells']==len(rows['group_bin'])*16
    if label=='REAL':
        feature=arrays['REAL_ROWS.npz'];bin_arrays=[feature['output_Q_bin']]
    elif label.endswith('_intrinsic'):
        feature=arrays[label.replace('_intrinsic','_INTRINSIC_MEAN_ROWS')+'.npz'];bin_arrays=[inputs['input_Q_bin'],arrays[label[:5]+'_COUPLED067_fast_NATIVE_ROWS.npz']['native_output_Q_bin']]
    else:
        feature=arrays[label+'_NATIVE_ROWS.npz'];bin_arrays=[inputs['input_Q_bin'],feature['native_output_Q_bin']]
    assert np.array_equal(rows['group_bin'],np.array(bin_arrays))
    assert np.array_equal(rows['counts'],np.array([[np.sum(bins==b) for b in range(8)] for bins in bin_arrays]))
    assert np.array_equal(rows['counts'].sum(1),np.full(len(bin_arrays),N)) and np.array_equal(rows['counts'],rows['parity_counts'].sum(2))
    assert np.array_equal(rows['counts'],definition['group_counts']) and np.array_equal(rows['parity_counts'],definition['parity_counts'])
    for g,bins in enumerate(bin_arrays):
        weights=rows['counts'][g]/N
        for view in (0,1):
            mean=rows['mean'][g,view];SD=rows['SD_population'][g,view];overall=rows['overall_mean'][view];means64=mean[:,:64]-overall[:64]
            reconstructed=np.einsum('b,bt->t',weights,mean)
            variance=np.einsum('b,bt->t',weights,SD**2+(mean-overall)**2)
            total_cov=np.einsum('b,bij->ij',weights,rows['covariance_first64'][g,view]+np.einsum('bi,bj->bij',means64,means64))
            blockmean=rows['block_integral_mean'][g,view];block_overall=overall.reshape(16,16).sum(1);offset=blockmean-block_overall
            blockcov=np.einsum('b,bij->ij',weights,rows['block_integral_covariance'][g,view]+np.einsum('bi,bj->bij',offset,offset))
            max_moment_error=max(max_moment_error,float(np.max(np.abs(reconstructed-overall))),float(np.max(np.abs(variance-rows['overall_SD_population'][view]**2))))
            max_cov_error=max(max_cov_error,float(np.max(np.abs(total_cov-rows['overall_covariance_first64'][view]))),float(np.max(np.abs(blockcov-rows['overall_block_integral_covariance'][view]))))
            for b in range(8):
                mask=bins==b;n=int(mask.sum());counts=rows['parity_counts'][g,b]
                assert np.array_equal(counts,[np.sum(mask&(np.arange(N)%2==parity)) for parity in (0,1)])
                merged=np.einsum('p,pt->t',counts/n,rows['parity_mean'][g,view,b]);max_parity_error=max(max_parity_error,float(np.max(np.abs(merged-mean[b]))))
                max_moment_error=max(max_moment_error,float(np.max(np.abs(rows['SE_IID_formula'][g,view,b]-SD[b]/np.sqrt(n)))))
                max_cov_error=max(max_cov_error,float(np.max(np.abs(np.diag(rows['covariance_first64'][g,view,b])-SD[b,:64]**2))))
                assert np.max(np.abs(rows['covariance_first64'][g,view,b]-rows['covariance_first64'][g,view,b].T))<1e-12
                assert np.max(np.abs(rows['block_integral_covariance'][g,view,b]-rows['block_integral_covariance'][g,view,b].T))<1e-12
                max_linear_error=max(max_linear_error,float(np.max(np.abs(mean[b].reshape(16,16).sum(1)-blockmean[b]))))
                if view==0:
                    max_linear_error=max(max_linear_error,abs(float(mean[b].sum()-feature['Q_Vns'][mask].mean())),abs(float(mean[b,:46].sum()-feature['front_Vns'][mask].mean())),
                        abs(float(mean[b,46:].sum()-feature['tail_Vns'][mask].mean())),abs(float(mean[b,16]-feature['peak_voltage_V'][mask].mean())),
                        abs(float(rows['block_integral_covariance'][g,view,b].sum()-feature['Q_Vns'][mask].var())),
                        abs(float(rows['covariance_first64'][g,view,b,:46,:46].sum()-feature['front_Vns'][mask].var())))
                else:
                    qq=feature['Q_Vns'][mask];denom=np.where(qq<0.,-1.,1.)*np.maximum(np.abs(qq),qfloor)
                    max_linear_error=max(max_linear_error,abs(float(mean[b].sum()-np.mean(qq/denom))),abs(float(mean[b,16]-np.mean(feature['peak_voltage_V'][mask]/denom))))
                assert rows['Q_nonpositive_counts'][g,b]==np.sum(feature['Q_Vns'][mask]<=0.) and rows['Q_guarded_counts'][g,b]==np.sum(np.abs(feature['Q_Vns'][mask])<qfloor)
    profile_audit.append(dict(label=label,rows=N,group_view_cells=len(bin_arrays)*16,group_count_min=int(rows['counts'].min()),parity_count_min=int(rows['parity_counts'].min()),moments_and_scalar_reconstruction_verified=True))
assert len(profile_audit)==9 and sum(x["rows"] for x in profile_audit)==73791 and sum(x["group_view_cells"] for x in profile_audit)==272
assert max(max_moment_error,max_linear_error,max_cov_error,max_parity_error)<1e-10
assert coupling_examples_verified==156 and max(coupling_formula_error,coupling_scalar_error,coupling_feature_error,amplitude_scale_error)<1e-12 and query_error<1e-10
from scipy.stats import wasserstein_distance
W1audit=[];real=arrays['REAL_ROWS.npz']
assert np.isnan(real['rise10_90_ns']).sum()==94 and np.isnan(real['FWHM_ns']).sum()==5
for seed in range(2):
    for case in ('COUPLED067','VOLTAGEWARP'):
        for h in ('fast','slow','common_level'):
            feature=arrays[f'seed{seed}_{case}_{h}_NATIVE_ROWS.npz'];m=files[f'run_01/seed{seed}_{case}.json']['metrics'][h]
            sf={k:feature[k] for k in FIELDS044};rf={k:real[k] for k in FIELDS044}
            for data in (sf,rf):
                data['PSD_raw']=data['tail_Vns']/data['Q_Vns'];data['PSD_guarded']=data['tail_Vns']/(np.where(data['Q_Vns']<0.,-1.,1.)*np.maximum(np.abs(data['Q_Vns']),qfloor))
            for k in sf:
                a=sf[k][np.isfinite(sf[k])];b=rf[k][np.isfinite(rf[k])];saved=m['values'][k];dist=float(wasserstein_distance(a,b))
                support=np.r_[a,b];order=np.argsort(support,kind='stable');gap=np.diff(support[order]);mass=np.r_[np.full(len(a),len(b),np.int64),np.full(len(b),-len(a),np.int64)];cdf=np.cumsum(mass[order],dtype=np.int64)[:-1]
                independent=math.fsum((gap*np.abs(cdf)/(len(a)*len(b))).tolist());span=float(support.max()-support.min());eps=np.finfo(float).eps
                assert abs(dist-independent)<=8*eps*span and abs(saved-independent)<=8*eps*len(support)*span,(case,h,k,saved,independent)
                W1audit.append(dict(case=case,hypothesis=h,field=k,native=saved,scipy=dist,integer_CDF_fsum=independent,absolute_native_error=abs(saved-independent)))
assert len(W1audit)==108
for name,digest in p['source_pins'].items():assert sha(r/name)==digest,name
manifest=read(w/'collected/SOURCE_MANIFEST.json');assert manifest['config_sha256']==sha(r/'configs/response068_20261018.json')
for name,meta in manifest['files'].items():assert (r/name).stat().st_size==meta['bytes'] and sha(r/name)==meta['sha256'],name
assert v['terminal_source_verification']['all_hashes_verified'] and v['terminal_source_verification']['source_files']==len(manifest['files'])==445
points=all(x['decision']['feasible'] and all(u['passes_point_targets'] for u in x['point_usefulness'].values()) for x in result['results'])
assert points==result['original_point_targets_passed'] and result['stage_outcome']==('FIT_USEFUL_MEAN_PENDING_CONFIRMATION' if points else 'CLOSED_NEGATIVE_PEAK_ANCHORED_VOLTAGE_WARP')
record=dict(status='PASS',UTC=datetime.now(timezone.utc).isoformat(),job_id=p['run_id'],source_pins_verified=len(p['source_pins']),deployed_files_verified=len(manifest['files']),execution_counts=result['execution_counts'],original_native_H_JSON_exact067=12,full_native_H_verified=18,original067_baseline_arrays_exact=True,
    geometry_rows_verified=26372,native_feature_rows_verified=79116,intrinsic_rows=26372,mean_examples_independently_verified=156,original_mean_EX_consistency_rows=128,native_examples_verified=384,fast_carrier_examples_verified=128,coupling_rows_verified=26372,coupling_examples_independently_verified=156,candidate_query_rows_independently_reconstructed=13186,
    formula_max_error_V=formula_error,template_normalization_max_error=normalization_error,independent_column_mass_max_error=column_error,independent_template_mapping_max_error=template_error,independent_mass_displacement_max_error_samples=displacement_error,query_max_error_samples=query_error,amplitude_scale_max_error=amplitude_scale_error,observation_flag_tables_verified=12,observation_flag_rows_verified=79116,ADC_integer_codes_exact=True,ADC_voltage_max_error_V=adc_voltage_error,window_max_error_V=window_error,example_feature_max_error=feature_error,example_crossing_max_error_ns=crossing_error,
    coupling_formula_max_error_V=coupling_formula_error,coupling_scalar_max_error=coupling_scalar_error,coupling_native_feature_max_error=coupling_feature_error,W1_CDF_checked=len(W1audit),W1_max_native_error=max(x['absolute_native_error'] for x in W1audit),W1_audit=W1audit,
    profiles=profile_audit,profile_moment_max_error=max_moment_error,profile_scalar_linear_max_error=max_linear_error,profile_covariance_reconstruction_max_error=max_cov_error,profile_parity_reconstruction_max_error=max_parity_error,
    original_point_targets_passed=points,stage_outcome=result['stage_outcome'],full_random_response_acceptance=False,frozen_A0_M0_005_verified=True,uncertainty_scope='All saved rows,query maps andaggregate moments,156 mean/random examples plus128 fast-carrier and384 native windows checked; unsaved full-population waves not reread.')
(w/'ARTIFACT_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('W1_audit','profiles','execution_counts')}))
