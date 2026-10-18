"""Offline verification against saved BASE054 rows and independent NumPy examples."""
import json,math
import numpy as np
from run_response044 import FIELDS044
from wp1_evaluation import summary


def verify_diagnostics071(root,arrays,files,seed):
    prior=root/'receipts/response060_20261018/wake_20261018_1605_01/collected/run_01'
    read=lambda p:json.loads(p.read_text());load=lambda p:dict(np.load(p,allow_pickle=False))
    report=read(prior/f'seed{seed}_BASE054.json');audit=[];max_error=0.;example_count=0
    for h in ('fast','slow','common_level'):
        label=f'seed{seed}_SPLITWARP_{h}';d=arrays[label+'_DECOMPOSITION_ROWS.npz'];meta=files['run_01/'+label+'_DECOMPOSITION.json']
        n=arrays[label+'_NATIVE_ROWS.npz'];b=load(prior/f'seed{seed}_BASE054_{h}_NATIVE_ROWS.npz');g=arrays[f'seed{seed}_SPLITWARP_GEOMETRY_ROWS.npz'];c=arrays[f'seed{seed}_SPLITWARP_COUPLING_ROWS.npz']
        assert len(d)==52 and meta['rows']==6593 and meta['numeric_columns']==43 and meta['boolean_columns']==9
        assert np.array_equal(d['parent_row'],np.arange(6593)) and np.array_equal(d['reference_peak'],g['old_mean_peak_index'])
        assert np.array_equal(d['BASE054_native_peak_index'],b['native_peak_index']) and np.array_equal(d['BASE054_native_output_Q_bin'],b['native_output_Q_bin'])
        assert np.array_equal(d['candidate_native_peak_index'],n['native_peak_index']) and np.array_equal(d['coupled_signal_peak_index'],c['new_signal_peak_index']) and np.array_equal(d['mean_only_signal_peak_index'],c['old_signal_peak_index'])
        for key in list(FIELDS044)+[key for key in b if key.endswith('_flag')]:assert np.array_equal(d['BASE054_'+key],b[key],equal_nan=True),(seed,h,key)
        assert meta['BASE054_observation_flags']==report['metrics'][h]['observation_flags']
        for key,val in d.items():
            if key!='parent_row':assert summary(val)==meta['row_summaries'][key],(seed,h,key)
        closure=0.
        for name in ('Q','tail'):
            total=d[name+'_BASE054_window_shift_Vns']+sum(d[name+'_same_new_window_'+key+'_Vns'] for key in ('mean','random','ADC','reassociation'))
            closure=max(closure,float(np.abs(d[name+'_native_delta_Vns']-total).max()),float(np.abs(d[name+'_native_delta_Vns']-(n[name+'_Vns']-b[name+'_Vns'])).max()),float(np.abs(d[name+'_closure_Vns']).max()))
        terms=sum(d['peak_advantage_'+name+'_V'] for name in ('BASE054_mean','mean_change','original_random','random_change','electronic_noise','candidate_ADC_error','candidate_reassociation'))
        peak_closure=float(np.abs(terms-d['peak_advantage_candidate_ADC_V']).max())
        assert closure<1e-12 and peak_closure<1e-12 and np.min(d['peak_advantage_candidate_ADC_V'])>=0. and np.max(d['peak_advantage_BASE054_ADC_V'])<=0.
        assert np.max(np.abs(d['tail_reference_mean_Vns']))<1e-12 and np.max(np.abs(d['tail_reference_random_Vns']))<1e-12
        audit.append(dict(seed=seed,hypothesis=h,rows=6593,BASE054_native_rows_flags_and_peaks_exact060=True,all_summaries_exact=True,integral_closure_max_error_Vns=closure,peak_closure_max_error_V=peak_closure,reference_mean_tail_residual_max_Vns=float(np.abs(d['tail_reference_mean_Vns']).max()),reference_random_tail_residual_max_Vns=float(np.abs(d['tail_reference_random_Vns']).max())))
        if h!='fast':continue
        e=arrays[f'seed{seed}_SPLITWARP_COUPLING_EXAMPLES.npz']
        for i,row in enumerate(e['parent_row']):
            row=int(row);fixed=e['fixed_mean_pre_V'][i];mean=e['adapted_mean_pre_V'][i];rnd=e['old_random_V'][i];dr=e['random_correction_V'][i];noise=e['noise_fast_V'][i];signal=e['coupled_signal_V'][i]
            old=fixed+rnd+noise;new=signal+noise;dm=mean-fixed;reassoc=(new-old)-dm-dr
            oa=np.round((np.clip(old,-.5,.5)+.5)*16383)/16383-.5;na=np.round((np.clip(new,-.5,.5)+.5)*16383)/16383-.5
            op=int(np.argmax(oa[112:144])+112);npk=int(np.argmax(na[112:144])+112);ref=int(np.argmax(fixed[112:144])+112)
            for key,value in [('reference_peak',ref),('BASE054_native_peak_index',op),('candidate_native_peak_index',npk),('BASE054_signal_peak_index',int(np.argmax((fixed+rnd)[112:144])+112)),('mean_only_signal_peak_index',int(np.argmax(e['base_signal_V'][i,112:144])+112)),('coupled_signal_peak_index',int(np.argmax(signal[112:144])+112)),('BASE054_preADC_peak_index',int(np.argmax(old[112:144])+112)),('candidate_preADC_peak_index',int(np.argmax(new[112:144])+112))]:assert d[key][row]==value,(seed,row,key)
            for name,lo,length in [('tail',30,210),('Q',-16,256)]:
                oi=slice(op+lo,op+lo+length);ni=slice(npk+lo,npk+lo+length);ri=slice(ref+lo,ref+lo+length)
                expected={name+'_native_delta_Vns':math.fsum(na[ni])-math.fsum(oa[oi]),name+'_BASE054_window_shift_Vns':math.fsum(oa[ni])-math.fsum(oa[oi])}
                for key,value in [('mean',dm),('random',dr),('ADC',(na-new)-(oa-old)),('reassociation',reassoc)]:expected[name+'_same_new_window_'+key+'_Vns']=math.fsum(value[ni])
                if name=='tail':
                    for key,value in [('mean',dm),('random',dr),('ADC',(na-new)-(oa-old))]:expected['tail_reference_'+key+'_Vns']=math.fsum(value[ri])
                for key,value in expected.items():max_error=max(max_error,abs(d[key][row]-value))
            for name,value in [('BASE054_mean',fixed),('mean_change',dm),('original_random',rnd),('random_change',dr),('electronic_noise',noise),('candidate_ADC_error',na-new),('candidate_reassociation',new-mean-rnd-dr-noise),('BASE054_ADC',oa),('candidate_ADC',na)]:max_error=max(max_error,abs(d['peak_advantage_'+name+'_V'][row]-(value[npk]-value[op])))
            example_count+=1
    assert example_count==39 and max_error<1e-12
    return dict(head=seed,diagnostic_tables=audit,rows_verified=19779,independent_fast_examples=example_count,independent_example_max_error=max_error,scope='All saved decomposition rows: old BASE054 feature/flag/peak replay, summaries and algebraic closure. Full independent NumPy waveform attribution for39 stored fast examples. Unsaved full-population carriers not independently reconstructed.')
