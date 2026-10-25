"""Independent readback of the frozen finite-fit endpoint and native artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np

def verify(config_path,out):
 p=json.loads(Path(config_path).read_text());out=Path(out)
 d=json.loads((out/'NATIVE_CANDIDATE_RESULT.json').read_text());n=json.loads((out/'RESULT.json').read_text())
 assert n['status']=='COMPLETE' and n['config_sha256']==hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
 assert d['status']=='COMPLETE_ONE_FIXED_SHAPE_ADDITIVE_CANDIDATE_NATIVE_READBACK'
 exit_record=json.loads((out.parent/'control_native/exit_receipt.json').read_text())
 assert exit_record['exit_code']==0 and exit_record['peak_rss_bytes']<=12*1024**3
 assert n['elapsed_seconds']<300 and n['peak_GPU_reserved_bytes']<4*1024**3
 assert not d['scientific_acceptance'] and not d['endpoint_adopted'] and d['REPORT_rows_used']==0 and d['optimizer_updates']==0
 assert d['parameter_fits']==0 and d['candidate_selections']==0
 fit_result=json.loads(Path(p['source_fit_result']).read_bytes())
 assert fit_result['status']=='COMPLETE_ONE_ACTUAL_FINITE_NATIVE_FIT' and fit_result['eligible_for_one_native_readback']
 assert fit_result['job_number']==p['source_fit_job']
 assert d['mean_relative_RMS_max']<=.01+1e-12
 assert set(n['execution_counts'])==set(p['operation_budget'])
 for k,v in p['operation_budget'].items():assert n['execution_counts'][k]<=v,k
 with np.load(out/'FIXED_CANDIDATE_POINT.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 with np.load(p['source_joint_solution'],allow_pickle=False) as z:
  np.testing.assert_array_equal(a['joint_coefficients'],z['primal']);np.testing.assert_array_equal(a['coordinate_steps'],z['coordinate_steps'])
  np.testing.assert_array_equal(a['head_delta'],z['primal'][:161]*z['coordinate_steps'][:161])
  np.testing.assert_array_equal(a['NN_coefficients'],z['primal'][161:]*z['coordinate_steps'][161:])
  np.testing.assert_allclose(a['predicted_relative_changes'],z['native']@z['primal'],atol=1e-14,rtol=0)
 np.testing.assert_array_equal(a['active_coordinates'],np.empty(0,dtype=np.int64))
 with np.load(p['source_mean_tangent'],allow_pickle=False) as z:np.testing.assert_array_equal(a['original_head'],z['initial_head'])
 with np.load(p['source_parameters'],allow_pickle=False) as z:np.testing.assert_array_equal(a['original_theta'],z['original_theta'])
 np.testing.assert_array_equal(a['theta'],a['original_theta']);np.testing.assert_array_equal(a['frozen_coordinates'],np.arange(54));assert float(a['factor'])==1
 assert str(a['source_NN_head_SHA256'])==p['active_inputs'][p['source_additive_head']]['sha256']
 with np.load(out/'MEMORY_AUDIT_ARRAYS.npz',allow_pickle=False) as z:m={k:z[k].copy() for k in z.files}
 with np.load(p['source_memory'],allow_pickle=False) as z:
  for k in ['target','weights','joint_cholesky','wave_denominator','wave_supported']:np.testing.assert_array_equal(m[k],z[k])
 levels=p['memory']['levels'];cell_count=sum(levels);thresholds=p['memory']['thresholds'];at=0;sections={}
 for key,length in [('Q_mass',cell_count),('PSD_CDF',cell_count*thresholds),('Q_tail_CDF',cell_count*thresholds),('undefined_mass',3),('native_joint',272),('Q_wave',cell_count*56)]:sections[key]=slice(at,at+length);at+=length
 cell_names=[f'Q_wave_L{k}_{j:02d}' for k in levels for j in range(k)]
 supported=[name for name,ok in zip(cell_names,m['wave_supported']) if ok]
 assert len(supported)==92 and at==len(m['target'])
 def metrics(x):
  avg=x.mean(0);residual=avg-m['target'];product=m['weights']*residual**2
  result={name:float(product[s].sum()) for name,s in sections.items()}
  raw=m['joint_cholesky']@residual[sections['native_joint']];variance=(m['joint_cholesky']**2).sum(1)
  for i,name in enumerate(['wave','charge_morphology','peak_codes','baseline']):result['block_'+name]=float((raw[i*56:(i+1)*56]**2/variance[i*56:(i+1)*56]).mean())
  result['block_joint']=float((raw[224:272]**2/variance[224:272]).mean())
  local=(residual[sections['Q_wave']].reshape(-1,56)/m['wave_denominator'][:,None])**2;result.update(zip(cell_names,local.mean(1)))
  real_mass=m['target'][sections['Q_mass']];mass=avg[sections['Q_mass']];s=sections['PSD_CDF']
  conditional=real_mass[:,None]*(avg[s].reshape(-1,thresholds)/mass[:,None]-m['target'][s].reshape(-1,thresholds)/real_mass[:,None])
  result['fixed_REAL_conditional_PSD']=float((m['weights'][s].reshape(-1,thresholds)*conditional**2).sum())
  return result
 recomputed={};errors=[]
 for case in ['original','inherited','candidate']:
  recomputed[case]={'metrics':metrics(m[case]),'metrics_by_bank':[metrics(np.stack((v,v))) for v in m[case]]}
  for k,v in recomputed[case]['metrics'].items():errors.append(abs(v-d['records'][case]['metrics'][k]))
  for bank in range(2):
   for k,v in recomputed[case]['metrics_by_bank'][bank].items():errors.append(abs(v-d['records'][case]['metrics_by_bank'][bank][k]))
 assert max(errors)<1e-9
 actual_checks={}
 for bank in range(2):
  inherited=recomputed['inherited']['metrics_by_bank'][bank]
  for case in ['original','candidate']:
   values=recomputed[case]['metrics_by_bank'][bank];values['local_excess_penalty']=float(np.mean([max(values[name]/(1.02*inherited[name])-1,0)**2 for name in supported]))
  origin=recomputed['original']['metrics_by_bank'][bank];candidate=recomputed['candidate']['metrics_by_bank'][bank]
  for name in p['risk_names']:actual_checks[f'bank{bank}/{name}']=bool(candidate[name]<=origin[name]+1e-10*max(1,abs(origin[name])))
 with np.load(out/'CANDIDATE_HARD_READOUT.npz',allow_pickle=False) as z:candidate={k:z[k].copy() for k in z.files}
 with np.load(p['source_fit_cached_readout'],allow_pickle=False) as z:
  assert set(z.files)==set(candidate)
  for key,value in candidate.items():np.testing.assert_array_equal(value,z[key])
 indices=candidate['global_generated_indices'];parents=(indices%(6593*8))//8
 with np.load(p['source_parameters'],allow_pickle=False) as z:fit=z['SOURCE_FIT_rows'].copy();real_fit=z['REAL_FIT_rows'].copy()
 assert len(fit)==6465 and len(real_fit)==18948 and set(parents)==set(fit) and len(np.unique(indices))==103440
 np.testing.assert_array_equal(np.bincount(parents,minlength=6593)[fit],np.full(6465,16))
 with np.load(p['original_readout'],allow_pickle=False) as z:original={k:z[k][indices].copy() for k in z.files if k!='source_parent_rows'}
 np.testing.assert_array_equal(candidate['prefix_adc_codes'],original['prefix_adc_codes']);np.testing.assert_array_equal(candidate['baseline_mean_V'],original['baseline_mean_V'])
 for bank in range(2):
  select=slice(bank*51720,(bank+1)*51720)
  assert np.all(indices[select]//(6593*8)==bank)
  np.testing.assert_array_equal(np.bincount(parents[select],minlength=6593)[fit],np.full(6465,8))
  for case,arrays in [('original',original),('candidate',candidate)]:
   assert set(d['records'][case]['flags'][bank])=={k[5:] for k in arrays if k.startswith('flag_')} and len(d['records'][case]['flags'][bank])==7
   for key,value in d['records'][case]['flags'][bank].items():assert int(arrays['flag_'+key][select].sum())==value
  actual_checks[f'bank{bank}/flags']=all(d['records']['candidate']['flags'][bank][key]<=value for key,value in d['records']['original']['flags'][bank].items())
 assert actual_checks==d['common_checks']
 cdf_errors=[]
 for case in ['original','candidate']:
  path=p['original_CDF'] if case=='original' else out/'INDEPENDENT_CDF_candidate.npz'
  with np.load(path,allow_pickle=False) as z:
   weights=np.zeros(8776);visited=[]
   for name,(start,stop) in d['independent'][case]['wave_slices'].items():
    kind,*parts=name.split(':');width=stop-start;visited.extend(range(start,stop))
    if kind=='swd':assert width==32;weight=.5/(6*width)
    elif kind=='coordinate':assert width==256;weight=.25/(2*width)
    else:assert width==(256-int(parts[1]))*4;weight=.25/(8*width)
    weights[start:stop]=weight
   np.testing.assert_array_equal(np.sort(visited),np.arange(8776));np.testing.assert_array_equal(weights,z['wave_weights'])
   qweights=np.r_[np.full(2,.25/2),np.full(16,.25/16),np.full(8,.5/8)];np.testing.assert_array_equal(qweights,z['QPSD_weights'])
   for key,value in [('wave',float(z['wave_columns']@weights)),('QPSD',float(z['QPSD_columns']@qweights))]:
    cdf_errors.extend([abs(value-d['independent'][case][key]),abs(value-d['native'][case]['values'][key])])
 assert max(cdf_errors)<1e-9
 saved=out/'FROZEN_TOLERANCES.json';assert hashlib.sha256(saved.read_bytes()).hexdigest()==p['active_inputs'][p['tolerance_file']]['sha256']
 tolerances=json.loads(saved.read_text())['T_by_hypothesis']['fast'];expected_guards={}
 for key,value in tolerances.items():
  if value is not None and value>0:
   new,old=d['native']['candidate']['values'][key],d['native']['original']['values'][key]
   expected_guards[key]={'status':'undefined' if new is None or old is None else 'defined','threshold':value,'noninferior':None if new is None or old is None else new-old<=value}
 assert expected_guards==d['original_calibrated_guard_checks']
 changes={k:d['native']['candidate']['values'][k]/d['native']['original']['values'][k]-1 for k in ['wave','QPSD','noise']}
 for k in ['wave','QPSD']:assert abs(changes[k]-fit_result['actual_relative_changes'][k])<1e-9
 for k,v in changes.items():assert abs(v-d['native_relative_changes'][k])<1e-14
 gate=changes['wave']<=-.02 and changes['QPSD']<=0 and all(actual_checks.values()) and all(v['status']=='defined' and v['noninferior'] for v in expected_guards.values())
 assert bool(gate)==d['original_entry_conditions_satisfied']
 result={'status':'PASS_FINITE_FIT_NATIVE_CDF_GUARDS_FIT_ROLES','UTC':datetime.now(timezone.utc).isoformat(),'max_feature_recompute_error':max(errors),'max_full_CDF_recompute_error':max(cdf_errors),'native_relative_changes':changes,'original_entry_passed':bool(gate),'fit_cache_native_arrays_exact':True,'mean_relative_RMS_max':d['mean_relative_RMS_max'],'SOURCE_FIT_parents':6465,'REPORT_rows_used':0,'scientific_acceptance':False,'endpoint_adopted':False,'scope':'Independent scalar,feature,CDF-column,role and guard readback of the one frozen fit endpoint'}
 dest=out.parent/'INDEPENDENT_NATIVE_VERIFICATION.json'
 with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();print(json.dumps(verify(a.config,a.out)))
