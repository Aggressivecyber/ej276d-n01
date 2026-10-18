from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
from scipy.stats import spearmanr,wasserstein_distance
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;o=w/'collected/run_01';sys.path[:0]=[str(r/'scripts'),str(r/'src')]
from run_response044 import FIELDS044
from run_response053 import CROSSINGS053
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(w/'ARTIFACT_VERIFICATION.json')['status']=='PASS'
real=dict(np.load(o/'REAL_ROWS.npz'));real_width=real['FWHM_ns'];real_bins=real['output_Q_bin'];records=[];crossing_support=[];paired_stage=[];case_contrasts=[]
quantiles=np.array([0.,.01,.05,.1,.25,.5,.75,.9,.95,.99,1.]);native_tables={};stage_tables={};intrinsic_tables={}
for seed in (0,1):
 for case in ('BASE054','PROBE_1_neg','PROBE_1_pos'):
  stage_tables[seed,case]=dict(np.load(o/f'seed{seed}_{case}_fast_STAGE_ROWS.npz'))
  intrinsic_tables[seed,case]=dict(np.load(o/f'seed{seed}_{case}_INTRINSIC_MEAN_ROWS.npz'))
  for h in ('fast','slow','common_level'):native_tables[seed,case,h]=dict(np.load(o/f'seed{seed}_{case}_{h}_NATIVE_ROWS.npz'))
tables=[dict(seed=seed,case=case,hypothesis=h,stage='native',table=z) for (seed,case,h),z in native_tables.items()]
for (seed,case),z in intrinsic_tables.items():
 tables.append(dict(seed=seed,case=case,hypothesis='intrinsic_no_random',stage='intrinsic_mean',table=dict(z,native_output_Q_bin=native_tables[seed,case,'fast']['native_output_Q_bin'],BASE054_output_Q_bin=native_tables[seed,case,'fast']['BASE054_output_Q_bin'])))
for item in tables:
 z=item['table'];meta={k:v for k,v in item.items() if k!='table'};width=z['FWHM_ns'];valid=np.isfinite(width);N=len(width);n=int(valid.sum());mean=float(width[valid].mean());var=float(width[valid].var())
 peak=z['native_peak_index'] if item['stage']=='native' else z['intrinsic_peak_index']
 suffix='_absolute_ns' if item['stage']=='native' else ''
 all_fields={**{k:z[k] for k in FIELDS044},**{k+'_relative_peak_ns':z[k+suffix]-peak for k in CROSSINGS053}}
 q=z['Q_Vns'];correlations={}
 for label,xx in (('own_output_Q',q),('A0_input_Q',z['input_Q_Vns'])):
  correlation=spearmanr(xx[valid],width[valid]);correlations[label]=dict(rho=float(correlation.statistic),defined_rows=n,total_rows=N)
 marginal=dict(defined_rows=n,total_rows=N,mean_ns=mean,variance_ns2=var,quantiles_ns=np.quantile(width[valid],quantiles).tolist(),
  W1_to_REAL_ns=float(wasserstein_distance(width[valid],real_width[np.isfinite(real_width)])),correlations=correlations)
 for group_key in ('input_Q_bin','BASE054_output_Q_bin','native_output_Q_bin'):
  group=z[group_key];within=0.;between=0.;bins=[]
  for b in range(8):
   mask=group==b;v=mask&valid;nb=int(v.sum());mu=float(width[v].mean());variance=float(width[v].var());weight=nb/n
   within+=weight*variance;between+=weight*(mu-mean)**2
   fields={}
   for key,xx in all_fields.items():
    finite=mask&np.isfinite(xx);values=xx[finite]
    fields[key]=dict(total_rows=int(mask.sum()),defined_rows=int(finite.sum()),undefined_rows=int((mask&~np.isfinite(xx)).sum()),
     mean=float(values.mean()),SD=float(values.std()),quantiles=np.quantile(values,quantiles).tolist())
   counterpart=(real_bins==b)&np.isfinite(real_width)
   source_real=dict(comparable_output_Q_conditions=group_key!='input_Q_bin',SOURCE_REAL_event_pairing=False,
    REAL_total_rows=int(np.sum(real_bins==b)),REAL_defined_rows=int(counterpart.sum()),REAL_width_mean_ns=float(real_width[counterpart].mean()),
    width_W1_to_same_REAL_output_bin_ns=float(wasserstein_distance(width[v],real_width[counterpart])) if group_key!='input_Q_bin' else None,
    width_mean_gap_to_same_REAL_output_bin_ns=mu-float(real_width[counterpart].mean()) if group_key!='input_Q_bin' else None)
   bins.append(dict(bin=b,total_rows=int(mask.sum()),defined_rows=nb,variance_ns2=variance,fields=fields,source_real=source_real))
  assert abs(var-within-between)<1e-11
  records.append(dict(**meta,grouping=group_key,marginal=marginal,bins=bins,within_variance_ns2=within,between_variance_ns2=between,
   variance_closure_error_ns2=var-within-between,grouping_note='Intrinsic native_output_Q_bin refers to paired fast native case membership, not intrinsic own-Q.' if item['stage']=='intrinsic_mean' else 'Original memberships; no event matching to REAL.'))
 for group_key in ('input_Q_bin','BASE054_output_Q_bin','native_output_Q_bin'):
  group=z[group_key]
  for b in range(8):
   mask=group==b;cross={}
   for key in CROSSINGS053:
    value=z[key+suffix];good=mask&np.isfinite(value);count=int(good.sum())
    cross[key]=dict(total_rows=int(mask.sum()),defined_rows=count,undefined_rows=int((mask&~np.isfinite(value)).sum()),
     below110_rows=int(np.sum(good&(value<110))),inside110_142_rows=int(np.sum(good&(value>=110)&(value<142))),
     at_or_beyond142_rows=int(np.sum(good&(value>=142))),within1_of110_rows=int(np.sum(good&(np.abs(value-110)<=1))),
     within1_of141_rows=int(np.sum(good&(np.abs(value-141)<=1))),absolute_quantiles_ns=np.quantile(value[good],quantiles).tolist())
   flags={k:int(np.sum(z[k][mask])) for k in z if k.endswith('_flag')}
   crossing_support.append(dict(**meta,grouping=group_key,bin=b,total_rows=int(mask.sum()),crossings=cross,flags=flags))
real_bin_records=[];rv=real_width[np.isfinite(real_width)].var();rm=real_width[np.isfinite(real_width)].mean();rn=np.isfinite(real_width).sum();rwithin=0.;rbetween=0.
for b in range(8):
 mask=real_bins==b;valid=mask&np.isfinite(real_width);n=int(valid.sum());mu=float(real_width[valid].mean());var=float(real_width[valid].var());rwithin+=n/rn*var;rbetween+=n/rn*(mu-rm)**2
 fields={}
 for k in (*FIELDS044,*[key+'_from_selected_peak_ns' for key in CROSSINGS053]):
  value=real[k];good=mask&np.isfinite(value);fields[k]=dict(total_rows=int(mask.sum()),defined_rows=int(good.sum()),undefined_rows=int((mask&~np.isfinite(value)).sum()),mean=float(value[good].mean()),SD=float(value[good].std()),quantiles=np.quantile(value[good],quantiles).tolist())
 real_bin_records.append(dict(bin=b,total_rows=int(mask.sum()),fields=fields,width_variance_ns2=var))
assert abs(rv-rwithin-rbetween)<1e-11
for (seed,case),z in stage_tables.items():
 width=z['FWHM_ns_by_stage'];base=stage_tables[seed,'BASE054'];basewidth=base['FWHM_ns_by_stage']
 for group_key in ('input_Q_bin','BASE054_output_Q_bin'):
  group=z[group_key]
  for b in [-1,*range(8)]:
   mask=np.ones(6593,dtype=bool) if b==-1 else group==b;common=mask&np.isfinite(width).all(0);values=width[:,common]
   components=np.concatenate((values[:1],np.diff(values,axis=0)),axis=0);cov=np.cov(components,ddof=0)
   mean_closure=float(np.max(np.abs(components.sum(0)-values[-1])));variance_closure=float(abs(cov.sum()-values[-1].var()));assert mean_closure<1e-12 and variance_closure<1e-10
   paired_stage.append(dict(seed=seed,case=case,grouping=group_key,bin=b,total_rows=int(mask.sum()),common_defined_rows=int(common.sum()),excluded_any_undefined_rows=int((mask&~np.isfinite(width).all(0)).sum()),
    all_row_defined_counts=[int(np.sum(mask&np.isfinite(x))) for x in width],all_row_mean_ns=[float(np.nanmean(x[mask])) for x in width],all_row_SD_ns=[float(np.nanstd(x[mask])) for x in width],
    common_mean_ns=values.mean(1).tolist(),common_SD_ns=values.std(1).tolist(),ordered_mean_components_ns=components.mean(1).tolist(),
    ordered_component_covariance_ns2=cov.tolist(),ordered_component_variances_ns2=np.diag(cov).tolist(),variance_closure_ns2=variance_closure,mean_closure_ns=mean_closure,
    unique_causal_attribution=False,flag_counts_by_stage={k:[int(np.sum(v[mask])) for v in z[k]] for k in z if k.endswith('_flag_by_stage')}))
   if case=='BASE054':continue
   joint=mask&np.isfinite(width).all(0)&np.isfinite(basewidth).all(0);delta=width[:,joint]-basewidth[:,joint]
   comp=np.concatenate((delta[:1],np.diff(delta,axis=0)),axis=0);cov=np.cov(comp,ddof=0)
   assert np.max(np.abs(comp.sum(0)-delta[-1]))<1e-12 and abs(cov.sum()-delta[-1].var())<1e-10
   case_contrasts.append(dict(seed=seed,case=case,reference='BASE054',grouping=group_key,bin=b,total_rows=int(mask.sum()),common_defined_rows=int(joint.sum()),
    common_stage_delta_mean_ns=delta.mean(1).tolist(),common_stage_delta_SD_ns=delta.std(1).tolist(),ordered_delta_mean_components_ns=comp.mean(1).tolist(),
    ordered_delta_covariance_ns2=cov.tolist(),variance_closure_ns2=float(cov.sum()-delta[-1].var()),unique_causal_attribution=False))
migrations=[]
for (seed,case,h),z in native_tables.items():
 migration=np.zeros((8,8),np.int64);np.add.at(migration,(z['BASE054_output_Q_bin'],z['native_output_Q_bin']),1)
 migrations.append(dict(seed=seed,case=case,hypothesis=h,BASE_to_case_bin_matrix=migration.tolist(),changed_bin_rows=int(np.sum(z['BASE054_output_Q_bin']!=z['native_output_Q_bin'])),
  native_peak_changed_rows=int(np.sum(z['native_peak_index']!=z['BASE054_peak_index']))))
result=dict(UTC=datetime.now(timezone.utc).isoformat(),quantile_probabilities=quantiles.tolist(),SOURCE_rows=6593,REAL_rows=21047,full_stage_scope='fast only',
 records=records,crossing_support=crossing_support,paired_stage=paired_stage,case_contrasts=case_contrasts,migrations=migrations,
 REAL=dict(bins=real_bin_records,width_mean_ns=float(rm),total_variance_ns2=float(rv),within_variance_ns2=float(rwithin),between_variance_ns2=float(rbetween),
  Q_width_Spearman=float(spearmanr(real['Q_Vns'][np.isfinite(real_width)],real_width[np.isfinite(real_width)]).statistic)),
 all_population_and_NaN_preserved=True,no_new_scientific_evaluations=True,parameters_selected=False,mean_stage_accepted=False)
(w/'WIDTH_STAGE_ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
key_records=[x for x in records if x['seed']==0 and x['hypothesis']=='fast' and x['grouping']=='native_output_Q_bin']
print(json.dumps(dict(native=[dict(case=x['case'],marginal=x['marginal'],within=x['within_variance_ns2'],between=x['between_variance_ns2'],
  extremes=[dict(bin=b,mean=x['bins'][b]['fields']['FWHM_ns']['mean'],width_W1=x['bins'][b]['source_real']['width_W1_to_same_REAL_output_bin_ns'],
  left=x['bins'][b]['fields']['left_crossing50_ns_relative_peak_ns']['mean'],right=x['bins'][b]['fields']['right_crossing50_ns_relative_peak_ns']['mean']) for b in (0,7)]) for x in key_records],
 stage_contrasts=[x for x in case_contrasts if x['seed']==0 and x['grouping']=='BASE054_output_Q_bin' and x['bin'] in (0,7)],
 migrations=[x for x in migrations if x['seed']==0 and x['hypothesis']=='fast']),allow_nan=False))
