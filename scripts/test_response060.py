"""Analytic conditional waveform moments, identity splits and ordered covariance; no scientific data."""
import argparse,json
from pathlib import Path
import numpy as np
from run_response060 import profiles060,ordered_wave060

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
parents=np.arange(64);groups=np.array([parents//8,(parents//2)%8]);a=(parents+1)*.002;v=np.linspace(.1,1.,256);wave=a[:,None]*v[None,:]
rows=profiles060(wave,groups,parents,.01);mean_error=variance_error=cov_error=block_error=split_error=closure_error=0.
assert np.array_equal(rows['counts'],np.full((2,8),8)) and np.array_equal(rows['parity_counts'],np.full((2,8,2),4))
for g in range(2):
 for b in range(8):
  mask=groups[g]==b;mean=a[mask].mean()*v;var=a[mask].var()*v**2;cov=a[mask].var()*np.outer(v[:64],v[:64]);block=v.reshape(16,16).sum(1)
  mean_error=max(mean_error,float(np.max(np.abs(rows['mean'][g,0,b]-mean))))
  variance_error=max(variance_error,float(np.max(np.abs(rows['SD_population'][g,0,b]**2-var))))
  cov_error=max(cov_error,float(np.max(np.abs(rows['covariance_first64'][g,0,b]-cov))))
  block_error=max(block_error,float(np.max(np.abs(rows['block_integral_mean'][g,0,b]-a[mask].mean()*block))),float(np.max(np.abs(rows['block_integral_covariance'][g,0,b]-a[mask].var()*np.outer(block,block)))))
  assert np.max(np.abs(rows['mean'][g,1,b]-v/v.sum()))<1e-14 and np.max(rows['SD_population'][g,1,b])<1e-14
  assert np.max(np.abs(rows['SE_IID_formula'][g,:,b]-rows['SD_population'][g,:,b]/np.sqrt(8)))<1e-14
  for parity in (0,1):split_error=max(split_error,float(np.max(np.abs(rows['parity_mean'][g,0,b,parity]-a[mask&(parents%2==parity)].mean()*v))))
 for view in (0,1):
  weights=rows['counts'][g]/64.;mu=rows['overall_mean'][view]
  rebuilt=np.sum(weights[:,None]*rows['mean'][g,view],axis=0);total=np.sum(weights[:,None]*(rows['SD_population'][g,view]**2+(rows['mean'][g,view]-mu)**2),axis=0)
  closure_error=max(closure_error,float(np.max(np.abs(rebuilt-mu))),float(np.max(np.abs(total-rows['overall_SD_population'][view]**2))))
assert max(mean_error,variance_error,cov_error,block_error,split_error,closure_error)<1e-12
permutation=np.random.default_rng(2026101860).permutation(64);permuted=profiles060(wave[permutation],groups[:,permutation],parents[permutation],.01)
permutation_error=max(float(np.max(np.abs(rows[k]-permuted[k]))) for k in rows if k not in ('parent_row','group_bin'))
assert permutation_error<1e-12
guarded=wave.copy();guarded[0]=0.;guarded[1]=-wave[1];guard=profiles060(guarded,groups,parents,.01)
assert np.array_equal(guard['counts'],rows['counts']) and int(guard['Q_nonpositive_counts'][0].sum())==2 and int(guard['Q_guarded_counts'][0].sum())==1
assert np.isfinite(guard['mean']).all() and np.isfinite(guard['covariance_first64']).all()
factors=np.array([1.,1.1,.95,1.03,1.02]);waves=factors[:,None,None]*wave[None,:,:];ordered,closure=ordered_wave060(waves,groups)
coeff=np.r_[factors[0],np.diff(factors)];ordered_error=0.
for g in range(2):
 for b in range(8):
  variance=a[groups[g]==b].var();expected=np.einsum('i,j,k->ijk',coeff,coeff,variance*v**2)
  ordered_error=max(ordered_error,float(np.max(np.abs(expected-ordered['component_covariance_V2'][g,b]))))
assert ordered_error<1e-12 and closure['mean_closure_max_error_V']<1e-12 and closure['variance_closure_max_error_V2']<1e-12
receipt=dict(status='PASS',device='numpy_CPU',manufactured_rows=64,checkpoint_loads=0,TRAIN_decodes=0,AD_queries=0,
 analytic_conditional_mean_max_error=mean_error,analytic_population_variance_max_error=variance_error,analytic_first64_covariance_max_error=cov_error,
 analytic_block_integral_moments_max_error=block_error,analytic_identity_split_mean_max_error=split_error,whole_population_moment_closure_error=closure_error,
 normalized_rank_one_shape_constant=True,zero_and_negative_Q_rows_retained=True,identity_permutation_moment_max_error=permutation_error,
 analytic_ordered_covariance_max_error=ordered_error,ordered_closure=closure,scientific_acceptance=False)
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
