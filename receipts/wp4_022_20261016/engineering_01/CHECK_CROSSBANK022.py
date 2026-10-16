"""One fixed artificial regression of022 axes, per-parent combination and signed U-self."""
from pathlib import Path
import hashlib
import json
import math
import os
import sys

here=Path(__file__).resolve().parent
repo=here.parents[2]
(here/'tmp').mkdir()
for name in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
    os.environ[name]=str(here/'tmp')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
assert sys.dont_write_bytecode
sys.path.insert(0,str(repo/'scripts'))
import numpy as np
import run_wp4_022 as driver
from run_wp4_018 import wave_features018
from n01.stats018 import DISTANCE_COUNTS018,self_mean018,energy_discrepancy018
from n01.stats017 import weighted017

P,R=3,4
p=np.arange(P)[:,None,None,None]
b=np.arange(2)[None,:,None,None]
r=np.arange(R)[None,None,:,None]
d=np.arange(19)[None,None,None,:]
Tobs=p*17.+b*.8+r*.05+d*.012
Sobs=Tobs+(1+p*.2)*np.where(b==0,.37,-.11)+d*.001*(r-1.5)
k=np.arange(256)[None,None,None,:]
Twave=(np.cos((k+1)*(r+1)/71.)+p*.7+b*.9+r*.15)/16383
Swave=Twave+(.3*np.sin((k+3)/17.)+p*.09+b*.11+r*.02)/16383
# Independent Python scalar-distance oracle. No production distance routine is used here.
oracle_counts={}
def oracle_sum(A,B,tag):
    sums=[]
    for ai,bi in zip(A,B):
        value=0.
        for left in ai:
            for right in bi:
                value+=math.sqrt(sum((float(x)-float(y))**2 for x,y in zip(left,right)))
                oracle_counts[tag]=oracle_counts.get(tag,0)+1
        sums.append(value)
    return np.asarray(sums)

def oracle_wave_features(waves):
    delta=1/16383
    return [[[[float(x)/(16*delta) for x in wave]+[sum(float(x) for x in wave)/(256*delta)]
        for wave in bank] for bank in parent] for parent in waves]

twf=np.asarray(oracle_wave_features(Twave));swf=np.asarray(oracle_wave_features(Swave))
reference_cross=oracle_sum(twf[:,0],twf[:,1],'WF_reference_cross')/R**2
reference_s0=oracle_sum(twf[:,0],twf[:,0],'WF_reference_self0')/(R*(R-1))
reference_s1=oracle_sum(twf[:,1],twf[:,1],'WF_reference_self1')/(R*(R-1))
reference=reference_cross-.25*(reference_s0+reference_s1)
DISTANCE_COUNTS018.clear()
actual=driver.crossbank_scores022(Swave,Twave,Sobs,Tobs,reference)
selfS=[oracle_sum(Sobs[:,i],Sobs[:,i],'O19_selfS'+str(i))/(R*(R-1)) for i in (0,1)]
selfT=[oracle_sum(Tobs[:,i],Tobs[:,i],'O19_selfT'+str(i))/(R*(R-1)) for i in (0,1)]
c01=oracle_sum(Sobs[:,0],Tobs[:,1],'O19_cross01')/R**2
c10=oracle_sum(Sobs[:,1],Tobs[:,0],'O19_cross10')/R**2
nullcross=oracle_sum(Sobs[:,0],Sobs[:,1],'O19_T2null_cross')/R**2
expected01=c01-.5*(selfS[0]+selfT[1]);expected10=c10-.5*(selfS[1]+selfT[0])
expected=.5*(expected01+expected10)
expected_null=nullcross-.5*(selfS[0]+selfS[1])
expected_wave=.5*(oracle_sum(swf[:,1],twf[:,0],'WF_cross10')/R**2-.5*oracle_sum(swf[:,1],swf[:,1],'WF_self1')/(R*(R-1))+
    oracle_sum(swf[:,0],twf[:,1],'WF_cross01')/R**2-.5*oracle_sum(swf[:,0],swf[:,0],'WF_self0')/(R*(R-1)))-reference
for key,value in dict(cross_parent=expected,cross_direction01=expected01,cross_direction10=expected10,T2_null=expected_null,
    full257_excess=expected_wave,T2_self0=selfS[0],T2_self1=selfS[1],T1_self0=selfT[0],T1_self1=selfT[1]).items():
    assert np.allclose(actual[key],value,rtol=1e-11,atol=1e-12),key
# Negative controls ensure bank reversal cannot silently become parent reversal or same-bank pairing.
wrong_parent=.5*(oracle_sum(Sobs[:,0],Tobs[::-1,1],'wrong_parent_cross01')/R**2-.5*(selfS[0]+selfT[1][::-1])+
    oracle_sum(Sobs[:,1],Tobs[::-1,0],'wrong_parent_cross10')/R**2-.5*(selfS[1]+selfT[0][::-1]))
same_bank=.5*(oracle_sum(Sobs[:,0],Tobs[:,0],'wrong_samebank_cross00')/R**2-.5*(selfS[0]+selfT[0])+
    oracle_sum(Sobs[:,1],Tobs[:,1],'wrong_samebank_cross11')/R**2-.5*(selfS[1]+selfT[1]))
pooledS=Sobs.reshape(P,2*R,19);pooledT=Tobs.reshape(P,2*R,19)
pooled=oracle_sum(pooledS,pooledT,'wrong_pooled_cross')/(2*R)**2-.5*(
    oracle_sum(pooledS,pooledS,'wrong_pooled_selfS')/(2*R*(2*R-1))+
    oracle_sum(pooledT,pooledT,'wrong_pooled_selfT')/(2*R*(2*R-1)))
assert not np.allclose(actual['cross_parent'],wrong_parent)
assert not np.allclose(actual['cross_parent'],same_bank)
assert not np.allclose(actual['cross_parent'],pooled)
# The two directions are averaged per parent BEFORE computing the parent-cluster SE.
parent_summary=weighted017(actual['cross_parent'],np.ones(P))
alpha=np.ones(P)/P
oracle_SE=math.sqrt(P/(P-1)*sum(float(a*a*(x-float(expected.mean()))**2) for a,x in zip(alpha,expected)))
assert math.isclose(parent_summary['SE'],oracle_SE,rel_tol=1e-11,abs_tol=1e-12)
naive=weighted017(np.concatenate((expected01,expected10)),np.ones(2*P))
assert not math.isclose(parent_summary['SE'],naive['SE'],rel_tol=1e-6,abs_tol=1e-12)
# Same samples do not have U-discrepancy zero: cross usesm², U-self usesm(m−1).
A=Tobs[:,0]
selfA=self_mean018(A)
identity=energy_discrepancy018(A,A,student_self=selfA,teacher_self=selfA)
expected_identity=-selfT[0]/R
assert np.allclose(identity,expected_identity,rtol=1e-11,atol=1e-12) and (identity<0).all()
fixture_distances=sum(DISTANCE_COUNTS018.values())
independent_oracle_distances=sum(oracle_counts.values())
assert DISTANCE_COUNTS018=={19:9*P*R**2,257:4*P*R**2}
assert fixture_distances+independent_oracle_distances<=100000
assert 'torch' not in sys.modules
receipt=dict(schema='n01/wp4-022-crossbank-artificial-regression/v1',status='PASS',artificial_only=True,
    fixture_domain='fixed feature/metric-space arrays; not generated hard waveforms or new physical evidence',
    P=P,banks=2,draws=R,source_sha256=hashlib.sha256((repo/'scripts/run_wp4_022.py').read_bytes()).hexdigest(),
    fixture_distance_counts=DISTANCE_COUNTS018,fixture_distance_pairs=fixture_distances,
    independent_oracle_distance_counts=oracle_counts,independent_oracle_distance_pairs=independent_oracle_distances,
    total_artificial_distance_pairs=fixture_distances+independent_oracle_distances,artificial_distance_cap=100000,
    all_crossbank_and_Uself_terms_match_independent_scalar_distance_oracle=True,
    parent_axis_reversal_detected=True,same_bank_pairing_detected=True,pooled64_semantics_detected=True,
    parent_cluster_SE=parent_summary['SE'],independent_scalar_oracle_parent_SE=oracle_SE,
    wrongly_independent_direction_SE=naive['SE'],same_sample_U_discrepancy_negative=identity.tolist(),
    scientific_NN_CP_AD_G_RNG=0,original_arrays_or_waveforms_read=0,new_T2_scientific_probe=0,
    old_005_observation_codec_or_engineering_suites_repeated=0)
(here/'CROSSBANK_SEMANTICS_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(status='PASS',fixture_pairs=fixture_distances,oracle_pairs=independent_oracle_distances,
    total_pairs=fixture_distances+independent_oracle_distances,NN_CP_AD_G_RNG=0)),flush=True)
