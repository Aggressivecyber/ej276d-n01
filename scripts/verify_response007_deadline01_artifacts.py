"""CPU-only postflight: journals, checkpoints, stored readouts and bootstrap.

This is primary-agent verification, not an independent reviewer. It does not
generate waveforms, fit models, or decode the original SOURCE/REAL container.
"""
from pathlib import Path
import os
import sys
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import datetime as dt
import hashlib
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
ARMS=('deterministic','full')
HYPOTHESES=('fast','slow','common_level')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1024**2),b''):h.update(c)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def require(value,message):
    if not bool(value):raise AssertionError(message)
def close(actual,expected,message,atol=3e-12):
    require(np.allclose(np.asarray(actual,dtype=float),np.asarray(expected,dtype=float),rtol=3e-12,atol=atol,equal_nan=True),message)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--science',default=str(ROOT/'runs/response_007_20261013/deadline01/runs/science'))
    args=parser.parse_args();science=Path(args.science);folder=science/'TRAIN_diagnostics'
    receipt=ROOT/'receipts/response_007_20261013/deadline01/ARTIFACT_VERIFICATION.json'
    require(not receipt.exists(),'Preserve the first verification receipt')
    complete=read(science/'COMPLETE.json');summary=read(folder/'SUMMARY.json')
    require(complete['G_updates']==8000 and complete['new_execution_G_updates']==7999 and complete['D_updates']==6000,'Update budget mismatch')
    require(complete['C2ST_updates']==complete['noise_updates']==0 and not complete['DEV_waveforms_read'],'Unexpected evaluation/training role')
    require(not complete['original_19000_per_arm_endpoint_reached'],'Partial endpoint mislabeled')
    checked={}
    for relative,digest in complete['files'].items():
        require(sha(science/relative)==digest,'Completed artifact changed: '+relative)
        checked[str(science/relative)]=digest
    frozen=read(ROOT/'RESPONSE007_DEADLINE01_SOURCE_MANIFEST.json')
    for relative,digest in frozen['files'].items():require(sha(ROOT/relative)==digest,'Frozen source changed: '+relative)
    journals={};stage={};endpoints={};audits={}
    oldline=(ROOT/'provenance/response007_statusfix01/step00001_training.jsonl').read_text(encoding='utf8').splitlines()[0]
    for arm in ARMS:
        p=science/'train'/arm
        lines=(p/'training.jsonl').read_text(encoding='utf8').splitlines()
        require(len(lines)==4000,arm+' journal row count')
        if arm=='deterministic':require(lines[0]==oldline,'Prior G1 journal was changed or repeated')
        rows=[json.loads(line) for line in lines];journals[arm]=rows
        for i,row in enumerate(rows,1):
            require(row['G_updates']==i and row['arm']==arm,arm+' journal sequence')
            require(row['stage']==('mean_response' if i<=3000 else 'joint_stochastic'),arm+' stage schedule')
            require(row['D_updates']=={h:max(0,i-3000) for h in HYPOTHESES},arm+' D counts')
            require(row['noise_updates']==row['population_updates']==0,arm+' frozen roles')
            require(row['signal_prefix_absolute_max_V']<=1/16383,arm+' signal prefix leakage')
            require(np.isfinite(row['weighted_total']),arm+' nonfinite loss')
            for h in HYPOTHESES:
                en=row['hypotheses'][h]['engineering']
                require(en['nonpositive_Q']==en['guarded_PSD']==0,arm+' invalid Q/PSD')
        endpoint=load(p/'bounded_04000.pt');endpoints[arm]=endpoint
        require(endpoint['G_updates']==4000 and endpoint['D_updates']=={h:1000 for h in HYPOTHESES},arm+' endpoint count')
        require(endpoint['stage']=='joint_stochastic' and endpoint['noise_updates']==0 and not endpoint['population_learning'],arm+' endpoint stage')
        require(int(endpoint['source_sampler']['visits'].sum())==4000*512,arm+' SOURCE sampler accounting')
        require(int(endpoint['real_sampler']['visits'].sum())==4000*1024,arm+' REAL sampler accounting')
        require(bool((endpoint['source_sampler']['visits']>0).all()) and bool((endpoint['real_sampler']['visits']>0).all()),arm+' incomplete population coverage')
        require(all(bool(torch.isfinite(v).all()) for v in endpoint['model'].values()),arm+' endpoint nonfinite state')
        stage[arm]=load(p/'stage_03000.pt')
        audits[arm]=dict(journal_rows=len(rows),G_updates=4000,D_updates=3000,
            peak_reported_GPU_reserved=max(r['gpu_reserved_bytes'] for r in rows),
            max_signal_prefix_V=max(r['signal_prefix_absolute_max_V'] for r in rows),
            all_hypotheses_Q_positive=True,all_noise_population_updates_zero=True,
            last_branch_gradients=rows[-1].get('G_branch_postclip_gradient_norms'),
            final_coverage=rows[-1]['coverage'])
    common_keys=set(stage['deterministic']['model'])&set(stage['full']['model'])
    equal_keys=[k for k in sorted(common_keys) if torch.equal(stage['deterministic']['model'][k],stage['full']['model'][k])]
    require(len(equal_keys)==len(common_keys),'Mean-stage shared parameters differ')
    for i in range(3000):
        a,b=journals['deterministic'][i],journals['full'][i]
        for key in ['weighted_total','regular_once','trend_once','hypotheses','G_preclip_gradient_norm','projection_context']:
            require(a[key]==b[key],f'Mean-stage paired journal differs at {i+1}/{key}')
    for name in ['source_sampler','real_sampler']:
        for key in ['order','visits']:
            require(torch.equal(endpoints['deterministic'][name][key],endpoints['full'][name][key]),'Paired final sampler differs')
    for key in endpoints['full']['streams']:
        require(torch.equal(endpoints['deterministic']['streams'][key],endpoints['full']['streams'][key]),'Paired final stream differs')
    grad=endpoints['full']['last_metrics']['G_branch_postclip_gradient_norms']
    for key in ['physical_head','basis_head','envelope_head','local_filter']:
        require(grad.get(key,0)>0,'Full stochastic branch has zero last-audit gradient: '+key)
    with np.load(folder/'TRAIN_observables.npz',allow_pickle=False) as z:obs={k:z[k] for k in z.files}
    parent=obs['source_parent_row'];require(np.array_equal(parent,np.repeat(np.arange(6593),4)),'Parent/draw map differs')
    require(np.array_equal(obs['draw'],np.tile(np.arange(4),6593)),'Draw map differs')
    require(len(obs['REAL_Q'])==21047 and len(np.unique(obs['real_original_ids']))==21047,'REAL population differs')
    stats=read(ROOT/'provenance/integration006_reference/statistics001.json')['loss']
    def bins(q):
        floor,reference=stats['q_floor'],stats['q_reference']
        coordinate=np.where(q>=floor,np.log(np.maximum(q,floor)/reference),np.log(floor/reference)+np.arcsinh((q-floor)/floor))
        return np.searchsorted(stats['charge_bin_boundaries'],coordinate,side='left')
    def scalar(v):return dict(mean=float(v.mean()),SD=float(v.std(ddof=0)),IQR=float(np.quantile(v,.75)-np.quantile(v,.25)))
    mean_scales=np.array([obs['REAL_Q'].std(ddof=0),obs['REAL_PSD'].std(ddof=0)])
    points=[];stats_checked=0
    for arm in ARMS:
        fixed=obs[arm+'_fixed_bins'];require(np.array_equal(fixed,bins(obs[arm+'_signal_Q'])),'Fixed signal bins changed')
        means=np.array([[obs[f'{arm}_{h}_{k}'].mean() for k in ['Q','PSD']] for h in HYPOTHESES])
        ratios=[]
        for b in range(8):
            widths=np.array([obs[f'{arm}_{h}_PSD'][fixed==b].std(ddof=0) for h in HYPOTHESES])
            ratios.append(float(widths.max()/widths.min()))
        points.append(np.r_[np.ptp(means,axis=0)/mean_scales,ratios])
        for view in ('signal_only',)+HYPOTHESES:
            record=summary['population_summary'][arm][view]
            for scheme,where in [('fixed_signal_Q_bins',fixed),('original_SOURCE_Q_bins',np.repeat(obs['source_bins'],4)),('output_Q_bins',bins(obs[f'{arm}_{view}_Q']))]:
                for b in range(8):
                    mask=where==b;saved=record[scheme][b]
                    require(saved['rows']==int(mask.sum()),'Conditional row count differs')
                    require(saved['parents']==len(np.unique(parent[mask])),'Conditional parent count differs')
                    for kind in ['Q','PSD']:
                        values=obs[f'{arm}_{view}_{kind}'][mask]
                        if len(values):
                            for k,v in scalar(values).items():close(v,saved[kind][k],'Conditional statistic differs')
                            stats_checked+=3
                        else:require(saved[kind]['mean'] is None,'Empty bin was filled')
            require(int(np.array(record['fixed_to_output_migration']).sum())==6593*4,'Migration loses rows')
    close(points,summary['sensitivity']['point'],'Sensitivity point differs')
    with np.load(folder/'TRAIN_bootstrap.npz',allow_pickle=False) as z:bs=z['statistics']
    require(bs.shape==(1000,2,10),'Bootstrap dimensions differ')
    close(np.moveaxis(np.quantile(bs,[.025,.975],axis=0),0,-1),summary['sensitivity']['percentile_95_interval'],'Bootstrap intervals differ')
    samples=np.load(folder/'TRAIN_sample_waveforms.npz',allow_pickle=False)
    adc_checks={}
    for arm in ARMS:
        signal=samples[arm+'_signal_pre']
        require(np.max(np.abs(signal[:,:100]))<=1/16383,'Sample signal prefix leakage')
        for h in HYPOTHESES:
            pre=signal+samples['noise_'+h]
            code=np.rint((np.clip(pre,-.5,.5)+.5)*16383)
            actual=samples[arm+'_'+h]
            saved_code=np.rint((actual+.5)*16383)
            require(np.array_equal(code,saved_code),'Sample ADC code differs')
            canonical=code/16383-.5
            require(np.max(np.abs(canonical-actual))<=4*np.finfo(np.float64).eps,'Sample ADC representation differs')
            q=actual[:,100:400].sum(1)-3*actual[:,:100].sum(1)
            psd=(actual[:,150:400].sum(1)-2.5*actual[:,:100].sum(1))/q
            close(q,obs[f'{arm}_{h}_Q'][:len(q)],'Sample Q differs')
            close(psd,obs[f'{arm}_{h}_PSD'][:len(q)],'Sample PSD differs')
            adc_checks[arm+'/'+h]=dict(rows=len(q),integer_code_differences=0)
    selection=summary['matched_selection']
    require(len(set(selection['source_rows']))==2048,'Matched SOURCE selection repeats')
    require(not set(selection['real_A'])&set(selection['real_B']),'REAL-REAL control overlaps')
    require(all(v['matched_rows']==[2048,2048] for a in ARMS for v in summary['matched_distances'][a].values()),'Unmatched distribution comparison')
    result=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='passed',reviewer='primary_agent_self_verification_no_subagents',
        source_files_verified=len(frozen['files']),completed_files_sha256_verified=len(checked),
        journals=audits,mean_stage_shared_state_keys_bitwise_equal=len(equal_keys),mean_stage_paired_metric_steps_bitwise_equal=3000,
        paired_final_sampler_and_random_states_equal=True,full_random_branch_gradients_nonzero=True,
        conditional_scalar_values_recomputed=stats_checked,bootstrap_intervals_recomputed=20,
        sample_ADC_checks=adc_checks,G_updates=8000,new_execution_G_updates=7999,D_updates=6000,C2ST_updates=0,
        original_full_endpoint_reached=False,scientific_status='response_match_not_established',
        model_forward_calls=0,original_waveform_decodes=0,new_unique_random_draws=0,new_training_updates=0,
        verification_limits=['stored aggregate waveform curves need native generation replay for full reproduction','TRAIN only; no scientific equivalence or cross-acquisition claim'],
        complete_sha256=sha(science/'COMPLETE.json'),summary_sha256=sha(folder/'SUMMARY.json'),implementation_sha256=sha(Path(__file__)))
    receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','source_files_verified','completed_files_sha256_verified','mean_stage_shared_state_keys_bitwise_equal','mean_stage_paired_metric_steps_bitwise_equal','conditional_scalar_values_recomputed','bootstrap_intervals_recomputed']},ensure_ascii=False))

if __name__=='__main__':main()
