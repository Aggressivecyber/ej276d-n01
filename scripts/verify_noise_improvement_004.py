"""Reopen and replay the frozen endpoint. No optimizer or classifier refit."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from n01.electronics_v3 import ObservedBaselineNoiseV3
from n01.model import digitize
from n01.noise_validation import BaselineClassifier,_view,balanced_real_domains
from n01.evaluate import _weighted_auc
from n01.data import sha256,write_json,canonical_real

def state_digest(state):
    h=hashlib.sha256()
    for k in sorted(state):
        a=state[k].detach().cpu().contiguous()
        h.update(k.encode());h.update(str(tuple(a.shape)).encode());h.update(str(a.dtype).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--science',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--old-science',type=Path);a=ap.parse_args()
    if (a.out/'REPLAY_VERIFICATION.json').exists():raise FileExistsError('Preserve prior replay')
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(12);torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device=torch.device(a.device)
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(device).total_memory,device=device)
    complete=json.loads((a.science/'COMPLETE.json').read_text(encoding='utf8'))
    assert complete['status']=='complete' and complete['noise_training_updates']==5000 and complete['classifier_training_updates']==1200
    assert complete['response_training_updates']==0 and complete['response_integration_allowed'] is False
    for name,want in complete['files'].items():
        assert sha256(a.science/name)==want,'Scientific payload hash mismatch: '+name
    manifestp=ROOT/'EXECUTION_SOURCE_MANIFEST.json'
    manifest=json.loads(manifestp.read_text(encoding='utf8'))
    assert complete['source_manifest_sha256']==sha256(manifestp)
    for name,want in manifest['files'].items():
        assert sha256(ROOT/name)==want,'Execution source hash mismatch: '+name
    for prefix in ('config','protocol','preregistration'):
        assert sha256(ROOT/manifest[prefix+'_path'])==manifest[prefix+'_sha256']
    state=torch.load(a.science/'noise/final.pt',map_location='cpu',weights_only=False)
    initial=torch.load(a.science/'noise/initial_state.pt',map_location='cpu',weights_only=False)
    replay=torch.load(a.science/'noise/replay.pt',map_location='cpu',weights_only=False)
    cfg=json.loads((ROOT/'noise_improvement_004_config.json').read_text(encoding='utf8'))
    assert state['config']==cfg and state['config_sha256']==sha256(ROOT/'noise_improvement_004_config.json')
    assert state['source_manifest_sha256']==sha256(manifestp)
    assert complete['final_checkpoint_sha256']==sha256(a.science/'noise/final.pt')
    assert complete['replay_sha256']==sha256(a.science/'noise/replay.pt')
    assert complete['noise_gate_sha256']==sha256(a.science/'noise_validation/NOISE_GATE.json')
    assert state['update']==cfg['noise']['updates']==5000 and state['response_training_updates']==0
    assert state['engineering'] is False and initial['update']==0
    assert state_digest(state['model'])==state['model_state_sha256']
    assert state_digest(initial['model'])==initial['model_state_sha256']
    for bound in (initial,replay):
        for key in ('config_sha256','source_manifest_sha256','protocol_sha256','preregistration_sha256'):
            assert bound[key]==state[key]
    assert replay['seed']==cfg['noise']['generation_seed']+10000 and replay['rows']==1024 and replay['chunk_size']==256
    container=cfg['local_training_container'] if os.name=='nt' else cfg['remote_training_container']
    identity=ROOT/'data/PARENT_IDENTITIES.npz'
    assert sha256(container)==cfg['train_container_sha256'] and sha256(identity)==cfg['identities_sha256']
    with np.load(container,allow_pickle=False) as z:
        train=canonical_real(z['target'])[:,:100];validation=canonical_real(z['target_val'])[:,:100]
    with np.load(identity,allow_pickle=False) as z:
        train_groups=z['real_train_group'];val_groups=z['real_val_group']
    assert not np.intersect1d(train_groups,val_groups).size
    visits=torch.as_tensor(state['sampler']['visits'])
    assert len(visits)==21047 and bool((visits>0).all()) and int(visits.sum())==5000*512
    model=ObservedBaselineNoiseV3(state['initialization']).to(device)
    model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
    assert sum(p.numel() for p in model.parameters())==5151
    assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    changed={k:float((state['model'][k]-initial['model'][k]).abs().max()) for k in state['model']
             if state['model'][k].is_floating_point()}
    chunks=[]
    with torch.no_grad():
        for i,chunk in enumerate(replay['chunks']):
            noise={k:v.to(device) for k,v in chunk['noise'].items()}
            value=model(noise,length=100);pre=value.cpu();post=digitize(value,ste=False).cpu()
            pe=(pre-chunk['pre_adc']).abs().max().item()
            qe=(post-chunk['post_adc']).abs().max().item()
            same=torch.equal(pre,chunk['pre_adc'])
            if device.type=='cuda':assert same,'Same-device generator replay mismatch'
            else:assert pe<1e-12,'CPU generator replay mismatch'
            assert torch.equal(post,chunk['post_adc']),'Digitized replay mismatch'
            chunks.append(dict(chunk=i,rows=len(pre),pre_ADC_max_abs_error=pe,post_ADC_max_abs_error=qe,pre_ADC_bitwise=same,post_ADC_bitwise=True))
    assert sum(x['rows'] for x in chunks)==1024
    rejected=[]
    for length in (0,99,101,480):
        try:model.sample_noise(1,device=device,length=length)
        except ValueError:rejected.append(length)
        else:raise AssertionError('Unsupported length accepted')
    gate=json.loads((a.science/'noise_validation/NOISE_GATE.json').read_text(encoding='utf8'))
    assert gate['model_state_sha256']==state['model_state_sha256']==replay['model_state_sha256']
    assert gate['noise_checkpoint_sha256']==complete['final_checkpoint_sha256'] and gate['config_sha256']==state['config_sha256']
    assert gate['response_integration_allowed'] is False and gate['next_action']=='stop_no_validated_480point_response'
    with np.load(a.science/'noise_validation/baseline_populations.npz',allow_pickle=False) as f:
        populations={k:f[k] for k in f.files}
    assert np.array_equal(populations['REAL_validation'],validation)
    assert np.array_equal(populations['real_validation_groups'],val_groups)
    generation_rng=torch.Generator(device=device).manual_seed(cfg['noise']['generation_seed'])
    generated_replay_rows=0
    with torch.no_grad():
        for key,count in [('generated_classifier_fit',cfg['noise']['classifier_fit_per_class']),('generated',cfg['noise']['generated_statistics_rows'])]:
            rows=[]
            for start in range(0,count,256):
                noise=model.sample_noise(min(256,count-start),device=device,generator=generation_rng,length=100)
                rows.append(digitize(model(noise,length=100),ste=False)[:,0].cpu().numpy())
            actual=np.concatenate(rows)
            assert np.array_equal(actual,populations[key]),'Frozen generated population replay mismatch'
            generated_replay_rows+=count
    fit0,fit1=balanced_real_domains(train_groups,cfg['noise']['classifier_fit_per_class'],cfg['noise']['split_seed'])
    test0,test1=balanced_real_domains(val_groups,cfg['noise']['classifier_test_per_class'],cfg['noise']['split_seed']+1)
    for name,value in [('real_train_fit0',fit0),('real_train_fit1',fit1),('real_validation_test0',test0),('real_validation_test1',test1)]:
        assert np.array_equal(populations[name],value),'Frozen evaluation partition mismatch'
    n=cfg['noise']['classifier_test_per_class'];rr=[];results=[]
    for view in ('raw','demean','detrend'):
        for case in ('generated_REAL','REAL_REAL'):
            folder=a.science/'noise_validation/classifiers'/view/case
            c=torch.load(folder/'classifier.pt',map_location='cpu',weights_only=False)
            with np.load(folder/'predictions.npz',allow_pickle=False) as z: saved={k:z[k] for k in z.files}
            v=populations['REAL_validation'];idx0=populations['real_validation_test0'];idx1=populations['real_validation_test1']
            if case=='generated_REAL':left,right=populations['generated'][:n],v[idx0]
            else:left,right=v[idx0],v[idx1]
            fleft=populations['generated_classifier_fit'] if case=='generated_REAL' else train[fit0]
            fright=train[fit0] if case=='generated_REAL' else train[fit1]
            fit=np.concatenate([_view(fleft,view),_view(fright,view)]).astype(np.float32)
            center=fit.mean(0,dtype=np.float64).astype(np.float32)
            scale=np.maximum(fit.std(0,dtype=np.float64),1/16383).astype(np.float32)
            assert np.array_equal(center,c['center']) and np.array_equal(scale,c['scale']),'Fit-only scaler mismatch'
            assert np.array_equal(saved['labels'],np.r_[np.zeros(n),np.ones(n)])
            expected0=np.arange(n) if case=='generated_REAL' else val_groups[test0]
            expected1=val_groups[test0] if case=='generated_REAL' else val_groups[test1]
            assert np.array_equal(saved['groups0'],expected0) and np.array_equal(saved['groups1'],expected1)
            assert c['seed']==cfg['noise']['classifier_seed'] and c['updates']==cfg['noise']['classifier_updates']
            test=np.concatenate([_view(left,view),_view(right,view)]).astype(np.float32)
            normalized=(test-c['center'])/c['scale']
            classifier=BaselineClassifier().to(device);classifier.load_state_dict(c['model']);classifier.eval()
            with torch.no_grad():score=classifier(torch.as_tensor(normalized,device=device)).cpu().numpy()
            error=float(np.max(np.abs(score-saved['scores'])))
            bitwise=bool(np.array_equal(score,saved['scores']))
            if device.type=='cuda':assert bitwise,'Same-device classifier replay mismatch'
            else:assert error<2e-6,'CPU classifier replay exceeded tolerance'
            auc=float(_weighted_auc(saved['labels'],saved['scores']))
            assert auc==gate['classifiers'][view][case]['AUC']
            brng=np.random.default_rng(cfg['noise']['classifier_seed']+2);values=[]
            if case=='REAL_REAL':
                unique,index=np.unique(np.r_[expected0,expected1],return_inverse=True)
                for _ in range(cfg['noise']['bootstrap_resamples']):
                    weights=np.bincount(brng.integers(0,len(unique),len(unique)),minlength=len(unique))[index]
                    values.append(_weighted_auc(saved['labels'],saved['scores'],weights))
            else:
                u0,i0=np.unique(expected0,return_inverse=True);u1,i1=np.unique(expected1,return_inverse=True)
                for _ in range(cfg['noise']['bootstrap_resamples']):
                    w0=np.bincount(brng.integers(0,len(u0),len(u0)),minlength=len(u0))[i0]
                    w1=np.bincount(brng.integers(0,len(u1),len(u1)),minlength=len(u1))[i1]
                    values.append(_weighted_auc(saved['labels'],saved['scores'],np.r_[w0,w1]))
            interval=np.quantile(values,[.025,.975])
            assert np.array_equal(interval,np.asarray(gate['classifiers'][view][case]['percentile_95_interval']))
            results.append(dict(view=view,case=case,rows=len(score),max_abs_logit_error=error,bitwise_equal=bitwise,saved_score_AUC=auc))
            if case=='REAL_REAL' and a.old_science:
                with np.load(a.old_science/'noise_validation/classifiers'/view/case/'predictions.npz',allow_pickle=False) as z:
                    equal=all(np.array_equal(z[k],saved[k]) for k in ('labels','scores','groups0','groups1'))
                rr.append(dict(view=view,old_v2_control_bitwise_equal=bool(equal)))
                assert equal,'Unchanged REAL-REAL control did not reproduce'
    cov=model.covariance().detach().cpu().numpy()
    eigen=np.linalg.eigvalsh(cov);assert eigen[0]>0
    result=dict(status='reopened_and_replayed',device=str(device),noise_training_updates_added=0,classifier_updates_added=0,response_updates_added=0,
        final_checkpoint_sha256=sha256(a.science/'noise/final.pt'),replay_payload_sha256=sha256(a.science/'noise/replay.pt'),
        generator_rows_replayed=1024,generator_chunks=chunks,classifier_checkpoints_replayed=6,
        generated_evaluation_population_rows_replayed=generated_replay_rows,generated_evaluation_populations_bitwise_equal=True,
        classifier_score_rows_replayed=sum(x['rows'] for x in results),classifiers=results,unchanged_REAL_REAL_controls=rr,
        all_REAL_TRAIN_rows_visited=21047,sampler_total_rows=int(visits.sum()),min_visits=int(visits.min()),max_visits=int(visits.max()),
        strict_length_rejections=rejected,parameter_count=5151,parameter_max_abs_change=changed,
        pre_ADC_covariance_min_eigenvalue_V2=float(eigen[0]),noise_gate_passed=gate['passed'],failed_checks=gate['failed_checks'],
        scientific_payload_manifest_entries_verified=len(complete['files']),execution_source_files_verified=len(manifest['files']),
        execution_source_manifest_sha256=sha256(manifestp),complete_sha256=sha256(a.science/'COMPLETE.json'),
        input_hashes_verified=True,all_classifier_fit_only_scalers_verified=True,test_labels_and_groups_verified=True,
        AUC_and_all_six_intervals_recomputed=True,other_gate_metrics='unchanged_report_bound_by_COMPLETE_manifest',
        scientific_acceptance_inferred_from_replay=False)
    write_json(a.out/'REPLAY_VERIFICATION.json',result)
    print(json.dumps({k:result[k] for k in ('status','generator_rows_replayed','classifier_checkpoints_replayed','classifier_score_rows_replayed','noise_gate_passed')}))
if __name__=='__main__':main()
