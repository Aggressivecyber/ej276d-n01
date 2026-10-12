"""Reopen the fixed noise checkpoint and C2ST artifacts without new fitting."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from n01.data import canonical_real,write_json,sha256
from n01.noise_validation import BaselineClassifier,_view
from n01.evaluate import _weighted_auc
from n01.losses_v2 import ProjectionContext
from n01.train_repair import validate_noise_endpoint,state_digest


def main():
    torch.set_num_threads(2)
    out=ROOT/'runs/repair_002_20261012'
    science=out/'runs/science'
    cfg=json.loads((ROOT/'repair_run_config.json').read_text(encoding='utf8'))
    model=torch.load(science/'noise/final.pt',map_location='cpu',weights_only=False)
    validate_noise_endpoint(model,cfg)
    for value in model['model'].values():
        if not torch.isfinite(value).all():raise ValueError('Nonfinite checkpoint')
    ProjectionContext.from_state_dict(model['projection_context'])
    log=[json.loads(x) for x in (science/'noise/training.jsonl').read_text().splitlines()]
    assert [x['update'] for x in log]==list(range(1,5001))
    assert len(set(x['projection_context']['fingerprint'] for x in log))==5000
    assert min(model['sampler']['visits'])>0
    container=Path('E:/GAN/research/sim2real_two_papers_20261008_0305/work/R23_ONLINE_STANDARDIZED_CRITIC_001/TRAIN_PRIVATE.npz')
    assert sha256(container)==cfg['train_container_sha256']
    with np.load(container) as z:train=canonical_real(z['target'])[:,:100]
    with np.load(science/'noise_validation/baseline_populations.npz') as z:data={k:z[k].copy() for k in z.files}
    gate=json.loads((science/'noise_validation/NOISE_GATE.json').read_text())
    checks=[]
    for view in ('raw','demean','detrend'):
        for case in ('generated_REAL','REAL_REAL'):
            folder=science/'noise_validation/classifiers'/view/case
            checkpoint=torch.load(folder/'classifier.pt',map_location='cpu',weights_only=False)
            with np.load(folder/'predictions.npz') as z:pred={k:z[k].copy() for k in z.files}
            if case=='generated_REAL':
                fit0=data['generated_classifier_fit'];fit1=train[data['real_train_fit0']]
                test0=data['generated'][:256];test1=data['REAL_validation'][data['real_validation_test0']]
            else:
                fit0=train[data['real_train_fit0']];fit1=train[data['real_train_fit1']]
                test0=data['REAL_validation'][data['real_validation_test0']]
                test1=data['REAL_validation'][data['real_validation_test1']]
            x=np.concatenate([_view(fit0,view),_view(fit1,view)]).astype(np.float32)
            center=x.mean(0,dtype=np.float64).astype(np.float32)
            scale=np.maximum(x.std(0,dtype=np.float64),1/16383).astype(np.float32)
            assert np.array_equal(center,checkpoint['center'])
            assert np.array_equal(scale,checkpoint['scale'])
            test=np.concatenate([_view(test0,view),_view(test1,view)]).astype(np.float32)
            classifier=BaselineClassifier();classifier.load_state_dict(checkpoint['model']);classifier.eval()
            with torch.no_grad():actual=classifier(torch.from_numpy((test-center)/scale)).numpy()
            max_delta=float(np.max(abs(actual-pred['scores'])))
            assert max_delta<1e-4,(view,case,max_delta)
            auc=_weighted_auc(pred['labels'],pred['scores'])
            assert auc==gate['classifiers'][view][case]['AUC']
            checks.append(dict(view=view,case=case,AUC=auc,fit_scaler_bitwise_equal=True,
                               reopened_CPU_vs_saved_CUDA_max_abs_logit_error=max_delta))
    receipt=dict(status='reopened_and_verified',noise_updates=5000,response_updates=0,
                 distinct_training_projection_contexts=5000,all_TRAIN_rows_seen=True,
                 visits_range=[int(model['sampler']['visits'].min()),int(model['sampler']['visits'].max())],
                 checkpoint_sha256=sha256(science/'noise/final.pt'),electronics_state_sha256=state_digest(model['model']),
                 reopened_classifiers=checks,classifier_updates_added=0,noise_updates_added=0,
                 native_exit_code=json.loads((science/'exit_receipt.json').read_text())['exit_code'])
    write_json(out/'REOPEN_VERIFICATION.json',receipt)
    print(json.dumps(dict(status=receipt['status'],reopened_classifiers=len(checks),max_logit_error=max(x['reopened_CPU_vs_saved_CUDA_max_abs_logit_error'] for x in checks))))


if __name__=='__main__':main()
