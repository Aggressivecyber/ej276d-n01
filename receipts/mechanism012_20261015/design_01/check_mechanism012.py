"""Tiny synthetic leaf-parameter fixture for the actual012 pipeline.

No original A/005 neural model is instantiated. No data/checkpoint is decoded.
All inputs, two1288-parameter leaves and affine noise algebra are artificial.
Engineering AD/ADC calls are recorded separately from zero scientific work.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts')); sys.path.insert(0, str(ROOT/'src'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
torch.set_num_threads(1)
import mechanism012_losses as helper
import run_mechanism012 as runner
from n01.losses008_v2 import N01Loss008
from n01.model008 import canonical_charge008
from n01.observation008 import observe_generated_carrier
from n01.response008 import fixed_noises
from n01.data import FullCoverageSampler


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats():
    return dict(fit_split='TRAIN', validity={}, minimum_conditional_support=16,
        q_reference=torch.tensor(.4, dtype=torch.float64), q_floor=torch.tensor(.001, dtype=torch.float64),
        peak_floor=torch.tensor(1/16383, dtype=torch.float64), charge_bin_boundaries=torch.tensor([-2.,-1.5,-1.,-.5,0.,.5,1.], dtype=torch.float64),
        views={name:dict(center=torch.zeros(d,dtype=torch.float64),scale=torch.ones(d,dtype=torch.float64))
            for name,d in zip(('voltage','shape','difference','curvature','difference_smooth_5','difference_smooth_17'),(256,256,255,254,255,255))},
        qpsd=dict(center=torch.zeros(2,dtype=torch.float64),scale=torch.ones(2,dtype=torch.float64)),
        noise=dict(center=torch.zeros(32,dtype=torch.float64),scale=torch.ones(32,dtype=torch.float64)))


class SyntheticLeafResponse:
    def __init__(self, zero=False):
        generator=torch.Generator(device='cpu').manual_seed(2026101501)
        self.weight=torch.nn.Parameter(torch.zeros(8,32,5) if zero else .02*torch.randn((8,32,5),generator=generator))
        self.bias=torch.nn.Parameter(torch.zeros(8) if zero else .02*torch.randn(8,generator=generator))
        self.baseline=SimpleNamespace(a_ref=torch.tensor(.02,dtype=torch.float64))
        t=torch.arange(384,dtype=torch.float64)
        gate=(t>=100).double()
        b=torch.stack([torch.sin((t-110)*(i+1)/50)*gate for i in range(8)])
        projection=torch.zeros(384,dtype=torch.float64);projection[110:366]=1/256
        self.bases=b-canonical_charge008(b)[:,None]*projection
        self.calls=0
    def trainable_parameters(self):return (self.weight,self.bias)
    def algebra(self,x,z):
        base=(x-x[:,:100].mean(1,keepdim=True)).clone();base[:,:100]=0
        controls=torch.tanh(self.weight.mean((1,2))+self.bias).double()
        random=.008*torch.einsum('bk,k,kt->bt',z.double(),controls,self.bases)
        signal=base+random
        return dict(signal_pre=signal,base_signal_pre_V=base,random_shape_pre_V=random,
            regular_rms=random.square().mean()/self.baseline.a_ref.square(),
            regular_curvature=torch.diff(random,n=2,dim=1).square().mean()/self.baseline.a_ref.square(),
            canonical_charge_identity_residual=canonical_charge008(signal)-canonical_charge008(base),
            canonical_random_shape_charge=canonical_charge008(random),baseline_signal_forward_calls=1,extra_baseline_signal_forward_calls=0)
    def random_forward(self,x,z,enabled=True):
        assert enabled is True
        self.calls+=1
        return self.algebra(x,z)


class SyntheticAffineNoise:
    def assert_frozen(self):pass
    def sample_prefix(self,latents):
        return dict(prefix_pre=.30+latents['baseline_gaussian'][:,0].double()*1e-4,
            scale=torch.ones(len(latents['baseline_gaussian']),1,dtype=torch.float64))
    def extend_prefix(self,prefix,scale,epsilon,hypothesis):
        factor={'fast':1,'slow':2,'common_level':3}[hypothesis]
        tail=prefix.mean(1,keepdim=True)+factor*epsilon[:,0].double()*1e-4
        return dict(noise_pre=torch.cat((prefix,tail),1))


def snapshot(model, extension):
    return dict(synthetic_only=True,head={str(i):hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()
        for i,p in enumerate(model.trainable_parameters())},no_original_A_005_or_extension_state=True)


def run_case(destination, zero):
    policy=copy.deepcopy(runner.fixed_policy(ROOT/'mechanism012_policy.json'))
    n=32;parents=np.repeat(np.arange(8),4)
    policy.update(matched_rows=n,source_parent_clusters=8,forward_chunk_rows=8)
    policy['budgets'].update(old_A_rows=n,head_final_rows=n,noise005_rows=n,ADC_hard_or_STE_rows=24*n,checkpoint_loads_max=0)
    policy['resource_limits']=dict(CPU_threads=1,GPU_GiB=0,job_tree_RSS_GiB=1.1,wall_seconds=120,output_MiB=16)
    model,ext=SyntheticLeafResponse(zero),SyntheticAffineNoise()
    g=torch.Generator(device='cpu').manual_seed(2026101502)
    t=torch.arange(384,dtype=torch.float64)
    clean=torch.stack([(.018+i*.001)*torch.exp(-((t-126)/13)**2) for i in range(8)])
    clean[:,:100]=0
    raw=clean+torch.arange(8,dtype=torch.float64)[:,None]*.01+.17
    source=raw[parents]
    latents=dict(shape_z=torch.randn(n,8,generator=g),extension=torch.randn(n,1,284,generator=g),
        baseline_gaussian=torch.randn(n,1,100,generator=g),baseline_scale=torch.randn(n,1,1,generator=g))
    loss=N01Loss008(stats(),seed=2026101426,evaluation_seed=2026101430)
    context=loss.create_evaluation_context()
    q,c,b=helper.source_q_strata(raw,loss)
    expected=canonical_charge008(raw-raw[:,:100].mean(1,keepdim=True))
    torch.testing.assert_close(q,expected,rtol=0,atol=0)
    assert not torch.equal(q,canonical_charge008(raw))
    with torch.no_grad():
        rendered=model.algebra(source,latents['shape_z'])
        noises=fixed_noises(ext,{k:latents[k] for k in ('baseline_gaussian','baseline_scale')},latents['extension'])
        anchors={a:{} for a in ('frozen_A','shape010_final')}
        for arm,signal in (('frozen_A',rendered['base_signal_pre_V']),('shape010_final',rendered['signal_pre'])):
            for h in runner.HYPOTHESES:
                obs=observe_generated_carrier(signal+noises[h],ste=False,q_floor=.001,return_carrier=True,return_indices=True)
                anchors[arm][h]=dict(pure_signal_V=signal.detach(),waveform_V=obs['window_V'],
                    **{k:obs[k] for k in ('peak_index','crop_start','crop_stop','Q')})
        real=anchors['frozen_A']['fast']['waveform_V'].roll(1,0)*1.03
    identities=dict(source_parent_ids=np.arange(8),source_parent_rows=np.arange(8),source_parent_row=parents,
        REAL_reference_rows=np.arange(n),REAL_reference_ids=np.arange(n),REAL_reference_groups=np.arange(n)//4)
    store=runner.OperationStore012(destination,policy,artificial=True)
    source_hashes={str(ROOT/k):sha(ROOT/k) for k in ('scripts/mechanism012_losses.py','scripts/run_mechanism012.py')}
    store.binding=dict(files_sha256=source_hashes,sources_sha256=source_hashes)
    with (patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('Optimizer construction forbidden')),
          patch.object(FullCoverageSampler,'__init__',side_effect=AssertionError('Sampler construction forbidden'))):
        summary=runner.run_diagnostic(model,ext,source,latents,real,anchors,loss,context,identities,store,
            snapshotter=snapshot,protection_guard=lambda:runner.verify_files(store.binding))
    assert store.counters['scalar_AD_queries']==9 and store.counters['masked_endpoint_VJP_queries']==2
    assert model.calls==4 and all(p.grad is None for p in model.trainable_parameters())
    assert store.counters['ADC_hard_or_STE_rows']==n*(6 if zero else 24)
    complete=json.loads((store.root/'COMPLETE.json').read_text())
    proof=json.loads((store.root/'FIRST_DIAGNOSTIC_PROOF.json').read_text())
    assert complete['artificial_only'] is True and complete['engineering_checks_passed'] and complete['frozen_state_unchanged']
    assert proof['scalar_AD_queries_completed']==1 and proof['AD_queries_completed']==1
    assert complete['head_tensors_unchanged'] and complete['direction_defined']==(not zero)
    journal=[json.loads(line) for line in (store.root/'operations.jsonl').read_text().splitlines()]
    entered=[row for row in journal if row['event']=='entered'];completed=[row for row in journal if row['event']=='completed']
    assert [row['operation_id'] for row in entered]==[row['operation_id'] for row in completed]
    assert not store.inflight and len(summary['probe']['signed_probes'])==6
    if zero:
        assert all(not r['defined'] and r['reason']=='all_zero_random_shape' for r in summary['probe']['signed_probes'])
        assert summary['probe']['actual_probe_ADC_rows']==0
    else:
        assert len(summary['probe']['central_differences'])==3
    return dict(name='actual_pipeline_zero_direction' if zero else 'actual_pipeline_six_signed_three_paths',passed=True,
        engineering_AD_queries=11,engineering_ADC_rows=store.counters['ADC_hard_or_STE_rows'],
        parameter_values_unchanged=True,global_S_analytic_absmax=summary['S_reference_analytic_pullback_absmax'],
        total_sum_relative_error=summary['gradient_attribution']['total_sum_relative_error'],
        output_bytes=sum(p.stat().st_size for p in store.root.iterdir()),proof_uses_query1_only=True,
        nonzero_source_prefix_centering_checked=True)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);assert not out.exists();out.mkdir(parents=True)
    results=[run_case(out/'synthetic_normal',False),run_case(out/'synthetic_zero_direction',True)]
    policy=copy.deepcopy(runner.fixed_policy(ROOT/'mechanism012_policy.json'))
    policy['resource_limits']=dict(CPU_threads=1,GPU_GiB=0,job_tree_RSS_GiB=1.1,wall_seconds=120,output_MiB=16)
    failure=runner.OperationStore012(out/'synthetic_first_operation_failure',policy,artificial=True)
    try:
        failure.call('synthetic_forward_failure',dict(old_A_rows=128,head_final_rows=128),lambda:(_ for _ in ()).throw(RuntimeError('synthetic bounded operation failure')))
    except RuntimeError as error:
        runner.write_failure(failure,error)
    assert failure.attempted['old_A_rows']==128 and failure.counters['old_A_rows']==0 and failure.inflight
    assert (failure.root/'FAILED.json').exists() and not (failure.root/'COMPLETE.json').exists()
    events=[json.loads(line)['event'] for line in (failure.root/'operations.jsonl').read_text().splitlines()]
    assert events==['entered','failed']
    results.append(dict(name='entered_failed_unknown_inflight_not_zero_actual_or_complete',passed=True))
    receipt=dict(schema='n01/mechanism012-engineering/v1',artificial_only=True,native_exit_code=0,
        source_sha256={name:sha(ROOT/name) for name in ('scripts/mechanism012_losses.py','scripts/run_mechanism012.py')},
        passed=len(results),total=len(results),results=results,engineering_AD_queries=22,
        scientific_AD_queries=0,scientific_old_A_rows=0,scientific_noise005_rows=0,scientific_head_final_rows=0,
        private_array_decodes=0,checkpoint_decodes=0,original_A_or005_model_instances=0,optimizer_created=0,optimizer_steps=0)
    (out/'ENGINEERING_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(receipt,allow_nan=False))


if __name__=='__main__':main()
