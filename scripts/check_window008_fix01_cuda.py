"""Native synthetic gradients/capacity and fixed historical failure inference."""
from pathlib import Path
import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import os
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from check_window008_engine import fixture, vjp_probe, serial
from n01.model import InvalidWaveformError
from n01.model008 import N01Response008
from n01.model008_fix01 import N01Response008Fix01
from n01.response008 import common_initial_state, make_extension, fixed_noises, state_digest, HYPOTHESES
from n01.response008_fix01 import Response008Fix01Engine
from n01.observation008 import observe_generated_carrier


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):digest.update(part)
    return digest.hexdigest()
def write(path,value):path.write_text(json.dumps(serial(value),indent=2,allow_nan=False)+'\n',encoding='utf8')
def sources():
    names=['src/n01/model008_fix01.py','src/n01/response008_fix01.py','scripts/check_window008_fix01_cuda.py','scripts/check_window008_engine.py']
    names+=['src/n01/'+name for name in ('model.py','model008.py','response008.py','losses008.py','losses008_v2.py','observation008.py','electronics_v3.py','extension006.py','data.py','train.py')]
    return {name:sha(ROOT/name) for name in names}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out)
    if os.name=='nt' and out.resolve().drive.upper() not in ('E:','D:'):raise ValueError('Outputs must stay on E/D')
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'window008_config.json').read_text(encoding='utf8'))
    recovery=json.loads((ROOT/'window008_fix01_config.json').read_text(encoding='utf8'))
    report=dict(schema='n01/window008-fix01-cuda-engineering/v1',status='running_engineering_only',
        pid=os.getpid(),parent_pid=os.getppid(),started_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256=sources(),base_config_sha256=sha(ROOT/'window008_config.json'),
        recovery_config_sha256=sha(ROOT/'window008_fix01_config.json'),
        scientific_G_updates=0,scientific_D_updates=0,engineering_G_updates=0,engineering_D_updates=0,
        DEV_numeric_decodes=0,actual_REAL_numeric_rows_read=0,actual_SOURCE_inference_rows=0,
        historical_latent_resampling=False,method_settings_selected=False,steps=[],
        model_forwards={key:dict(calls_attempted=0,rows_attempted=0,calls_completed=0,rows_completed=0)
                        for key in ('historical_SOURCE','artificial_SOURCE')})
    engine=None;base_G=0;base_D=0
    def meter(model,key):
        original=type(model).signal_forward
        def forward(self,source,response):
            ledger=report['model_forwards'][key]
            ledger['calls_attempted']+=1;ledger['rows_attempted']+=len(source)
            if key=='historical_SOURCE':report['actual_SOURCE_inference_rows']+=len(source)
            output=original(self,source,response)
            ledger['calls_completed']+=1;ledger['rows_completed']+=len(source)
            return output
        model.signal_forward=types.MethodType(forward,model)
    torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(2)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:cfg['cpu_threads']])
    device=torch.device('cuda:0')
    if not torch.cuda.is_available():raise RuntimeError('Native CUDA required')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    limit=cfg['gpu_memory_limit_GiB']*1024**3
    torch.cuda.set_per_process_memory_fraction(limit/torch.cuda.get_device_properties(device).total_memory,device)
    report.update(device=str(device),GPU=torch.cuda.get_device_name(device),torch_version=torch.__version__,CUDA_build=torch.version.cuda)
    try:
        inputs=recovery['reference_inputs']
        def reference(name):
            path=ROOT/inputs[name]['path']
            if sha(path)!=inputs[name]['sha256']:raise ValueError('Reference hash differs: '+name)
            return path
        checkpoint=torch.load(reference('legacy_failure_checkpoint'),map_location='cpu',weights_only=False)
        context=torch.load(reference('legacy_failure_context'),map_location='cpu',weights_only=False)
        batch=context['batches'][0]
        if context['attempted_G_update']!=3270 or checkpoint['G_updates']!=3269 or len(context['batches'])!=1:
            raise ValueError('Not the bound first failed microbatch')
        if sha(ROOT/cfg['dataset'])!=cfg['dataset_sha256']:raise ValueError('Dataset changed')
        with np.load(ROOT/cfg['dataset'],allow_pickle=False) as archive:
            source_carrier=archive['source_carrier384']
            source=source_carrier[np.asarray(batch['source_rows'])].copy()
        report['actual_SOURCE_carrier_rows_decoded']=len(source_carrier)
        response={key:value.to(device) for key,value in batch['response'].items()}
        source=torch.as_tensor(source,dtype=torch.float64,device=device)
        old=N01Response008(checkpoint['statistics']['model'],'full').to(device)
        fixed=N01Response008Fix01(checkpoint['statistics']['model'],'full').to(device)
        for model in (old,fixed):
            model.load_state_dict(checkpoint['model'],strict=True);model.set_stage('joint_stochastic');model.eval();model.requires_grad_(False)
            meter(model,'historical_SOURCE')
        hashes=[state_digest(model.state_dict()) for model in (old,fixed)]
        with torch.no_grad():
            try:old.signal_forward(source,response)
            except InvalidWaveformError as error:
                if error.rows!=[22] or error.reason!='Invalid Window008 shape canonical charge':raise ValueError('Original native failure reason/row changed')
                original_error=dict(reason=error.reason,rows=error.rows,values=error.values)
            else:raise RuntimeError('Original failure did not reproduce on native CUDA')
            output=fixed.signal_forward(source,response)
            basis_error=float(output['basis_canonical_charge'].abs().max())
            shape_error=float((output['shape_Q']-output['base_shape_Q']).abs().max())
            signal_error=float(output['canonical_charge_identity_residual'].abs().max())
            if basis_error>1e-10 or shape_error>1e-10 or signal_error>1e-10:
                raise RuntimeError('Fixed canonical decomposition exceeds engineering tolerance')
            if float(output['base_shape_Q'].min())<=1e-6 or float(output['shape_Q'].min())<=1e-6 or not bool(torch.isfinite(output['signal_pre']).all()):
                raise RuntimeError('Fixed historical failure remains invalid')
            if bool((output['signal_pre'][:,:100]!=0).any()):raise RuntimeError('Signal prefix leakage')
            noise_path=ROOT/cfg['noise_reference']['path']
            if sha(noise_path)!=cfg['noise_reference']['sha256']:raise ValueError('Frozen005 bytes changed')
            noise_checkpoint=torch.load(noise_path,map_location='cpu',weights_only=False)
            extension=make_extension(noise_checkpoint,device)
            noises=fixed_noises(extension,{key:value.to(device) for key,value in batch['noise005'].items()},batch['extension'].to(device))
            observed={}
            for h in HYPOTHESES:
                observation=observe_generated_carrier(output['signal_pre']+noises[h],ste=False,
                    q_floor=checkpoint['statistics']['loss']['q_floor'],return_carrier=True)
                if observation['waveform256'].shape!=(512,256) or not bool(torch.isfinite(observation['waveform256']).all()):
                    raise RuntimeError('Observed corrected failure is not finite Bx256')
                observed[h]=dict(rows=512,nonpositive_Q=int((observation['Q']<=0).sum()),
                    search_edges=int(observation['flags']['peak_at_search_boundary'].sum()))
            report['failure_case']=dict(rows=512,original_error=original_error,
                fixed_row22_Q=float(output['shape_Q'][22]),minimum_base_Q=float(output['base_shape_Q'].min()),
                basis_canonical_absmax=basis_error,shape_Q_minus_base_absmax=shape_error,
                signal_Q_identity_absmax=signal_error,prefix100_exact_zero=True,observation=observed)
        if [state_digest(model.state_dict()) for model in (old,fixed)]!=hashes:raise RuntimeError('Inference mutated model states')
        del old,fixed,source,response,output,extension,noises,checkpoint,context,source_carrier
        gc.collect();torch.cuda.empty_cache()
        data,statistics,fixture_sha=fixture(rows=1024)
        artificial_cfg=copy.deepcopy(cfg);artificial_cfg['engineering_fixture']=True
        common=common_initial_state(statistics['model'],cfg['seeds']['initialization'])
        identity=dict(run_id=recovery['run_id'],config_sha256=report['recovery_config_sha256'],
            source_manifest_sha256='e'*64,base_configuration_sha256=report['base_config_sha256'],
            base_source_manifest_sha256=recovery['legacy_source_manifest_sha256'])
        report.update(fixture_sha256=fixture_sha,formal_batch=[2,256,2,512],artificial_rows=1024)
        for arm in ('full','simple'):
            base_G=report['engineering_G_updates'];base_D=report['engineering_D_updates']
            engine=Response008Fix01Engine(data,statistics,artificial_cfg,noise_checkpoint,common,arm=arm,
                recovery_identity=identity,device=device,engineering_fixture=True)
            meter(engine.model,'artificial_SOURCE')
            for stage in ('mean_response','joint_stochastic'):
                engine.set_stage(stage);torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                began=time.monotonic();metrics=engine.step(audit=True,verify_replay=True);torch.cuda.synchronize(device)
                peak=max(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
                if peak>limit or not metrics['same_version_replay_verified']:raise RuntimeError('Synthetic formal-batch capacity/replay failed')
                report['steps'].append(dict(arm=arm,stage=stage,seconds=time.monotonic()-began,
                    peak_GPU_bytes=peak,metrics=metrics))
                report['engineering_G_updates']=base_G+engine.update
                report['engineering_D_updates']=base_D+sum(engine.d_updates.values())
            if arm=='full':report['direct_vs_two_pass_VJP']=vjp_probe(engine)
            engine.extension.assert_frozen();del engine;engine=None;gc.collect();torch.cuda.empty_cache()
        if report['engineering_G_updates']!=4 or report['engineering_D_updates']!=6:raise RuntimeError('Engineering optimizer count differs')
        if report['model_forwards']['artificial_SOURCE']['rows_completed']!=8200:raise RuntimeError('Synthetic forward ledger differs')
        if sources()!=report['source_sha256'] or sha(ROOT/'window008_fix01_config.json')!=report['recovery_config_sha256']:
            raise RuntimeError('Engineering source/config changed during execution')
        report['status']='passed_native_fix01_engineering_and_historical_failure_inference'
        report['finished_UTC']=dt.datetime.now(dt.timezone.utc).isoformat()
        write(out/'ENGINEERING_RECEIPT.json',report)
        print(json.dumps(dict(status=report['status'],engineering_G=4,engineering_D=6,scientific_G=0,
            actual_SOURCE_inference_rows=report['actual_SOURCE_inference_rows'],failure_case=report['failure_case'])))
    except BaseException as error:
        if engine is not None:
            report['engineering_G_updates']=base_G+engine.update
            report['engineering_D_updates']=base_D+sum(engine.d_updates.values())
        report.update(status='failed_engineering_preserved',error=repr(error));write(out/'ENGINEERING_RECEIPT.json',report);raise


if __name__=='__main__':main()
