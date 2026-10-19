"""Bounded golden-output validation of the research API, never model selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]

import numpy as np
import torch
from n01.data import write_json
from n01.finalization.api import load_response
from n01.finalization.native_storage import load_native
from n01.finalization.output import save_result
from n01.inference028 import verified_path028
from n01.model010 import tensor_state_digest010
from run_response039 import check_resources


def compare(actual, expected, label, *, exact):
    if isinstance(actual,torch.Tensor):
        actual=actual.detach().cpu().numpy()
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
        raise AssertionError('Golden shape/dtype changed: '+label)
    byte_equal=actual.tobytes()==expected.tobytes()
    if exact and not byte_equal:
        raise AssertionError('Golden bytes changed: '+label)
    if not exact and not np.allclose(actual,expected,atol=1e-12,rtol=0,equal_nan=True):
        raise AssertionError('Golden float tolerance exceeded: '+label)
    finite=np.isfinite(actual)&np.isfinite(expected)
    difference=float(np.abs(actual[finite]-expected[finite]).max(initial=0)) if actual.dtype.kind=='f' else 0.
    return dict(label=label,byte_equal=byte_equal,maximum_absolute_error=difference,exact_required=exact)


@torch.no_grad()
def run(p,out,device,live):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    write_json(out/'STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        config_sha256=p['_config_sha256'],run_id=p['run_id'],resources=p['resources']))
    for name,digest in p['source_pins'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Validation source changed: '+name)
    for record in p['golden_files'].values():verified_path028(record)
    setup=json.loads((ROOT/p['setup038']['path']).read_text())
    with np.load(verified_path028(setup['inputs']['dataset']),allow_pickle=False) as archive:
        source=archive['source_carrier384'].copy();ids=archive['source_ids'].copy();roles=archive['source_role'].copy()
    if source.shape!=(6593,384) or source.dtype!=np.float64 or set(np.unique(roles))!={'TRAIN'}:
        raise ValueError('Frozen SOURCE TRAIN population changed')
    counts=dict(TRAIN_archives_decoded=1,SOURCE_members_decoded=3,REAL_members_decoded=0,
        DEV_archives_decoded=0,legacy_golden_archives_decoded=1,legacy_golden_cases=0,
        legacy_golden_tensor_checks=0,S_native_golden_archives_decoded=0,S_operator_archives_decoded=0,
        S_golden_cases=0,additional_S_API_calls=0,CLI_calls=0,AD_queries=0,optimizer_steps=0,
        new_response_candidates=0,full_population_metric_evaluations=0)
    live['execution_counts']=counts
    input_hash=hashlib.sha256(source.tobytes()).hexdigest()
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    torch.set_num_threads(1)  # Original028 golden outputs were produced with one CPU thread.
    legacy=load_response(verified_path028(setup['reference028']),'cpu')
    live['legacy_accounting']=legacy.accounting
    oldcfg=legacy.reference.config;example=oldcfg['example'];selected=example['source_rows']
    x=torch.from_numpy(source[selected].copy());parent_ids=ids[selected].copy()
    checks=[]
    with np.load(p['golden_files']['legacy028']['path'],allow_pickle=False) as golden:
        case_index=0
        for head in example['model_seeds']:
            for call in example['calls_per_seed']:
                live['active_call']=f'legacy028/{head}/{call["name"]}'
                result=legacy.sample(x,parent_ids,example['draws'],example['sampling_seed'],head,
                    call['hypothesis'],True,shape_enabled=call['shape_enabled'],charge_enabled=call['charge_enabled'])
                for name in golden.files:
                    if name in ('parent_ids','source_rows','row_parent','draw'):continue
                    if name.startswith('c_'):actual=result['contributions'][name[2:]]
                    elif name.startswith('flag_'):actual=result['flags'][name[5:]]
                    else:actual=result[name]
                    checks.append(compare(actual,golden[name][case_index],f'028/{case_index}/{name}',exact=True))
                    counts['legacy_golden_tensor_checks']+=1
                assert np.array_equal(result['parent_ids'],golden['parent_ids'])
                assert np.array_equal(result['row_parent'].numpy(),golden['row_parent'])
                assert np.array_equal(result['draw'].numpy(),golden['draw'])
                if case_index==0:
                    save_result(out/'legacy028_example',result,dict(original_golden=True,scientific_acceptance=False))
                case_index+=1;counts['legacy_golden_cases']+=1
                check_resources(p,out,started)
                write_json(out/'PROGRESS.json',dict(phase='legacy028_golden',elapsed_seconds=time.monotonic()-started,
                    execution_counts=counts,legacy_accounting=legacy.accounting))
    legacy_after=dict(A=tensor_state_digest010(legacy.reference.models[0].baseline),
        extension=tensor_state_digest010(legacy.reference.extension),
        **{f'model{i}':tensor_state_digest010(m) for i,m in legacy.reference.models.items()})
    assert legacy_after==legacy.reference.frozen_digests
    legacy_accounting=legacy.accounting
    del legacy
    torch.set_num_threads(p['resources']['CPU_threads'])
    response=load_response(ROOT/p['api_config']['path'],str(device))
    live['S_accounting']=response.accounting
    state_before={str(i):tensor_state_digest010(m) for i,m in response.models.items()}
    state_before['extension']=tensor_state_digest010(response.extension)
    native_root=Path(p['S_native_root'])
    x=torch.from_numpy(source[:128].copy());parent_ids=ids[:128].copy()
    reference_result=None
    for head in (0,1):
        with np.load(native_root/f'seed{head}_S_OPERATOR.npz',allow_pickle=False) as archive:
            operator={k:archive[k][:128].copy() for k in ('fixed_mean_V','mean_correction_local_V','random_correction_local_V')}
        counts['S_operator_archives_decoded']+=1
        for h in ('fast','slow','common_level'):
            live['active_call']=f'S/{head}/{h}'
            result=response.sample(x,parent_ids,1,p['FIT_seeds'][head],head,h,True)
            with np.load(native_root/f'seed{head}_S_{h}_NATIVE.npz',allow_pickle=False) as archive:
                saved={k:archive[k][:128].copy() for k in archive.files}
            adc,wave,baseline=load_native(saved);counts['S_native_golden_archives_decoded']+=1
            for name,expected in (('carrier_adc',adc),('window_V',wave),('baseline100',baseline),
                                  ('baseline_mean_V',saved['baseline_mean_V'])):
                checks.append(compare(result[name],expected,f'S/{head}/{h}/{name}',exact=False))
            codes=np.rint((result['carrier_adc'].cpu().numpy()+.5)*16383).astype(np.uint16)
            checks.append(compare(codes,saved['ADC_codes'],f'S/{head}/{h}/ADC_codes',exact=True))
            checks.append(compare(result['peak_index'],saved['peak_index'],f'S/{head}/{h}/peak',exact=True))
            for name,flag in result['flags'].items():
                checks.append(compare(flag,saved[name+'_flag'],f'S/{head}/{h}/flag/{name}',exact=True))
            c=result['contributions']
            for name,expected in (('BASE054_mean_pre_V',operator['fixed_mean_V']),
                                  ('S_mean_correction_pre_V',operator['mean_correction_local_V']),
                                  ('S_random_correction_pre_V',operator['random_correction_local_V'])):
                actual=c[name] if name=='BASE054_mean_pre_V' else c[name][:,110:176]
                checks.append(compare(actual,expected,f'S/{head}/{h}/{name}',exact=False))
            assert result['applicability']['model_accepted'] is False
            if head==0 and h=='fast':
                reference_result=result
                save_result(out/'S_seed0_fast_example',result,dict(rows='first128 engineering golden rows',scientific_acceptance=False))
            counts['S_golden_cases']+=1
            check_resources(p,out,started)
            write_json(out/'PROGRESS.json',dict(phase='S_golden',head=head,hypothesis=h,
                elapsed_seconds=time.monotonic()-started,execution_counts=counts,S_accounting=dict(response.accounting)))
    for label,labels in (('exact_replay',parent_ids),('ID_labels_only',np.array(['alias_'+str(i) for i in range(128)]))):
        live['active_call']=label
        again=response.sample(x,labels,1,p['FIT_seeds'][0],0,'fast',True)
        for key in ('window_V','carrier_adc','peak_index','row_parent','draw'):
            checks.append(compare(again[key],reference_result[key].cpu().numpy(),label+'/'+key,exact=True))
        for key in reference_result['contributions']:
            checks.append(compare(again['contributions'][key],reference_result['contributions'][key].cpu().numpy(),label+'/'+key,exact=True))
        assert tuple(again['parent_ids'])==tuple(labels)
        counts['additional_S_API_calls']+=1
    multi=[]
    for i,seed in enumerate((p['multidraw_seed'],p['multidraw_seed'],p['multidraw_seed']+1)):
        live['active_call']=f'multidraw/{i}'
        result=response.sample(x[:2],parent_ids[:2],3,seed,1,'slow',i!=1)
        assert result['row_parent'].tolist()==[0,0,0,1,1,1] and result['draw'].tolist()==[0,1,2,0,1,2]
        multi.append(result['window_V'].cpu().numpy().copy())
        counts['additional_S_API_calls']+=1
    checks.append(compare(multi[1],multi[0],'multidraw/replay_without_contributions',exact=True))
    assert multi[0].tobytes()!=multi[2].tobytes()
    invalid=[dict(x=x.float()),dict(x=torch.zeros(2,480,dtype=torch.float64)),
             dict(x=torch.full((2,384),float('nan'),dtype=torch.float64)),
             dict(draws=0),dict(seed=-1),dict(hypothesis='unknown')]
    before=dict(response.accounting)
    for item in invalid:
        try:
            response.sample(item.get('x',x[:2]),parent_ids[:2],item.get('draws',1),item.get('seed',123),
                            0,item.get('hypothesis','fast'))
        except ValueError:
            pass  # Expected public-input rejection, asserted for every invalid case.
        else:
            raise AssertionError('Invalid public input was accepted')
    assert dict(response.accounting)==before
    state_after={str(i):tensor_state_digest010(m) for i,m in response.models.items()}
    state_after['extension']=tensor_state_digest010(response.extension)
    assert state_after==state_before
    assert hashlib.sha256(source.tobytes()).hexdigest()==input_hash
    assert torch.equal(cpu_rng,torch.get_rng_state())
    if device.type=='cuda':assert torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
    S_accounting=dict(response.accounting)
    # CLI exercise uses the same frozen S rule and only the first two TRAIN rows.
    cli_input=out/'PRIVATE_CLI_INPUT.npz'
    np.savez_compressed(cli_input,source_waveforms=source[:2],parent_ids=ids[:2])
    command=[sys.executable,'-B',str(ROOT/'finalization_run.py'),'--config',str(ROOT/p['api_config']['path']),
        '--config-sha256',p['api_config']['sha256'],'--input',str(cli_input),
        '--input-sha256',hashlib.sha256(cli_input.read_bytes()).hexdigest(),'--output',str(out/'CLI_example'),
        '--draws','2','--seed',str(p['multidraw_seed']),'--model-seed','0','--noise-hypothesis','fast',
        '--device',str(device),'--threads',str(p['resources']['CPU_threads']),
        '--gpu-gib',str(p['resources']['CLI_allocator_GiB']),'--contributions']
    live['active_call']='native_CLI'
    cli=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
    (out/'CLI.stdout.txt').write_text(cli.stdout);(out/'CLI.stderr.txt').write_text(cli.stderr)
    if cli.returncode:raise RuntimeError('CLI inference failed: '+cli.stderr)
    counts['CLI_calls']=1
    cli_receipt=json.loads((out/'CLI_example/RECEIPT.json').read_text())
    cli_accounting=cli_receipt['provenance']['operation_accounting']
    live['CLI_accounting']=cli_accounting
    assert cli_receipt['rows']==4 and cli_receipt['status']=='PASS_LOSSLESS_REOPEN'
    assert counts==p['operation_counts_exact'],(counts,p['operation_counts_exact'])
    assert S_accounting==p['S_API_operation_counts_exact'],(S_accounting,p['S_API_operation_counts_exact'])
    assert cli_accounting==p['CLI_API_operation_counts_exact'],(cli_accounting,p['CLI_API_operation_counts_exact'])
    for key,value in legacy_accounting['completed'].items():
        assert value<=oldcfg['budget_caps'][key],key
    assert legacy_accounting['completed']['sample_calls']==12
    check_resources(p,out,started)
    result=dict(status='PASS_RESEARCH_API_GOLDENS',UTC=datetime.now(timezone.utc).isoformat(),
        run_id=p['run_id'],checks=checks,execution_counts=counts,legacy_accounting=legacy_accounting,
        S_accounting=S_accounting,CLI_accounting=cli_accounting,
        checkpoint_loads_total=legacy_accounting['completed']['CP_loads']+S_accounting['checkpoint_loads']+cli_accounting['checkpoint_loads'],
        inference_output_rows_total=48+S_accounting['native_observation_rows']+cli_accounting['native_observation_rows'],
        global_CPU_and_CUDA_rng_unchanged=True,input_bytes_unchanged=True,frozen_model_state_unchanged=True,
        parent_IDs_do_not_change_values=True,multidraw_lineage_verified=True,invalid_input_cases=6,
        sampling_seed_changes_outputs=True,full_native_output_roundtrip=True,native_CLI_verified=True,
        old028_original_tensor_checks_exact=True,S_science_status='closed_negative',scientific_acceptance=False,
        full_M5_M6_scientific_plan_complete=False,elapsed_seconds=time.monotonic()-started)
    write_json(out/'RESULT.json',result);check_resources(p,out,started)
    write_json(out/'COMPLETE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),native_exit_expected=0,
        result_sha256=hashlib.sha256((out/'RESULT.json').read_bytes()).hexdigest()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if hashlib.sha256(args.config.read_bytes()).hexdigest()!=args.config_sha256:raise ValueError('Validation config changed')
    p=json.loads(args.config.read_text());p['_config_sha256']=args.config_sha256
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(p['resources']['parent_allocator_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    live={}
    try:
        run(p,args.out,device,live)
    finally:
        error=sys.exc_info()[1]
        if error is not None and args.out.exists():
            write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),automatic_restart=False,**live))


if __name__=='__main__':
    main()
