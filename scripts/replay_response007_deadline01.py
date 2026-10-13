"""Exact saved-input engineering replay after the bounded science job exits."""
from pathlib import Path
import copy
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from n01.data import load_training,sha256,write_json
from n01.evaluate_integration006 import clean
from n01.evaluate_response007 import generate_split,paired_width_bootstrap,distance_receipt
from n01.losses_v2 import N01LossV2
from n01.noise_validation import balanced_real_domains
from n01.response007 import ARMS,HYPOTHESES,SignalResponse007,make_extension,state_digest,loss_tensor_statistics

def main():
    out=ROOT/'runs/verification_deadline01';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();counts={};models={}
    def counted(obj,name,key):
        method=getattr(obj,name)
        def wrapper(*args,**kwargs):
            x=args[0];n=len(next(iter(x.values()))) if isinstance(x,dict) else len(x)
            record=counts.setdefault(key,dict(attempted_rows=0,completed_rows=0,calls=0))
            record['attempted_rows']+=n;record['calls']+=1
            result=method(*args,**kwargs);record['completed_rows']+=n
            return result
        setattr(obj,name,wrapper)
    try:
        assert os.name=='posix' and str(ROOT)=='/home/rylan/projects/ej276d-n01/response_007_20261013_deadline01'
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,13,3,30,tzinfo=dt.timezone.utc),'No replay starts after cutoff'
        cfg=json.loads((ROOT/'response007_config.json').read_text(encoding='utf8'))
        bound=json.loads((ROOT/'response007_deadline01_config.json').read_text(encoding='utf8'))
        science=ROOT/'runs/science';complete=json.loads((science/'COMPLETE.json').read_text())
        native=json.loads((ROOT/'runs/supervisor/exit_receipt.json').read_text())
        assert native['exit_code']==0 and complete['G_updates']==8000 and not complete['original_19000_per_arm_endpoint_reached']
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        assert not apps,'Another GPU process is active; no replay'
        manifest=json.loads((ROOT/'RESPONSE007_DEADLINE01_SOURCE_MANIFEST.json').read_text())
        for relative,digest in manifest['files'].items():assert sha256(ROOT/relative)==digest,relative
        for relative,digest in complete['files'].items():assert sha256(science/relative)==digest,relative
        assert sha256(cfg['remote_training_container'])==cfg['train_container_sha256']
        write_json(out/'START.json',dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='verified_bindings_before_TRAIN_decode',
            scientific_complete_sha256=sha256(science/'COMPLETE.json'),implementation_sha256=sha256(Path(__file__)),
            source_manifest_sha256=sha256(ROOT/'RESPONSE007_DEADLINE01_SOURCE_MANIFEST.json'),new_training_updates=0,new_unique_draws=0,DEV_decodes=0))
        torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(2)
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:cfg['cpu_threads']])
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
        device=torch.device('cuda:0')
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_limit_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
        data=load_training(cfg['remote_training_container'],ROOT/'data/PARENT_IDENTITIES.npz')
        stats=json.loads((ROOT/cfg['references']['statistics001']['path']).read_text())
        hashes={}
        for a in ARMS:
            checkpoint=torch.load(science/'train'/a/'bounded_04000.pt',map_location='cpu',weights_only=False)
            assert checkpoint['G_updates']==4000
            m=SignalResponse007(stats['model'],a);m.load_state_dict(checkpoint['model']);m.set_stage('joint_stochastic');m.eval().requires_grad_(False);m.to(device)
            models[a]=m;hashes[a]=state_digest(m.state_dict());counted(m,'signal_forward','response/'+a)
        noise=torch.load(ROOT/cfg['references']['noise005']['path'],map_location='cpu',weights_only=False)
        extension=make_extension(noise,device)
        counted(extension.model005,'forward','noise005')
        counted(extension,'extend_prefix','extension_all_hypotheses')
        diag=copy.deepcopy(cfg);diag['seeds'].update(bound['diagnostic_seeds'])
        selected=np.random.default_rng(diag['seeds']['selected_TRAIN']).permutation(len(data['source']))[:2048]
        real_a,real_b=balanced_real_domains(data['real_groups'],2048,diag['seeds']['real_TRAIN'])
        original=science/'TRAIN_diagnostics'
        saved=json.loads((original/'SUMMARY.json').read_text())
        with torch.no_grad():
            generated=generate_split('TRAIN',data,models,extension,stats,diag,out,selected,device)
            sensitivity,bootstrap=paired_width_bootstrap(generated,data,stats,diag,'TRAIN')
            objective=N01LossV2(loss_tensor_statistics(stats['loss']),seed=diag['seeds']['train_projection'],evaluation_seed=diag['seeds']['evaluation_projection']).to(device)
            context=objective.create_evaluation_context()
            distances={a:{h:distance_receipt(objective,generated['matched'][a][h],data['real'][real_a],context) for h in HYPOTHESES} for a in ARMS}
            distances['REAL_REAL']=distance_receipt(objective,data['real'][real_b],data['real'][real_a],context)
        array_checks={}
        for name in ['TRAIN_observables.npz','TRAIN_curves.npz','TRAIN_sample_waveforms.npz']:
            with np.load(original/name,allow_pickle=False) as old,np.load(out/name,allow_pickle=False) as new:
                assert old.files==new.files,name+' members'
                for key in old.files:
                    a,b=old[key],new[key]
                    assert a.shape==b.shape and a.dtype==b.dtype,(name,key,'schema')
                    equal=np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)
                    assert equal,(name,key,'values')
                array_checks[name]=len(old.files)
        old_generation=json.loads((original/'TRAIN_generation.json').read_text())
        new_generation=json.loads((out/'TRAIN_generation.json').read_text())
        assert old_generation==new_generation,'Generation hashes, statistics or engineering differ'
        with np.load(original/'TRAIN_bootstrap.npz',allow_pickle=False) as old:assert np.array_equal(old['statistics'],bootstrap,equal_nan=True),'Bootstrap draws differ'
        assert saved['sensitivity']==clean(sensitivity),'Sensitivity summary differs'
        assert saved['matched_distances']==clean(distances),'Matched distances differ'
        assert saved['engineering']==clean(generated['engineering']),'Engineering differs'
        assert saved['population_summary']==clean(generated['summary']),'Population summary differs'
        for a,m in models.items():assert state_digest(m.state_dict())==hashes[a],'Replay changed checkpoint'
        extension.assert_frozen()
        result=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='passed',
            actual_engineering_forward_counts=counts,array_members_bitwise_verified=array_checks,
            source_parents=6593,draws_per_parent=4,generated_view_rows=2*4*6593*4,REAL_TRAIN_readout_rows=21047,
            generation_chunks_verified=len(old_generation['chunks']),bootstrap_draws_bitwise_verified=1000,
            matched_distribution_comparisons_verified=7,all_population_summaries_equal=True,
            source_model_states_unchanged=True,new_training_updates=0,new_unique_random_draws=0,DEV_decodes=0,
            original_full_endpoint_reached=False,scientific_status='response_match_not_established',
            source_manifest_sha256=sha256(ROOT/'RESPONSE007_DEADLINE01_SOURCE_MANIFEST.json'),
            scientific_complete_sha256=sha256(science/'COMPLETE.json'),implementation_sha256=sha256(Path(__file__)),elapsed_seconds=time.monotonic()-started)
        write_json(out/'REPLAY_VERIFICATION.json',result);print(json.dumps(result),flush=True)
    except BaseException as error:
        write_json(out/'FAILURE.json',dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='failed_preserved',error=repr(error),
            traceback=traceback.format_exc(),actual_engineering_forward_counts=counts,new_training_updates=0,new_unique_random_draws=0,
            elapsed_seconds=time.monotonic()-started));raise

if __name__=='__main__':main()
