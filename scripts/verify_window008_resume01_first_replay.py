"""Compare the repeated science6001 metrics with the preserved pre-shutdown step."""
import json
from window008_resume01_runtime import ROOT,read,sha,write_new,require

def main():
    rec=ROOT/'receipts/window008_resume01_20261013'
    native=read(rec/'START_NATIVE_VERIFICATION.json');new=native['native']['first_update']
    collection=read(ROOT/'receipts/window008_fix01_20261013/USER_STOP_COLLECTION.json')
    name='runs/science/train/full/training.jsonl';path=ROOT/'runs/window008_fix01_20261013'/name
    require(sha(path)==collection['files'][name]['sha256'],'Preserved original log differs')
    previous=None
    with path.open(encoding='utf8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['G_updates']==6001:previous=row;break
    require(previous is not None,'Original science6001 record missing')
    ignore={'utc','update_seconds','new_G_updates','new_D_updates','inherited_G_updates','inherited_D_updates',
        'recovery','checkpoint_schema','lineage','same_version_replay_verified','gradient_audit','gradients'}
    ignore|={'gpu_allocated_bytes','gpu_reserved_bytes','recovery_run_id'}
    equal=[];differences={}
    for key in sorted(set(previous)&set(new)-ignore):
        if previous[key]==new[key]:equal.append(key)
        else:differences[key]=dict(previous=previous[key],resumed=new[key])
    result=dict(status='exact_shared_scientific_metrics_match' if not differences else 'shared_metric_differences_preserved',
        arm_G_update=6001,old_log_sha256=sha(path),new_native_receipt_sha256=sha(rec/'START_NATIVE_VERIFICATION.json'),
        equal_keys=equal,differences=differences,declared_exclusions=sorted(ignore),additional_model_forwards=0,
        additional_optimizer_updates=0,scope='Re-executed registered first step metadata; no additional science or new random draws.')
    write_new(rec/'FIRST_STEP_REPLAY_COMPARISON.json',result)
    print(json.dumps(dict(status=result['status'],equal_keys=equal,different_keys=sorted(differences))))

if __name__=='__main__':main()
