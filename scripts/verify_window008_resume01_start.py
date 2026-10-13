"""One startup-only native lineage/replay receipt; never polls a training job."""
import datetime as dt
import hashlib
import json
import shlex
import subprocess
from window008_resume01_runtime import ROOT,read,sha,write_new,require

PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'

def main():
    cfg=read(ROOT/'window008_resume01_config.json');remote=cfg['remote_root']
    code='''from pathlib import Path
import hashlib,json,subprocess
r=Path(ROOT_VALUE)
binding=json.loads((r/'runs/science/BINDING.json').read_text())
launch=json.loads((r/'runs/supervisor/launch.json').read_text())
with (r/'runs/science/train/full/training.jsonl').open('rb') as f:first=f.readline()
pid=binding['pid'];supervisor=launch['supervisor_pid'];wrapper=launch['child_pid']
processes={str(p):dict(argv=(Path('/proc')/str(p)/'cmdline').read_bytes().decode().strip('\\0').split('\\0'),
 status=(Path('/proc')/str(p)/'status').read_text(),stat=(Path('/proc')/str(p)/'stat').read_text()) for p in (pid,supervisor,wrapper)}
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(binding=binding,launch=launch,first_update=json.loads(first),
 first_update_line_sha256=hashlib.sha256(first).hexdigest(),processes=processes,
 resource=json.loads((r/'runs/supervisor/resource_status.json').read_text()),GPU_compute_processes=apps,
 complete_present=(r/'runs/science/TRAINING_COMPLETE.json').exists(),failure_present=(r/'runs/science/FAILURE.json').exists(),
 native_exit_present=(r/'runs/supervisor/exit_receipt.json').exists())))
'''.replace('ROOT_VALUE',repr(remote))
    command=' '.join([shlex.quote(PYTHON),'-B','-c',shlex.quote(code)])
    native=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ForwardX11=no','rylan@100.82.231.102',command],text=True,encoding='utf8'))
    first=native['first_update'];binding=native['binding'];launch=native['launch']
    require(first['arm']=='full' and first['stage']=='joint_stochastic' and first['G_updates']==6001
            and first['inherited_G_updates']==6000 and first['new_G_updates']==1,'First scientific step differs')
    require(first['same_version_replay_verified'] is True,'First scientific same-version replay not verified')
    require(binding['parent_pid']==launch['child_pid'],'Native trainer/timeout chain differs')
    require(sha(ROOT/cfg['source_manifest'])==binding['recovery']['source_manifest_sha256'],'Native source binding differs')
    trainer=native['processes'][str(binding['pid'])]['argv']
    require('-m' in trainer and trainer[trainer.index('-m')+1]=='n01.train_window008_resume01','Native trainer identity differs')
    gpu_pids=[int(row.split(',')[0]) for row in native['GPU_compute_processes'].splitlines() if row.strip()]
    require(gpu_pids==[binding['pid']],'Startup GPU computation identity differs')
    require(not any(native[k] for k in ('failure_present','complete_present','native_exit_present')),'Startup terminal state differs')
    require(native['resource']['peak_rss_bytes']<=8*1024**3,'Startup RSS cap exceeded')
    record=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='correct_scientific_start_verified_once',
        native=native,scope='One startup proof only; fixed endpoints and native exit remain pending.',
        scientific_status='response_match_not_established')
    write_new(ROOT/'receipts/window008_resume01_20261013/START_NATIVE_VERIFICATION.json',record)
    print(json.dumps(dict(status=record['status'],pid=binding['pid'],supervisor_pid=launch['supervisor_pid'],
        first_new_G_update=1,arm_G_update=6001,same_version_replay_verified=True,peak_rss_bytes=native['resource']['peak_rss_bytes'])))

if __name__=='__main__':main()
