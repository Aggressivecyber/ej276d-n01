"""Execute the preregistered two-stage audit sequentially, without generator fits."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/science'

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    tmp.replace(path)

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def owned_bytes():
    # The whole newly-created run root, including source, inputs, cache and logs.
    total=0
    for p in ROOT.rglob('*'):
        try:
            if p.is_file():
                total+=p.stat().st_size
        except FileNotFoundError:
            pass  # Concurrent atomic progress-file replacement.
    return total

def stage_completion(name):
    directory=OUT/name
    report_name='ORIGIN_REPORT.json' if name=='origin' else 'RR_CALIBRATION_REPORT.json'
    reportp=directory/report_name
    receipt=json.loads((directory/'COMPLETE.json').read_text(encoding='utf8'))
    report=json.loads(reportp.read_text(encoding='utf8'))
    if receipt['status']!='complete' or receipt['report_sha256']!=hashlib.sha256(reportp.read_bytes()).hexdigest():
        raise RuntimeError('Scientific completion receipt/hash mismatch: '+name)
    if name=='real_real':
        if (report['repeats'],report['evaluation_classifier_updates'],report['noise_model_training_updates'],report['response_model_training_updates'])!=(128,25600,0,0):
            raise RuntimeError('REAL-REAL scientific accounting mismatch')
        if report['original_VAL_waveforms_decoded'] or report['old_noise_gate_modified']:
            raise RuntimeError('REAL-REAL scope violated')
    else:
        if receipt['rows_completed']!=21047 or receipt['bitwise_target_equal'] is not True:
            raise RuntimeError('Origin row coverage/bitwise reproduction incomplete')
        if receipt['validation_waveforms_decoded'] or receipt['noise_training_updates'] or receipt['response_training_updates']:
            raise RuntimeError('Origin scientific scope violated')
    return {'report':str(reportp.relative_to(ROOT)), 'report_sha256':receipt['report_sha256'], 'scientific_completion_verified':True}

def main():
    configp=ROOT/'noise_origin_003_config.json'
    config=json.loads(configp.read_text(encoding='utf8'))
    if os.name=='nt' or str(ROOT)!=config['remote_root']:
        raise RuntimeError('Sequential execution is bound to the dedicated remote audit root')
    manifest=json.loads((ROOT/'receipts/noise_origin_audit_003_20261012/IMPLEMENTATION_BINDING.json').read_text(encoding='utf8'))
    for entry in manifest['files']:
        path=ROOT/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise RuntimeError('Implementation binding changed: '+entry['path'])
    if (OUT/'SCIENCE_EXIT.json').exists() or (OUT/'status.json').exists():
        raise FileExistsError('Preserve previous audit; no unrecorded restart')
    OUT.mkdir(parents=True,exist_ok=True)
    status={'experiment_id':config['experiment_id'],'started_UTC':utc(),'pid':os.getpid(),'stage':'starting',
            'noise_training_updates':0,'response_training_updates':0,'classifier_updates_budget':25600,
            'stages':[]}
    write(OUT/'status.json',status)
    code=0
    proc=None
    try:
        for name,script,extra in [('origin','audit_noise_origin.py',[]),('real_real','calibrate_real_real_null.py',['--device','cuda'])]:
            status.update(stage=name,updated_UTC=utc())
            write(OUT/'status.json',status)
            cmd=[sys.executable,'-B',str(ROOT/'scripts'/script),'--config',str(configp),'--out',str(OUT/name),
                 '--container',config['remote_training_container'],'--identities',str(ROOT/'data/PARENT_IDENTITIES.npz')]+extra
            start=time.monotonic()
            receipt={'stage':name,'started_UTC':utc(),'command':cmd}
            proc=subprocess.Popen(cmd,cwd=ROOT)
            reason=None
            maxbytes=0
            while proc.poll() is None:
                size=owned_bytes()
                maxbytes=max(maxbytes,size)
                if time.monotonic()-start>config['resource']['wall_limit_s']:
                    reason='registered_stage_wall_limit_exceeded'
                elif size>config['resource']['output_limit_GiB']*1024**3:
                    reason='registered_owned_root_size_limit_exceeded'
                if reason:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill();proc.wait()
                    break
                time.sleep(1)
            stagecode=proc.wait()
            receipt.update(native_exit_code=stagecode,timeout=reason=='registered_stage_wall_limit_exceeded',
                           termination_reason=reason,peak_observed_owned_root_bytes=maxbytes,
                           output_monitor_scope='entire_dedicated_run_root_including_source_inputs_cache_logs')
            if reason:
                stagecode=124 if receipt['timeout'] else 1
            receipt.update(ended_UTC=utc(),elapsed_s=time.monotonic()-start)
            status['stages'].append(receipt)
            write(OUT/(name+'_EXIT.json'),receipt)
            if stagecode!=0:
                code=stagecode if stagecode>0 else 1
                raise RuntimeError(f'{name} failed with code {stagecode}')
            receipt.update(stage_completion(name))
            write(OUT/(name+'_EXIT.json'),receipt)
            size=owned_bytes()
            if size>config['resource']['output_limit_GiB']*1024**3:
                raise RuntimeError('Registered output limit exceeded')
        status.update(stage='completed',completed=True)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()
        code=code or 1
        status.update(stage='failed',completed=False,error=f'{type(exc).__name__}: {exc}')
    status.update(ended_UTC=utc(),exit_code=code)
    write(OUT/'status.json',status)
    write(OUT/'SCIENCE_EXIT.json',status)
    return code

if __name__=='__main__':
    raise SystemExit(main())
