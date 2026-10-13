"""Reconcile registered zero-update evaluation jobs from saved native snapshots."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import shutil
from window008_resume01_runtime import ROOT, read, sha, write_new

RUN='window008_resume01_20261013'
BASE=ROOT/'receipts'/RUN
SPECS={
    'primary': dict(job_id='N01_WINDOW008_RESUME01_EVALUATION_PRIMARY', relative='runs/evaluation_jobs/primary',
        manifest='WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json', planned_signal_forward_rows=85260,
        planned_observation_rows=255780,additional_unique_signal_rows=85260),
    'replay01': dict(job_id='N01_WINDOW008_RESUME01_EVALUATION_REPLAY01',relative='runs/evaluation_jobs/replay01',
        manifest='WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json',planned_signal_forward_rows=85260,
        planned_observation_rows=255780,additional_unique_signal_rows=0),
    'diagnostics': dict(job_id='N01_WINDOW008_RESUME01_DIAGNOSTICS_PRIMARY',relative='runs/diagnostic_jobs/primary',
        manifest='WINDOW008_RESUME01_DIAGNOSTICS_SOURCE_MANIFEST.json',planned_signal_forward_rows=29952,
        planned_observation_rows=89856,additional_unique_signal_rows=28416),
}


def replace_json(path,value):
    temporary=path.with_name(path.name+'.evaluation-state.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    os.replace(temporary,path)


def record(phase,snapshot=None):
    paths=[ROOT/'PROGRAM.json',ROOT/'jobs.json',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json']
    before=BASE/'evaluation_state_before'
    if not before.exists():
        before.mkdir()
        for path in paths:shutil.copy2(path,before/path.name)
    now=datetime.now(timezone.utc).isoformat()
    cfg=read(ROOT/'window008_resume01_config.json');jobs=read(ROOT/'jobs.json')
    for kind,spec in SPECS.items():
        found=[row for row in jobs['jobs'] if row['job_id']==spec['job_id']]
        if not found:
            jobs['jobs'].append(dict(**spec,status='registered_pending_predecessor_or_dispatch',registered_UTC=now,
                host='100.82.231.102',run_id=RUN,remote_root=cfg['remote_root'],scientific_job=True,
                source_manifest_sha256=sha(ROOT/spec['manifest']),planned_G_updates=0,planned_D_updates=0,
                noise_updates=0,population_updates=0,C2ST_updates=0,scientific_status='response_match_not_established'))
    state='window008_resume01_evaluation_registered';active=False
    next_action='Execute the frozen primary evaluation, then identical replay and registered diagnostics sequentially.'
    if phase!='prepared':
        if snapshot is None:raise ValueError('An actual saved snapshot is required')
        snapshot=Path(snapshot).resolve();snap=read(snapshot);files=snap['files']
        spec=SPECS[phase];job=next(x for x in jobs['jobs'] if x['job_id']==spec['job_id'])
        native=files.get('supervisor/exit_receipt.json');complete=files.get('JOB_COMPLETE.json')
        failure=files.get('JOB_FAILED.json',files.get('artifacts/FAILED.json',files.get('LAUNCH_FAILURE.json')))
        active=bool(snap['owned_processes']) and native is None
        state=('failed_preserved' if failure or (native and (native['exit_code']!=0 or native.get('reason')))
            else 'complete_native0' if complete and native and native['exit_code']==0
            else 'running' if active else 'completion_or_identity_unresolved')
        job.update(status=state,last_verified_UTC=snap['UTC'],snapshot=snapshot.relative_to(ROOT).as_posix(),
            snapshot_sha256=sha(snapshot),native_exit_code=None if native is None else native['exit_code'],
            owned_processes=snap['owned_processes'],failure=failure)
        if complete:job['job_complete']=complete
        if native:job['native_exit_receipt']=native
        state='window008_resume01_'+phase+'_'+state
        next_action=('Review the preserved failure before any further job.' if failure or (native and native['exit_code']!=0)
            else 'Proceed to the next registered sequential phase or report/verified private archive.' if complete and native
            else 'Return at the scheduled completion estimate; do not repeatedly poll.')
    replace_json(ROOT/'jobs.json',jobs)
    for path in (paths[0],paths[2]):
        value=read(path)
        value.update(status=state,current_stage=state,active_scientific_jobs=int(active),next_action=next_action,
            updated_UTC=now,last_verified_UTC=now,scientific_status='response_match_not_established',
            window008_resume01_evaluation_phase=phase,window008_resume01_evaluation_optimizer_updates=0)
        replace_json(path,value)
    result=dict(UTC=now,status=state,active_scientific_jobs=int(active),phase=phase,optimizer_updates=0)
    destination=BASE/('STATE_'+phase+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    write_new(destination,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('prepared',*SPECS),required=True)
    p.add_argument('--snapshot');args=p.parse_args();print(json.dumps(record(args.phase,args.snapshot)))
