"""Save the scheduler receipt and stage-based estimate after verified startup."""
from pathlib import Path
import datetime as dt
import json
import tomllib
from window008_fix01_runtime import ROOT,read,sha,write_new,require

def main():
    automation_path=Path('C:/Users/10629/.codex/automations/n01/automation.toml')
    automation=tomllib.loads(automation_path.read_text(encoding='utf8'))
    rec=ROOT/'receipts/window008_fix01_20261013'
    require(automation['id']=='n01' and automation['kind']=='heartbeat' and automation['status']=='ACTIVE','Scheduler identity/status differs')
    require(automation['rrule']=='FREQ=DAILY;COUNT=1;BYHOUR=22;BYMINUTE=0;BYSECOND=0','Single followup time differs')
    require(automation['target_thread_id']=='01a12bad-8839-7522-89a5-f028b107d7ff','Followup thread differs')
    require(automation['prompt'].strip()==(rec/'FOLLOWUP_PROMPT.md').read_text(encoding='utf8').strip(),'Saved followup prompt differs')
    start=read(rec/'START_RECEIPT.json');native=read(rec/'START_NATIVE_VERIFICATION.json')
    require(native['status']=='correct_scientific_start_verified_once','Startup native verification missing')
    snapshot=read(ROOT/start['first_observed_snapshot']);status=snapshot['files']['science/status.json']
    qa=read(rec/'engineering_cuda_01/ENGINEERING_RECEIPT.json')
    timing={(row['arm'],row['stage']):row['seconds'] for row in qa['steps']}
    remaining_full=16000-status['new_execution_G_updates']
    simple=3000*timing['simple','mean_response']+16000*timing['simple','joint_stochastic']
    low=remaining_full*status['update_seconds']+simple
    high=remaining_full*max(status['update_seconds'],timing['full','joint_stochastic'])+simple
    stamp=dt.datetime.fromisoformat(snapshot['UTC'])
    receipt=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='same_thread_single_followup_saved_after_verified_scientific_start',
        automation_id='n01',kind='heartbeat',scheduled_local_Asia_Singapore='2026-10-13T22:00:00+08:00',
        scheduled_UTC='2026-10-13T14:00:00+00:00',automation_config_sha256=sha(automation_path),
        prompt_sha256=sha(rec/'FOLLOWUP_PROMPT.md'),startup_snapshot_UTC=snapshot['UTC'],
        first_observed_new_G_updates=status['new_execution_G_updates'],new_G_budget=35000,new_D_budget=96000,
        remaining_full_G_updates=remaining_full,remaining_simple_mean_G_updates=3000,remaining_simple_joint_and_low_G_updates=16000,
        remaining_seconds_stage_estimate=[low,high],estimated_finish_UTC=[(stamp+dt.timedelta(seconds=x)).isoformat() for x in (low,high)],
        timing_scope='Early native full scientific step plus artificial simple timings; resource and checkpoint overhead uncertain.',
        new_run_training_complete=False,endpoint_evaluation_complete=False,scientific_status='response_match_not_established')
    write_new(rec/'FOLLOWUP_RECEIPT.json',receipt)
    for path in (ROOT/'PROGRAM.json',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'):
        value=read(path);value.update(followup_automation_id='n01',followup_scheduled_local='2026-10-13T22:00:00+08:00',
            followup_receipt='receipts/window008_fix01_20261013/FOLLOWUP_RECEIPT.json',
            next_action='Rest during the healthy scientific job; take one scheduled completion snapshot at 22:00 Asia/Singapore.')
        temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8');temp.replace(path)
    print(json.dumps(dict(status=receipt['status'],scheduled_local=receipt['scheduled_local_Asia_Singapore'],remaining_hours=[round(x/3600,2) for x in (low,high)])))

if __name__=='__main__':main()
