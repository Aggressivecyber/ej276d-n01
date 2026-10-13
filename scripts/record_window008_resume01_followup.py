"""Verify the single wakeup and record a measured remaining-stage estimate."""
from pathlib import Path
import datetime as dt
import json
import tomllib
from window008_resume01_runtime import ROOT,read,sha,write_new,require

def main():
    automation_path=Path('C:/Users/10629/.codex/automations/n01/automation.toml')
    automation=tomllib.loads(automation_path.read_text(encoding='utf8'));rec=ROOT/'receipts/window008_resume01_20261013'
    require(automation['id']=='n01' and automation['kind']=='heartbeat' and automation['status']=='ACTIVE','Scheduler identity/status differs')
    require(automation['rrule']=='FREQ=DAILY;COUNT=1;BYHOUR=3;BYMINUTE=0;BYSECOND=0','Single followup schedule differs')
    require(automation['target_thread_id']=='01a12bad-8839-7522-89a5-f028b107d7ff','Followup thread differs')
    require(automation['prompt'].strip()==(rec/'FOLLOWUP_PROMPT.md').read_text(encoding='utf8').strip(),'Saved prompt differs')
    native=read(rec/'START_NATIVE_VERIFICATION.json');require(native['status']=='correct_scientific_start_verified_once','Scientific startup not verified')
    require(read(rec/'FIRST_STEP_REPLAY_COMPARISON.json')['status']=='exact_shared_scientific_metrics_match','First-step science replay differs')
    start=read(rec/'START_RECEIPT.json');snapshot=read(ROOT/start['first_observed_snapshot']);status=snapshot['files']['science/status.json']
    qa=read(rec/'engineering_cuda_01/ENGINEERING_RECEIPT.json');timing={(x['arm'],x['stage']):x['seconds'] for x in qa['steps']}
    full=13000-status['new_execution_G_updates'];simple=3000*timing['simple','mean_response']+16000*timing['simple','joint_stochastic']
    low=full*status['update_seconds']+simple
    high=(full*max(status['update_seconds'],status['elapsed_seconds']/status['new_execution_G_updates'])+simple)*1.12
    stamp=dt.datetime.fromisoformat(snapshot['UTC']);receipt=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='same_thread_single_followup_saved_after_verified_scientific_continuation',automation_id='n01',automation_status='ACTIVE',
        scheduled_local_Asia_Singapore='2026-10-14T03:00:00+08:00',scheduled_UTC='2026-10-13T19:00:00+00:00',
        automation_config_sha256=sha(automation_path),prompt_sha256=sha(rec/'FOLLOWUP_PROMPT.md'),
        source_manifest_sha256=sha(ROOT/'WINDOW008_RESUME01_SOURCE_MANIFEST.json'),config_sha256=sha(ROOT/'window008_resume01_config.json'),
        first_observed_new_G_updates=status['new_execution_G_updates'],new_G_budget=32000,new_D_budget=87000,
        remaining_seconds_stage_estimate=[low,high],estimated_finish_UTC=[(stamp+dt.timedelta(seconds=x)).isoformat() for x in (low,high)],
        timing_scope='Actual first100 full scientific steps plus native artificial simple timings; upper estimate includes12percent overhead.',
        new_run_training_complete=False,evaluation_complete=False,scientific_status='response_match_not_established',
        subagent_model_preference='GPT-6.1-Sol Max')
    write_new(rec/'FOLLOWUP_RECEIPT.json',receipt)
    for path in (ROOT/'PROGRAM.json',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'):
        value=read(path);value.update(followup_automation_id='n01',followup_automation_status='ACTIVE',followup_scheduled_local='2026-10-14T03:00:00+08:00',
            followup_receipt='receipts/window008_resume01_20261013/FOLLOWUP_RECEIPT.json',
            next_action='Rest during the healthy resumed job; take one completion snapshot at03:00 Asia/Singapore on14October.',
            latest_subagent_model_preference='GPT-6.1-Sol Max')
        temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8');temp.replace(path)
    print(json.dumps(dict(status=receipt['status'],scheduled_local=receipt['scheduled_local_Asia_Singapore'],remaining_hours=[round(x/3600,2) for x in (low,high)])))

if __name__=='__main__':main()
