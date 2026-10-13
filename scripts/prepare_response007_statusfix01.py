"""Preserve the failed first update, create a logging-only resume entrypoint."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1];ID='response_007_20261013';REC=ROOT/'receipts'/ID/'status_fix01'
FAILED=ROOT/'runs'/ID/'initial_attempt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def replace_once(text,before,after):
    assert text.count(before)==1,before[:80]
    return text.replace(before,after)
def main():
    REC.mkdir(parents=True,exist_ok=False)
    failure=read(FAILED/'science/FAILURE.json');native=read(FAILED/'supervisor/exit_receipt.json')
    assert failure['G_updates']==1 and failure['D_updates']==failure['C2ST_updates']==failure['noise_updates']==0
    assert failure['current_arm']=='deterministic' and native['exit_code']==1
    old=read(ROOT/'RESPONSE007_SOURCE_MANIFEST.json')
    for n,d in old['files'].items():assert sha(ROOT/n)==d,n
    folder=ROOT/'provenance/response007_statusfix01';folder.mkdir(exist_ok=False)
    copies={'step00001.pt':FAILED/'science/FAILURE_CHECKPOINT.pt',
        'step00001_training.jsonl':FAILED/'science/train/deterministic/training.jsonl',
        'FAILURE.json':FAILED/'science/FAILURE.json','native_exit.json':FAILED/'supervisor/exit_receipt.json'}
    for name,p in copies.items():shutil.copy2(p,folder/name);assert sha(p)==sha(folder/name)
    lines=(folder/'step00001_training.jsonl').read_text(encoding='utf8').splitlines()
    assert len(lines)==1 and json.loads(lines[0])['G_updates']==1
    source=(ROOT/'src/n01/train_response007.py').read_text(encoding='utf8')
    source=source.replace('import signal\n','import signal\nimport shutil\n')
    source=replace_once(source,"ID='response_007_20261013'", """ID='response_007_20261013'


def training_status(metrics,total_G,total_D,elapsed):
    result=dict(metrics)
    result.update(status='training_response007',pid=os.getpid(),run_id=ID,total_G_updates=total_G,total_D_updates=total_D,
        total_C2ST_updates=0,elapsed_seconds=elapsed,new_execution_G_updates=total_G-1,
        previous_completed_G_updates=1,attempt_id='status_fix01')
    return result


def restore_first_update(engine,checkpoint,config):
    value=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (value['arm']!='deterministic' or value['G_updates']!=1 or any(value['D_updates'].values())
            or value['stage']!='mean_response' or value['config']!=config):
        raise ValueError('Only the bound completed first update may be resumed')
    engine.restore(checkpoint)
    if state_digest(engine.model.state_dict())!=state_digest(value['model']):raise RuntimeError('Resume model state changed')
    if not all(torch.equal(engine.streams.state_dict()[k],v.cpu()) for k,v in value['streams'].items()):
        raise RuntimeError('Resume stochastic streams changed')
    return value
""")
    source=replace_once(source,"p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0')",
        "p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--resume-checkpoint',required=True)")
    source=replace_once(source,"cfg,manifest=bind(args.config,args.source_manifest,args.container,args.identities)","""cfg,manifest=bind(args.config,args.source_manifest,args.container,args.identities)
        amendment=json.loads((ROOT/'receipts'/ID/'status_fix01/PREREGISTRATION.json').read_text(encoding='utf8'))
        required={'src/n01/train_response007_statusfix01.py','scripts/launch_response007_statusfix01.sh',
            'docs/RESPONSE_007_STATUS_FIX01.md','receipts/'+ID+'/status_fix01/PREREGISTRATION.json'}
        if not required<=set(manifest['files']):raise ValueError('Missing amendment sources')
        if sha256(args.resume_checkpoint)!=amendment['resume_checkpoint_sha256']:
            raise ValueError('Resume checkpoint differs from the preserved first update')
        if manifest.get('previous_G_updates')!=1 or manifest.get('new_execution_G_budget')!=37999:
            raise ValueError('Resume update accounting changed')""")
    source=replace_once(source,"engine.set_stage('mean_response');engine.checkpoint(directory/'initial.pt')","""engine.set_stage('mean_response')
            if arm=='deterministic':
                restored=restore_first_update(engine,args.resume_checkpoint,cfg)
                journal=ROOT/'provenance/response007_statusfix01/step00001_training.jsonl'
                lines=journal.read_text(encoding='utf8').splitlines()
                if len(lines)!=1 or json.loads(lines[0])['G_updates']!=1:raise ValueError('First update journal differs')
                shutil.copy2(journal,directory/'training.jsonl')
                engine.checkpoint(directory/'resume_start.pt')
                write_json(out/'RESUME_VERIFICATION.json',dict(UTC=utc(),resumed_G_updates=engine.update,
                    resumed_D_updates=engine.d_updates,checkpoint_sha256=sha256(args.resume_checkpoint),
                    model_state_sha256=state_digest(engine.model.state_dict()),
                    common_initial_state_sha256=engine.common_state_digest,noise005_state_sha256=engine.noise_state_digest,
                    explicit_streams_verified=True,new_scientific_forward_for_restore=0,new_G_updates_for_restore=0))
                del restored
            else:engine.checkpoint(directory/'initial.pt')""")
    source=replace_once(source,"""status=dict(status='training_response007',pid=os.getpid(),run_id=ID,total_G_updates=total_G,total_D_updates=total_D,
                        total_C2ST_updates=0,noise_updates=0,elapsed_seconds=time.monotonic()-started,**metrics)""",
        "status=training_status(metrics,total_G,total_D,time.monotonic()-started)")
    source=replace_once(source,"run_id=ID,G_updates=38000,D_updates=96000,C2ST_updates=evaluation['C2ST_updates'],noise_updates=0,",
        "run_id=ID,G_updates=38000,new_execution_G_updates=37999,previous_G_updates=1,D_updates=96000,C2ST_updates=evaluation['C2ST_updates'],noise_updates=0,")
    source=replace_once(source,"D_updates=sum(sum(x['D_updates'].values()) for x in finished)+current_D,C2ST_updates=c2st_updates,",
        "D_updates=sum(sum(x['D_updates'].values()) for x in finished)+current_D,C2ST_updates=c2st_updates,previous_G_updates=1,new_execution_G_updates=max(0,sum(x['G_updates'] for x in finished)+current_G-1),")
    target=ROOT/'src/n01/train_response007_statusfix01.py';assert not target.exists();target.write_text(source,encoding='utf8',newline='\n')
    launcher=(ROOT/'scripts/launch_response007.sh').read_text(encoding='utf8')
    launcher=launcher.replace('root=/home/rylan/projects/ej276d-n01/response_007_20261013\n','root=/home/rylan/projects/ej276d-n01/response_007_20261013_statusfix01\n')
    launcher=launcher.replace('-m n01.train_response007 \\','-m n01.train_response007_statusfix01 \\')
    launcher=launcher.replace('RESPONSE007_SOURCE_MANIFEST.json','RESPONSE007_STATUSFIX01_SOURCE_MANIFEST.json')
    launcher=launcher.replace('--device cuda:0','--device cuda:0 --resume-checkpoint "$root/provenance/response007_statusfix01/step00001.pt"')
    (ROOT/'scripts/launch_response007_statusfix01.sh').write_text(launcher,encoding='utf8',newline='\n')
    starter=(ROOT/'scripts/start_response007_remote.py').read_text(encoding='utf8')
    starter=starter.replace("str(ROOT)=='/home/rylan/projects/ej276d-n01/response_007_20261013'","str(ROOT)=='/home/rylan/projects/ej276d-n01/response_007_20261013_statusfix01'")
    starter=starter.replace('RESPONSE007_SOURCE_MANIFEST.json','RESPONSE007_STATUSFIX01_SOURCE_MANIFEST.json').replace('scripts/launch_response007.sh','scripts/launch_response007_statusfix01.sh')
    (ROOT/'scripts/start_response007_statusfix01_remote.py').write_text(starter,encoding='utf8',newline='\n')
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    receipt=dict(UTC=now,run_id=ID,attempt_id='status_fix01',config_sha256=sha(ROOT/'response007_config.json'),
        protocol_sha256=sha(ROOT/'docs/RESPONSE_007_PROTOCOL.md'),amendment_sha256=sha(ROOT/'docs/RESPONSE_007_STATUS_FIX01.md'),
        original_source_manifest_sha256=sha(ROOT/'RESPONSE007_SOURCE_MANIFEST.json'),
        previous_G_updates=1,previous_D_updates=0,previous_C2ST_updates=0,noise_updates=0,
        new_execution_G_budget=37999,cumulative_G_budget=38000,D_budget=96000,C2ST_budget=4200,
        resume_checkpoint_sha256=sha(folder/'step00001.pt'),resume_checkpoint='provenance/response007_statusfix01/step00001.pt',
        first_journal_sha256=sha(folder/'step00001_training.jsonl'),failed_receipt_sha256=sha(folder/'FAILURE.json'),
        failed_native_receipt_sha256=sha(folder/'native_exit.json'),failed_context_sha256=sha(FAILED/'science/FAILURE_CONTEXT.pt'),
        remote_root='/home/rylan/projects/ej276d-n01/response_007_20261013_statusfix01',
        previous_scientific_data_decoded=True,before_this_attempt_scientific_forward=True,
        all_original_sources_preserved=True,configuration_or_scientific_hyperparameter_changes=False,
        repair_reason='duplicate status keyword after first completed update')
    write(REC/'PREREGISTRATION.json',receipt)
    print(json.dumps(receipt))
if __name__=='__main__':main()
