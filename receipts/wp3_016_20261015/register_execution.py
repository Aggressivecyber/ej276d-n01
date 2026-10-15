"""Register, bind and back up one reviewed016 batch; never launch scientific work."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CURRENT = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
CFG, MF = 'wp3_016_config.json', 'WP3_016_SOURCE_MANIFEST.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value, new=False):
    with path.open('x' if new else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('ready','commit-readback','backup-source'), required=True)
    args = parser.parse_args()
    freeze = OUT/'runtime_package_01'
    p, manifest, frozen = [read(path) for path in (ROOT/CFG,ROOT/MF,freeze/'FREEZE_RECEIPT.json')]
    assert p['status']=='frozen' and sha(ROOT/CFG)==frozen['config_sha256']
    assert sha(ROOT/MF)==frozen['source_manifest_sha256']
    for name,row in manifest['files'].items():
        path=ROOT/name
        assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],name
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    review_path=OUT/'independent_01/FROZEN_PACKAGE_REVIEW.json'
    if args.stage=='commit-readback':
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
        records={**manifest['files'], CFG:dict(sha256=sha(ROOT/CFG),bytes=(ROOT/CFG).stat().st_size),
            MF:dict(sha256=sha(ROOT/MF),bytes=(ROOT/MF).stat().st_size)}
        for name,row in records.items():
            blob=subprocess.check_output(['git','cat-file','blob',commit+':'+name],cwd=ROOT)
            assert hashlib.sha256(blob).hexdigest()==row['sha256'] and len(blob)==row['bytes'],name
        write(OUT/'SOURCE_COMMIT_READBACK.json',dict(schema='n01/wp3-016-source-commit-readback/v1',UTC=stamp,
            source_commit=commit,source_controls=len(records),all_raw_Git_blobs_match_frozen_bytes=True,
            configuration_sha256=sha(ROOT/CFG),source_manifest_sha256=sha(ROOT/MF),files=records,
            public_push_performed=False,scientific_operations=0),new=True)
        print(json.dumps(dict(status='all_source_control_blobs_verified',source_commit=commit,source_controls=len(records))))
        return
    if args.stage=='backup-source':
        source=read(OUT/'SOURCE_COMMIT_READBACK.json')
        destination=Path('D:/GAN_backups/N01/wp3_016_20261015/source_01')
        destination.mkdir(parents=True,exist_ok=False)
        files=[ROOT/name for name in manifest['files']]+[ROOT/CFG,ROOT/MF,freeze/'FREEZE_RECEIPT.json',
            OUT/'SOURCE_COMMIT_READBACK.json',OUT/'EXECUTION_READY.json',review_path,
            OUT/'independent_01/SOURCE_REVIEW.json',OUT/'CONTROL_PREPARATION_RECEIPT.json']
        assert sum(path.stat().st_size for path in files)<24*1024**2
        inventory={}
        for path in dict.fromkeys(files):
            assert path.suffix.lower() not in ('.pt','.npz','.tar','.gz')
            name=path.relative_to(ROOT).as_posix();target=destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            assert sha(path)==sha(target)
            inventory[name]=dict(sha256=sha(target),bytes=target.stat().st_size)
        write(destination/'MANIFEST.json',dict(UTC=stamp,source_commit=source['source_commit'],files=inventory),new=True)
        (destination/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(inventory.items())),encoding='utf-8')
        write(OUT/'D_SOURCE_BACKUP_RECEIPT.json',dict(UTC=stamp,status='all_copied_files_SHA256_verified',
            destination=str(destination),files=len(inventory),bytes=sum(v['bytes'] for v in inventory.values()),
            manifest_sha256=sha(destination/'MANIFEST.json'),no_raw_or_private_checkpoints=True),new=True)
        print(json.dumps(dict(status='small_D_source_backup_verified',files=len(inventory))))
        return
    review=read(review_path)
    assert review['passed'] and not review['blockers']
    paths=[ROOT/n for n in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')]+[CURRENT]
    master,program,jobs,current=[read(path) for path in (paths[0],paths[1],paths[2],CURRENT)]
    assert all(d['active_scientific_jobs']==0 for d in (master,program,jobs,current))
    assert program['new_training_updates']==175667
    assert not any(j['job_id']=='N01_WP3_016_20261015' for j in jobs['jobs'])
    snapshot=OUT/'metadata_before_ready';snapshot.mkdir(exist_ok=False)
    for path in paths:
        shutil.copy2(path,snapshot/path.name)
    refs=dict(configuration=CFG,configuration_sha256=sha(ROOT/CFG),source_manifest=MF,
        source_manifest_sha256=sha(ROOT/MF),source_files=len(manifest['files']),source_controls=len(manifest['files'])+2,
        protected_source_records=frozen['protected_source_records'],original015_107_controls_preserved=True,
        freeze_receipt=ref(freeze/'FREEZE_RECEIPT.json'),independent_review=ref(review_path),
        remote_root=p['remote_root'],protocol='docs/WP3_016_CONFIGURATION.md',roles='WP3_016_DATA_ROLES.json',
        budget_caps=p['budget_caps'],resource_limits=p['resource_limits'],run_resource_limits=p['run_resource_limits'],
        status='frozen_ready_not_started',new_training_updates=0,new_AD_queries=0,
        cumulative_training_before_run=175667,source_commit_pending_local_readback=True)
    for value in (master,program,current):
        if 'wp3_execution' in value and 'wp3_015_execution' not in value:
            value['wp3_015_execution']=copy.deepcopy(value['wp3_execution'])
        value.update(status='WP3_016_frozen_ready',current_stage='WP3_016_two_charge_parameter_trials',
            current_work_package='WP3',active_scientific_jobs=0,active_research=None,pending_run_id=p['run_id'],
            wp3_016_execution=refs,wp3_execution=refs,wp3_016_G_updates=0,wp3_016_AD_queries=0,
            wp3_015_independent_readback_complete=True,wp3_independent_readback_complete=False,
            wp3_total_completed_G_updates_lower_bound=8000,
            next_action='Deploy frozen016, verify first real update with current same-snapshot health, set one future n01 wake and rest.',
            latest_subagent_model_preference='GPT-6.1-Sol Max',updated_UTC=stamp,
            scientific_status='response_match_not_established')
    jobs['jobs'].append(dict(job_id='N01_WP3_016_20261015',run_id=p['run_id'],scientific_job=True,
        status='frozen_ready_not_started',registered_UTC=stamp,user_authority='USER_MASTER_PLAN_20261015.txt',
        source_freeze_pending=False,exact_budget_pending=False,planned_budgets=p['budget_caps'],
        candidate_trials=p['candidate_trial_order'],planned_short_candidate_runs=2,completed_short_candidate_runs=0,
        G_updates=0,new_AD_queries=0,automatic_restart=False,DEV_numeric_read=False,execution=refs,
        resource_limits=p['resource_limits'],scientific_status='response_match_not_established'))
    jobs['updated_UTC']=stamp
    wp3=next(w for w in master['work_packages'] if w['id']=='WP3')
    wp3['execution015']=copy.deepcopy(wp3['execution'])
    wp3.update(status='016_frozen_ready',execution=refs,execution016=refs)
    master['budgets']['short_candidate_runs'].update(registered_WP3_trials=6,completed_WP3_trials=4,
        remaining_unregistered_WP3_slots=0,registered016_trials=2)
    master['budgets']['WP3_016']=dict(budget_caps=p['budget_caps'],resource_limits=p['resource_limits'],
        run_limits=p['run_resource_limits'],original_cumulative_training=175667)
    text=paths[3].read_text(encoding='utf-8-sig');marker='以下保留此前阶段说明。'
    assert marker in text
    prefix='''# N01 当前执行：WP3/016 已冻结，待启动

016冻结A/005及对应015 M0形状头、关闭均值适配，只训练两个正值均值校正电荷方差参数。两seed各2000G，最多4000G/AD；精确预算、输入角色和源码已独审，启动以原生回执为准。当前无科学作业，累计175667，015闭合报告保留。后续全部使用GPT-6.1-Sol Max。

[016配置](wp3_016_config.json) · [016合同](docs/WP3_016_CONFIGURATION.md) · [015报告](reports/wp3_015_20261015/REPORT.md) · [015最终回执](receipts/wp3_015_20261015/FINAL_RESULT_RECEIPT.json)

'''
    paths[3].write_text(prefix+text[text.index(marker):],encoding='utf-8')
    write(paths[0],master)
    program['master_plan_sha256']=current['master_plan_sha256']=sha(paths[0])
    current['plan_sha256']=sha(paths[3])
    for path,value in ((paths[1],program),(paths[2],jobs),(CURRENT,current)):
        write(path,value)
    write(OUT/'EXECUTION_READY.json',dict(schema='n01/wp3-016-execution-ready/v1',UTC=stamp,refs=refs,
        cumulative_training_updates=175667,actual016_G_AD=0),new=True)
    print(json.dumps(dict(status='016_frozen_ready_not_started',source_files=refs['source_files'],
        source_controls=refs['source_controls'],cumulative_training_updates=175667)))


if __name__=='__main__':
    main()
