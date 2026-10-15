"""Prepare a source-only launch fix after a zero-science bootstrap failure."""
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=HERE.parent


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)


def main():
    review_path=HERE/'INDEPENDENT_FIX_REVIEW.json';review=read(review_path)
    assert review['passed'] and not review['blockers']
    failed_path=OUT/'native_execution_01/NATIVE_START_SUMMARY.json';failed=read(failed_path)
    assert failed['status']=='DEPLOY_BOOTSTRAP_FAILED_BEFORE_INSTALL_OR_START'
    assert failed['scientific_launch_performed'] is False and failed['scientific_cost_from_this_failed_bootstrap']==0
    assert read(ROOT/'PROGRAM.json')['new_training_updates']==175667
    assert read(ROOT/'jobs.json')['active_scientific_jobs']==0
    for name,row in review['sources'].items():
        assert sha(ROOT/name)==row['sha256'] and (ROOT/name).stat().st_size==row['bytes']
    p=read(ROOT/'wp3_016_config.json');old=copy.deepcopy(p)
    assert p['status']=='frozen' and sha(ROOT/'wp3_016_config.json')==failed['pins']['config_sha256']
    before=HERE/'before_canonical_revision';before.mkdir(exist_ok=False)
    for name in ('wp3_016_config.json','WP3_016_SOURCE_MANIFEST.json'):
        shutil.copy2(ROOT/name,before/name)
    p.update(status='draft_launch_bootstrap_fix01',source_files={},protection_files={},launch_revision='fix01',
        previous_launch_configuration_sha256=failed['pins']['config_sha256'],
        launch_fix_reason='Restore pinned existing014 source fallback before the new016 directory is created.',
        previous_deployment_failure=ref(failed_path))
    spec=importlib.util.spec_from_file_location('root016_fix_factory',ROOT/'scripts/wp3_016_controller.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    settings,budget=module.factory_contract((ROOT/'scripts/run_wp3_016.py').read_bytes())
    assert budget(p)==old['budget_caps']==review['budget_caps']
    assert all(p[k]==old[k] for k in settings if k!='status')
    assert p['inputs']==old['inputs'] and p['contracts']==old['contracts']
    module.validate_config(p,frozen=False)
    with (ROOT/'wp3_016_config.json').open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(p,stream,ensure_ascii=False,indent=2);stream.write('\n')
    receipt=dict(schema='n01/wp3-016-launch-fix-controls/v1',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='source_only_launch_revision_ready_for_refreeze',config_draft_sha256=sha(ROOT/'wp3_016_config.json'),
        previous_failed_deploy=ref(failed_path),independent_fix_review=ref(review_path),
        scientific_settings_inputs_contracts_budgets_unchanged=True,
        candidate_trials_added=0,new_scientific_operations=0,cumulative_training_updates=175667)
    with (OUT/'CONTROL_PREPARATION_RECEIPT_FIX01.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
