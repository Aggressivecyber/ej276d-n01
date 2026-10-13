"""Freeze the reviewed correction before its formal scientific training."""
import datetime as dt
import subprocess
from window008_fix01_runtime import ROOT,read,sha,write_new,runtime_files,validate_evidence,require


def main():
    cfg,base,legacy=validate_evidence()
    manifest_path=ROOT/cfg['source_manifest']
    require(not manifest_path.exists(),'Preserve prior fix01 freeze')
    names=set(runtime_files())
    review=read(ROOT/'receipts/window008_fix01_20261013/PRELAUNCH_REVIEW.json')
    names|=set(review['files_sha256'])
    names|={'receipts/window008_fix01_20261013/PRELAUNCH_REVIEW.json',
        'receipts/window008_fix01_20261013/engineering_cuda_01/ENGINEERING_RECEIPT.json',
        'receipts/window008_fix01_20261013/engineering_supervisor_01/launch.json',
        'receipts/window008_fix01_20261013/engineering_supervisor_01/exit_receipt.json'}
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    registration=dict(UTC=stamp,run_id=cfg['run_id'],source_commit=commit,config_sha256=sha(ROOT/'window008_fix01_config.json'),
        new_G_budget=35000,new_D_budget=96000,retained_deterministic_G=19000,reused_full_mean_G=3000,
        discarded_legacy_joint_G=269,discarded_legacy_joint_D=807,
        before_first_fix01_scientific_training_forward_and_optimizer_update=True,DEV_numeric_decodes=0,
        registration_scope='Formal scientific training only; prior engineering and historical inference remain disclosed.',
        prior_CPU_engineering_G_updates=16,prior_CPU_engineering_D_updates=18,
        prior_CUDA_engineering_G_updates=4,prior_CUDA_engineering_D_updates=6,
        prior_CPU_actual_SOURCE_inference_rows=2593,prior_CUDA_actual_SOURCE_inference_rows=1024,
        historical_source_diagnostics_are_recorded_separately=True,scientific_status='response_match_not_established')
    prereg='receipts/window008_fix01_20261013/PREREGISTRATION.json'
    write_new(ROOT/prereg,registration);names.add(prereg)
    result=dict(schema='n01/window008-fix01-source/v1',run_id=cfg['run_id'],UTC=stamp,source_commit=commit,
        config_sha256=registration['config_sha256'],base_manifest_sha256=cfg['legacy_source_manifest_sha256'],
        before_scientific_model_forward=True,historical_failure_inference_already_performed=True,
        before_first_fix01_scientific_optimizer_update=True,DEV_waveforms_decoded=False,
        before_scientific_model_forward_scope='Formal fix01 scientific training; engineering and historical failure inference already performed.',
        prior_engineering_accounting={key:value for key,value in registration.items() if key.startswith('prior_')},
        module_inventory=sorted(name for name in names if name.startswith('src/n01/') and name.endswith('.py')),
        files={name:sha(ROOT/name) for name in sorted(names)})
    write_new(manifest_path,result)
    print(__import__('json').dumps(dict(status='fix01_source_frozen',source_commit=commit,files=len(names),manifest_sha256=sha(manifest_path))))


if __name__=='__main__':main()
