"""Freeze reviewed continuation sources before their scientific optimizer execution."""
import datetime as dt
import subprocess
from window008_resume01_runtime import ROOT,read,sha,write_new,runtime_files,validate_evidence,require

def main():
    cfg,_,_=validate_evidence();path=ROOT/cfg['source_manifest'];require(not path.exists(),'Preserve prior continuation freeze')
    review=read(ROOT/'receipts/window008_resume01_20261013/PRELAUNCH_REVIEW.json')
    names=set(runtime_files())|set(review['files_sha256'])|{'receipts/window008_resume01_20261013/PRELAUNCH_REVIEW.json'}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    prereg='receipts/window008_resume01_20261013/PREREGISTRATION.json'
    native=read(ROOT/'receipts/window008_resume01_20261013/engineering_cuda_01/ENGINEERING_RECEIPT.json')
    write_new(ROOT/prereg,dict(UTC=stamp,run_id=cfg['run_id'],source_commit=commit,config_sha256=sha(ROOT/'window008_resume01_config.json'),
        before_first_resume01_scientific_training_forward_and_optimizer_update=True,
        prior_native_engineering_G_updates=native['engineering_G_updates'],prior_native_engineering_D_updates=native['engineering_D_updates'],
        prior_native_scientific_model_forwards=0,new_scientific_G_updates=0,new_scientific_D_updates=0,
        native_checkpoint_import_is_metadata_only=True,DEV_numeric_decodes=0,
        inherited_effective_G_updates=25000,inherited_effective_D_updates=57000,
        historical_actual_G_updates_before_resume=25667,historical_actual_D_updates_before_resume=59001,
        new_G_budget=32000,new_D_budget=87000,scientific_status='response_match_not_established'))
    names.add(prereg)
    result=dict(schema='n01/window008-resume01-source/v1',run_id=cfg['run_id'],UTC=stamp,source_commit=commit,
        config_sha256=sha(ROOT/'window008_resume01_config.json'),base_manifest_sha256=cfg['legacy_source_manifest_sha256'],
        fix01_manifest_sha256=cfg['fix01_source_manifest_sha256'],before_scientific_model_forward=True,
        before_scientific_model_forward_scope='Only resume01 scientific training; native artificial engineering and metadata import already performed.',
        before_first_resume01_scientific_optimizer_update=True,DEV_waveforms_decoded=False,
        module_inventory=sorted(x for x in names if x.startswith('src/n01/') and x.endswith('.py')),
        files={name:sha(ROOT/name) for name in sorted(names)})
    write_new(path,result)
    print(__import__('json').dumps(dict(status='resume01_scientific_source_frozen',files=len(names),source_commit=commit,manifest_sha256=sha(path))))

if __name__=='__main__':main()
