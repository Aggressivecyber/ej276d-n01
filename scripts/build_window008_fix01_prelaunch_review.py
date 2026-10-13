"""Bind completed independent reviews and native receipts for source freezing."""
import datetime as dt
import json
from window008_fix01_runtime import ROOT,read,sha,write_new,validate_inputs,validate_evidence,require

def main():
    validate_inputs();files={};reviews={}
    paths=['window008_failure_fix01/model_review/cpu_01/REVIEW.json',
           'receipts/window008_fix01_20261013/engine_driver_independent_01/REVIEW.json',
           'receipts/window008_fix01_20261013/engine_driver_independent_01/HELPER_REVIEW.json',
           'window008_failure_fix01/helper_review/final_review.json']
    for name in paths:
        value=read(ROOT/name);require(value['status'].startswith('passed'),'Review not passed: '+name)
        reviews[name]=dict(status=value['status'],sha256=sha(ROOT/name));files[name]=sha(ROOT/name)
        bindings=value.get('reviewed_sources',value.get('files_sha256',{}))
        for source,digest in bindings.items():
            require(sha(ROOT/source)==digest,'Reviewed source changed: '+source);files[source]=digest
        if 'model_source' in value:
            require(value['model_source']['sha256']==sha(ROOT/'src/n01/model008_fix01.py'),'Reviewed model changed')
            files['src/n01/model008_fix01.py']=value['model_source']['sha256']
    names=['receipts/window008_fix01_20261013/engineering_cuda_01/ENGINEERING_RECEIPT.json',
           'receipts/window008_fix01_20261013/engineering_supervisor_01/launch.json',
           'receipts/window008_fix01_20261013/engineering_supervisor_01/exit_receipt.json',
           'receipts/window008_fix01_20261013/ENGINEERING_NATIVE_COLLECTION.json',
           'receipts/window008_fix01_20261013/REGISTRATION_RECEIPT.json',
           'receipts/window008_fix01_20261013/recovery_cpu_03/ENGINEERING_RECEIPT.json',
           'receipts/window008_fix01_20261013/model_cpu_02/ENGINEERING_RECEIPT.json',
           'scripts/build_window008_fix01_prelaunch_review.py','scripts/register_window008_fix01.py',
           'scripts/execute_window008_fix01_engineering.py',
           'docs/WINDOW008_STABILITY_FIX01_REVIEW.md','docs/WINDOW008_RECOVERY_EXECUTION_DESIGN_01.md']
    files.update({name:sha(ROOT/name) for name in names})
    qa=read(ROOT/names[0]);require(qa['status']=='passed_native_fix01_engineering_and_historical_failure_inference','Native QA failed')
    require({'src/n01/model008_fix01.py','src/n01/response008_fix01.py','src/n01/train_window008_fix01.py'}<=set(files),'Core review binding incomplete')
    result=dict(schema='n01/window008-fix01-prelaunch-review/v1',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='passed_independent_prelaunch_review',reviews=reviews,files_sha256=files,
        native_engineering_G_updates=4,native_engineering_D_updates=6,native_actual_SOURCE_inference_rows=1024,
        prior_CPU_engineering_G_updates=16,prior_CPU_engineering_D_updates=18,prior_CPU_actual_SOURCE_inference_rows=2593,
        new_scientific_G_updates=0,new_scientific_D_updates=0,DEV_numeric_decodes=0,
        scientific_status='response_match_not_established',
        scope='Independent bounded numerical/metadata reviews and native engineering; physical acceptance remains unestablished.')
    write_new(ROOT/'receipts/window008_fix01_20261013/PRELAUNCH_REVIEW.json',result)
    validate_evidence()
    print(json.dumps(dict(status=result['status'],bound_files=len(files),original_68_files_unchanged=True)))

if __name__=='__main__':main()
