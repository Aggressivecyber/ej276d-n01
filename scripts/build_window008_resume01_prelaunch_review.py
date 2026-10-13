"""Bind independent continuation review and target-platform native acceptance."""
import datetime as dt
import json
from window008_resume01_runtime import ROOT,read,sha,write_new,require,validate_engineering,validate_evidence

def main():
    validate_engineering();rec='receipts/window008_resume01_20261013/'
    independent=read(ROOT/(rec+'independent_01/INDEPENDENT_REVIEW.json'))
    require(independent['status'].startswith('passed'),'Final independent continuation review not passed')
    files=dict(independent['files_sha256'])
    for name,digest in files.items():require(sha(ROOT/name)==digest,'Independent reviewed file changed: '+name)
    names=[rec+'independent_01/INDEPENDENT_REVIEW.json',rec+'independent_01/ACTUAL_NATIVE_RECEIPT_REVIEW.json',
        rec+'engineering_cuda_01/ENGINEERING_RECEIPT.json',rec+'engineering_supervisor_01/launch.json',
        rec+'engineering_supervisor_01/exit_receipt.json',rec+'ENGINEERING_NATIVE_COLLECTION.json',
        rec+'engine_cpu_01/ARTIFICIAL_CPU_QA_SUMMARY.json',rec+'engine_cpu_01/NATIVE_CPU_PLATFORM_DIAGNOSTIC.json',
        rec+'engine_cpu_01/FORMAL_BATCH_FIXTURE_EXPORT_RECEIPT.json',rec+'engine_cpu_01/FINAL_ENGINE_SOURCE_RECEIPT.json',
        rec+'driver_cpu_03/ENGINEERING_RECEIPT.json',rec+'driver_cpu_03/NATIVE_EXIT.json',rec+'REGISTRATION_RECEIPT.json',
        'scripts/build_window008_resume01_prelaunch_review.py']
    native_review=read(ROOT/names[1]);require(native_review['status']=='passed','Independent native receipt review not passed')
    for name,digest in native_review['files_sha256'].items():require(sha(ROOT/name)==digest,'Native independently checked artifact changed: '+name);files[name]=digest
    driver=read(ROOT/(rec+'driver_cpu_03/ENGINEERING_RECEIPT.json'))
    require(driver['status']=='passed' and driver['passed']==driver['total']==111,'Final driver QA not passed')
    for name in names:files[name]=sha(ROOT/name)
    result=dict(schema='n01/window008-resume01-prelaunch-review/v1',status='passed_independent_prelaunch_review',
        UTC=dt.datetime.now(dt.timezone.utc).isoformat(),files_sha256=files,
        prior_CPU_artificial_G_updates=4,prior_CPU_artificial_D_updates=6,
        prior_native_artificial_G_updates=3,prior_native_artificial_D_updates=6,
        native_actual_checkpoint_import_passed=True,Windows_CPU_actual_import_failed_due_initial_D_arithmetic_bit_differences=True,
        Windows_failure_preserved_and_scientific_guard_unchanged=True,scientific_model_forwards=0,scientific_G_updates=0,scientific_D_updates=0,
        DEV_numeric_decodes=0,scientific_status='response_match_not_established')
    write_new(ROOT/(rec+'PRELAUNCH_REVIEW.json'),result);validate_evidence()
    print(json.dumps(dict(status=result['status'],bound_files=len(files))))

if __name__=='__main__':main()
