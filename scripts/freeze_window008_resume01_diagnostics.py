"""Bind diagnostic sources before primary evaluation first accesses DEV."""
from pathlib import Path
from datetime import datetime,timezone
import json
from window008_resume01_diagnostics_common import ROOT,EVAL_MANIFEST,POLICY,sha,write_new,evaluation_binding,require
import window008_resume01_evaluation_common as primary


def freeze(root=ROOT):
    root=Path(root).resolve()
    cfg,training,policy,main_manifest=primary.evaluation_binding(root)
    for folder in ('runs/evaluation_jobs/primary','runs/evaluation_jobs/replay01','runs/diagnostic_jobs/primary'):
        require(not (root/folder).exists(),'Numerical execution has already been attempted')
    files={name:digest for name,digest in main_manifest['files'].items() if 'diagnos' in name.lower()}
    files[primary.EVAL_MANIFEST]=sha(root/primary.EVAL_MANIFEST)
    value=dict(schema='n01/window008-resume01-diagnostics-source-manifest/v1',run_id=primary.RUN_ID,
        UTC=datetime.now(timezone.utc).isoformat(),before_DEV_numeric_decode=True,before_scientific_model_forward=True,
        policy_sha256=sha(root/POLICY),main_evaluation_source_manifest_sha256=sha(root/primary.EVAL_MANIFEST),
        files=files,optimizer_updates=0,scientific_status='response_match_not_established')
    write_new(root/EVAL_MANIFEST,value)
    evaluation_binding(root)
    return dict(status='diagnostic_sources_preregistered',manifest_sha256=sha(root/EVAL_MANIFEST),files=len(files),
        scientific_forwards=0,DEV_numeric_decoded=False)


if __name__=='__main__':print(json.dumps(freeze()))
