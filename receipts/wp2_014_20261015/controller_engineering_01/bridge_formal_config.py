"""Read-only stdlib bridge for the root's exact registered014 draft."""
from __future__ import annotations
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True,type=Path);parser.add_argument('--expected-sha256',required=True);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();repo=args.repo.resolve();target=args.out.resolve()
    assert target.drive.upper() in ('E:','D:') and not target.exists()
    target.mkdir(parents=True);temporary=target/'tmp';temporary.mkdir();os.environ.update(TEMP=str(temporary),TMP=str(temporary),PYTHONDONTWRITEBYTECODE='1')
    spec=importlib.util.spec_from_file_location('formal014_controller',repo/'scripts/wp2_controller.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    config=repo/c.CFG
    assert c.sha(config)==args.expected_sha256
    p=c.core().read_json(config)
    assert p['status']=='draft_registered_before_any_scientific_operation'
    c.validate_config(p,frozen=False,driver_source=(repo/c.DRIVER).read_bytes())
    assert len(p['inputs'])==8 and len(p['contracts'])==7
    roots=[PurePosixPath(r) for r in p['allowed_input_roots']]
    assert all(r.is_absolute() and '..' not in r.parts for r in roots)
    checked={str(config):args.expected_sha256};records={}
    for role,rows in (('input',p['inputs']),('contract',p['contracts'])):
        for name,row in rows.items():
            native=PurePosixPath(row['path'])
            assert native.is_absolute() and '..' not in native.parts and any(native.is_relative_to(r) for r in roots)
            local=c.checked_local(row,checked)
            assert local.drive.upper() in ('E:','D:')
            records[role+':'+name]=dict(native_path=row['path'],local_path=str(local),sha256=row['sha256'],bytes=row['bytes'])
    prior=c.core().read_json(repo/'WP1_SOURCE_MANIFEST.json')
    assert c.sha(repo/'WP1_SOURCE_MANIFEST.json')==c.PRIOR_MANIFEST_SHA and len(prior['files'])==84
    for name,row in prior['files'].items():assert c.named_record(repo/name)==row
    old_cfg=c.core().read_json(repo/'wp1_013_config.json')
    assert c.sha(repo/'wp1_013_config.json')==prior['config_sha256']
    planned_paths={name:repo/name for name in set(prior['files'])|set(c.OWNED)|{'WP1_SOURCE_MANIFEST.json'}}
    for role,rows in (('input',p['inputs']),('contract',p['contracts'])):
        for name,row in rows.items():
            native=PurePosixPath(row['path'])
            if native.is_relative_to(PurePosixPath(c.REMOTE)):
                relative=native.relative_to(PurePosixPath(c.REMOTE)).as_posix();c.core().safe_name(relative)
                assert not relative.endswith(('.npz','.pt','.pth','.bin'))
                local=Path(row['local_path'])
                assert relative not in planned_paths or c.named_record(planned_paths[relative])==c.named_record(local)
                planned_paths[relative]=local
    closure=c.project_import_closure(planned_paths,lambda n:planned_paths[n].read_bytes())
    protected=old_cfg['protection_files']
    for name,row in protected.items():
        assert any(PurePosixPath(row['path']).is_relative_to(r) for r in roots)
        c.checked_local(row,checked)
    for name in list(prior['files'])+['WP1_SOURCE_MANIFEST.json','wp1_013_config.json']:
        assert any(PurePosixPath(c.OLD013+'/'+name).is_relative_to(r) for r in roots)
    settings,budget=c.factory_contract((repo/c.DRIVER).read_bytes())
    assert p['budget_caps']==budget(p)
    assert p['budget_caps']['head_total_kernel_rows']==p['budget_caps']['head_actual_parameter_forward_rows']+p['budget_caps']['calibration_map_rows']==8653376
    assert p['contracts']['tolerance_table']['sha256']==c.T_SHA
    assert not {'numpy','torch','n01'}&set(sys.modules)
    for filename,digest in checked.items():assert c.sha(filename)==digest
    assert c.sha(config)==args.expected_sha256
    source_pins={n:c.sha(repo/n) for n in c.OWNED}
    receipt=dict(schema='n01/wp2-formal-draft-controller-bridge/v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='formal_draft_interfaces_and_hashes_pass',
        configuration_sha256=args.expected_sha256,configuration_status=p['status'],configuration_unchanged=True,controller_sha256=source_pins[c.CONTROLLER],owned_sources=source_pins,
        inputs_count=8,contracts_count=7,bound_file_checks=records,actual_factory_caps=p['budget_caps'],allowed_input_roots=p['allowed_input_roots'],
        old84_unchanged=True,old185_protection_hashes_verified=True,planned_protection_record_count=len(protected)+84+2,
        planned_source_file_count=len(planned_paths),planned_source_paths=sorted(planned_paths),lazy_import_closure=closure,
        frozen=False,deployed=False,SSH_calls=0,source_packages_created=0,scientific_operations=dict(array_decodes=0,checkpoint_deserializations=0,model_forwards=0,AD_queries=0,optimizer_created=0,optimizer_steps=0,training_updates=0,DEV_rows=0),
        numerical_packages_imported=False,notes='Byte SHA checks include bound checkpoint files; no checkpoint deserialization, scientific query, metadata mutation or freeze occurred.')
    c.core().write_new(target/'ENGINEERING_RECEIPT.json',receipt)
    print(json.dumps(dict(status=receipt['status'],configuration_sha256=args.expected_sha256,controller_sha256=receipt['controller_sha256'],input_count=8,contract_count=7,
        planned_source_file_count=len(planned_paths),receipt_sha256=c.sha(target/'ENGINEERING_RECEIPT.json'))))


if __name__=='__main__':main()
