"""Frozen public source import regression; only tiny manufactured CDF values.

Fresh E/D fixture, no scientific arrays/checkpoints/NN/AD/training/SSH. The
original package and seven scripts stay byte-identical. Cold child interpreters
use only fixture src/scripts plus installed standard/external libraries.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3]
REQUIRED='scripts/stability011_metrics.py'
OLD_SHA='fd87e544673f63887e57c8efafeb0327bbe4db430b04c9697c71cd2e244c3f78'
SOURCE_NAMES=('scripts/run_wp1_013.py','scripts/wp1_evaluation.py','scripts/wp1_parameter_probe.py','scripts/wp1_ops.py',
    'scripts/wp1_input_information.py','scripts/wp1_boundary_saved.py','scripts/wp1_controller.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,text):
    if not value:raise AssertionError(text)

def project_dependencies(package):
    """Static closure includes lazy imports without importing model modules."""
    pending=list(SOURCE_NAMES);seen=set();missing=[];edges=[]
    def resolve(module):
        for p in ('scripts/'+module.replace('.','/')+'.py','src/'+module.replace('.','/')+'.py','src/'+module.replace('.','/')+'/__init__.py'):
            if (package/p).is_file() or (ROOT/p).is_file():return p
        return None
    while pending:
        relative=pending.pop()
        if relative in seen:continue
        seen.add(relative);p=package/relative
        if not p.is_file():missing.append(relative);continue
        tree=ast.parse(p.read_text('utf8'));parts=relative.split('/')
        module=('.'.join(parts[1:])[:-3] if relative.startswith('src/') else p.stem)
        parent=module.split('.')[:-1]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    prefix=parent[:len(parent)-node.level+1]
                    base='.'.join(prefix+([base] if base else []))
                names=[base]+[base+'.'+x.name for x in node.names if x.name!='*']
            else:continue
            for name in names:
                target=resolve(name)
                if target:
                    edges.append(dict(caller=relative,imported=name,file=target));pending.append(target)
    return dict(project_modules=sorted(seen),missing=sorted(set(missing)),edges=edges)

CHILD = r'''
import importlib,json,pathlib,sys,traceback
root=pathlib.Path(sys.argv[1]).resolve();repository=pathlib.Path(sys.argv[2]).resolve();mode=sys.argv[3]
# Match script execution: scripts at entry, src inserted after the stdlib gate.
# Exclude the full repository/fixture launcher, even if PYTHONPATH polluted it.
original=[p for p in sys.path if p and not pathlib.Path(p).resolve().is_relative_to(repository)]
sys.path[:]=[str(root/'src'),str(root/'scripts')]+original
if mode=='masked_full_repository':sys.path.append(str(repository/'scripts'))
assert 'torch' not in sys.modules and 'stability011_metrics' not in sys.modules
module=importlib.import_module('wp1_evaluation')
assert 'numpy' not in sys.modules and 'stability011_metrics' not in sys.modules
metadata=dict(mode=mode,sys_path=sys.path.copy(),evaluation_file=module.__file__,initial_top_import_succeeded=True,
    dependency_cached_before_lazy_call=False,scientific_data_decodes=0,scientific_CP_loads=0,scientific_NN_rows=0,
    scientific_AD_queries=0,scientific_optimizer_steps=0,SSH_calls=0)
try:
    # Unequal support, tied values, one missing diagnostic value; pure toys.
    result=module.raw_distance([0.,1.,1.,float('nan')],[0.,2.])
    assert abs(result['value']-2/3)<1e-15 and result['fake_rows']==4 and result['fake_defined_rows']==3
    cdf=importlib.import_module('stability011_metrics');metadata.update(result=result,dependency_file=cdf.__file__)
    if mode=='isolated_fixed':assert pathlib.Path(cdf.__file__).resolve()==root/'scripts/stability011_metrics.py'
    assert 'torch' not in sys.modules
    print(json.dumps(metadata,allow_nan=False));sys.exit(0)
except ModuleNotFoundError as error:
    metadata.update(error_type=type(error).__name__,error=str(error),missing_name=error.name,traceback=traceback.format_exc())
    print(json.dumps(metadata,allow_nan=False));sys.exit(17)
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve();require(out.drive.upper() in ('E:','D:') and not out.exists(),'Fresh E/D output required');out.mkdir(parents=True)
    original_hashes={n:sha(ROOT/n) for n in SOURCE_NAMES}
    package_archive=ROOT/'receipts/wp1_013_20261015/runtime_package_01/wp1_source.tar.gz'
    package=out/'isolated83';package.mkdir();checks=[]
    with tarfile.open(package_archive,'r:gz') as archive:
        for row in archive.getmembers():
            p=PurePosixPath(row.name);require(row.isfile() and not p.is_absolute() and '..' not in p.parts and '\\' not in row.name,'Unsafe source tar')
            target=package/Path(*p.parts);require(target.resolve().is_relative_to(package.resolve()) and not target.exists(),'Outside fixture/overwrite')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as dst:shutil.copyfileobj(archive.extractfile(row),dst)
    manifest=json.loads((package/'WP1_SOURCE_MANIFEST.json').read_text('utf8'))
    require(len(manifest['files'])==83 and REQUIRED not in manifest['files'] and not (package/REQUIRED).exists(),'Wrong original frozen package')
    for name,row in manifest['files'].items():require(sha(package/name)==row['sha256'],'Original package source differs')
    original_closure=project_dependencies(package)
    require(original_closure['missing']==[REQUIRED],'Other missing lazy local dependencies')
    (out/'ORIGINAL_LAZY_DEPENDENCIES.json').write_text(json.dumps(original_closure,indent=2)+'\n',encoding='utf8')
    checks.append(dict(name='original83_has_one_reachable_missing_local_dependency',passed=True))
    # An SHA-protected old-root copy exists; it must not contaminate sys.path.
    old=out/'protected_old011/scripts';old.mkdir(parents=True);shutil.copyfile(ROOT/REQUIRED,old/'stability011_metrics.py')
    require(sha(old/'stability011_metrics.py')==OLD_SHA,'Old011 helper bytes differ')
    child=out/'cold_entry.py';child.write_text(CHILD,encoding='utf8')
    def launch(root,mode,expected):
        env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
        result=subprocess.run([sys.executable,'-B','-I',str(child),str(root),str(ROOT),mode],cwd=root,env=env,capture_output=True,text=True,timeout=60)
        (out/(mode+'_stdout.json')).write_text(result.stdout,encoding='utf8');(out/(mode+'_stderr.txt')).write_text(result.stderr,encoding='utf8')
        require(result.returncode==expected,'Unexpected cold-child native code '+mode+': '+result.stderr)
        payload=json.loads(result.stdout);return payload
    missing=launch(package,'isolated_original',17)
    require(missing['missing_name']=='stability011_metrics' and 'wp1_evaluation.py' in missing['traceback'],'Did not reproduce formal lazy import')
    checks.append(dict(name='cold83_top_import_passes_then_exact_lazy_call_fails',passed=True,child_native_code=17))
    masked=launch(package,'masked_full_repository',0)
    require(Path(masked['dependency_file']).resolve()==ROOT/REQUIRED,'Local fixture did not mask native omission')
    checks.append(dict(name='full_repository_path_masks_the_missing_source',passed=True,child_native_code=0))
    candidate=out/'isolated84_candidate';shutil.copytree(package,candidate)
    for name in ('scripts/wp1_ops.py','scripts/wp1_controller.py'):shutil.copyfile(ROOT/name,candidate/name)
    newops=(candidate/'scripts/wp1_ops.py').read_text('utf8')
    shutil.copyfile(ROOT/REQUIRED,candidate/REQUIRED);require(sha(candidate/REQUIRED)==OLD_SHA,'New package helper differs')
    closure=project_dependencies(candidate);require(not closure['missing'],'Candidate still misses project imports')
    (out/'CANDIDATE_LAZY_DEPENDENCIES.json').write_text(json.dumps(closure,indent=2)+'\n',encoding='utf8')
    checks.append(dict(name='candidate84_closes_all_reachable_static_lazy_project_imports',passed=True))
    # Changing ops inventory is sufficient: existing controller unions NEW_FILES,
    # and existing gate requires NEW_FILES in config.source_files before science.
    tree=ast.parse(newops);inventory=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NEW_FILES' for t in n.targets))
    require(REQUIRED in inventory and not set(inventory)<=set(manifest['files']),'Required source gate would accept old83')
    require(set(inventory)<=set(manifest['files'])|{REQUIRED},'Candidate inventory needs unrelated changes')
    checks.append(dict(name='same_inventory_addition_drives_controller_freeze_and_gate_requirement',passed=True))
    fixed=launch(candidate,'isolated_fixed',0)
    require(abs(fixed['result']['value']-masked['result']['value'])==0,'Candidate changed CDF result')
    checks.append(dict(name='cold84_actual_CDF_call_and_unequalN_ties_undefined_values_match',passed=True,child_native_code=0))
    require({n:sha(ROOT/n) for n in SOURCE_NAMES}==original_hashes,'Frozen seven source bytes changed')
    checks.append(dict(name='original_final_seven_sources_and_old011_helper_unchanged',passed=True))
    receipt=dict(schema='n01/wp1-native-import-candidate-check/v1',status='PASS',artificial_only=True,passed_checks=len(checks),checks=checks,
        source_archive_sha256=sha(package_archive),original_frozen_source_count=83,candidate_source_count=84,
        dependency=dict(relative_path=REQUIRED,sha256=OLD_SHA,bytes=(ROOT/REQUIRED).stat().st_size),
        candidate_ops_sha256=sha(candidate/'scripts/wp1_ops.py'),original_seven_sha256=original_hashes,
        fixture_only_not_frozen_or_deployed=True,scientific_data_decodes=0,scientific_CP_loads=0,scientific_NN_rows=0,
        scientific_AD_queries=0,scientific_optimizer_steps=0,SSH_calls=0)
    (out/'ENGINEERING_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',passed_checks=len(checks),candidate_ops_sha256=receipt['candidate_ops_sha256'],out=str(out))))

if __name__=='__main__':main()
