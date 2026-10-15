"""Artificial stdlib stdin bootstrap loader cases; no native calls."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import builtins
import difflib
import hashlib
import json
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'scripts/wp3_016_controller.py'
OLD014='/home/rylan/projects/ej276d-n01/wp2_014_20261015'
NEW016='/home/rylan/projects/ej276d-n01/wp3_016_20261015'
BASE_SHA='3480825e94e954a46c2c633e5df6ddef514e2b7def1332bb2e7688e78ae041a3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def load_stdin_fixture(new_root,old_root):
    """Execute unchanged controller bytes with __file__=<stdin>; route only
    the two otherwise-Linux roots into an isolated E-drive artificial tree.
    This import shim performs no filesystem or process mutation itself.
    """
    original_import=builtins.__import__;routed=[]
    def mapped_path(value):
        if str(value)==NEW016:routed.append('new016');return new_root
        if str(value)==OLD014:routed.append('old014');return old_root
        return Path(value)
    def imports(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='pathlib':return types.SimpleNamespace(Path=mapped_path,PurePosixPath=PurePosixPath)
        return original_import(name,globals,locals,fromlist,level)
    namespace=dict(__file__='<stdin>',__name__='artificial016_stdin_bootstrap',__builtins__=dict(vars(builtins),__import__=imports))
    exec(compile(SOURCE.read_bytes(),'<stdin>','exec'),namespace)
    assert namespace['SELF_ROOT']==new_root
    return namespace,routed


def main():
    out=Path(sys.argv[1]).resolve();assert out.drive.upper() in ('E:','D:') and not out.exists();out.mkdir(parents=True)
    started=time.monotonic();checks=[];source014=(ROOT/'scripts/wp2_controller.py').read_bytes()
    assert hashlib.sha256(source014).hexdigest()==BASE_SHA
    old=out/'existing014';(old/'scripts').mkdir(parents=True);(old/'scripts/wp2_controller.py').write_bytes(source014)
    absent=out/'not_created016';assert not absent.exists()
    namespace,routed=load_stdin_fixture(absent,old);operations=namespace['operations']()
    assert 'new016' in routed and 'old014' in routed and not absent.exists()
    assert operations.TRIALS==('seed0_Q1','seed1_Q1') and operations.CFG=='wp3_016_config.json'
    assert operations.identity_adaptations==dict(launch=1,start=1,child_CPU_binding=1)
    checks.append(dict(name='actual_stdin_controller_bytes_uncreated016_finds_exact_SHA014',passed=True,new_root_created=False))

    def refuse(name,new_root,old_root):
        namespace,_=load_stdin_fixture(new_root,old_root)
        try:namespace['operations']()
        except ValueError as error:assert str(error)=='016 exact reused014 operations unavailable'
        else:raise AssertionError(name+' accepted invalid loader')
        checks.append(dict(name=name,passed=True))
    refuse('both_candidate_sources_missing_refused',out/'missing016b',out/'missing014')
    corrupt=out/'corrupt014';(corrupt/'scripts').mkdir(parents=True);(corrupt/'scripts/wp2_controller.py').write_bytes(source014+b'\n# corrupt artificial fixture\n')
    refuse('fallback014_wrong_SHA_refused',out/'missing016c',corrupt)
    installed=out/'installed016';(installed/'scripts').mkdir(parents=True);(installed/'scripts/wp2_controller.py').write_bytes(source014)
    namespace,_=load_stdin_fixture(installed,out/'missing014b');assert namespace['operations']().RUN_ID=='wp3_016_20261015'
    checks.append(dict(name='installed016_exact_SHA_source_still_has_priority',passed=True))
    (installed/'scripts/wp2_controller.py').write_bytes(source014+b'\n# tampered installed source\n')
    refuse('present_but_wrong_SHA016_refused_even_with_valid_old014',installed,old)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    result=dict(schema='n01/wp3-016-stdin-bootstrap-engineering/v1',UTC=datetime.now(timezone.utc).isoformat(),passed=True,
        artificial_only=True,checks=checks,elapsed_seconds=time.monotonic()-started,source_sha256=sha(SOURCE),pinned014_SHA256=BASE_SHA,
        executed_controller_bytes_mode='Unmodified current source compiled with filename and __file__=<stdin>; isolated E-drive root-routing import shim only.',
        private_arrays_or_CP_reads=0,NN_AD_training_random_scientific=0,SSH_calls=0,native_processes_launched=0,
        original_new_remote_tree_created=False)
    with (out/'ENGINEERING_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,checks=len(checks),source_sha256=result['source_sha256'],receipt=str(out/'ENGINEERING_RECEIPT.json'))))


if __name__=='__main__':main()
