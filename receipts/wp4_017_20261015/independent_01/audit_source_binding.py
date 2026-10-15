"""Read-only source/JSON binding audit for the single requested SOURCE_REVIEW."""
from pathlib import Path
import ast
import copy
import hashlib
import json

ROOT=Path('E:/GAN/research/N01')
HERE=Path(__file__).resolve().parent
ENG=ROOT/'receipts/wp4_017_20261015/engineering_01'
OLD=ENG/'source_review_fix01/before'


def record(path):
    assert path.suffix.lower() in ('.py','.json','.md','.txt')
    b=path.read_bytes()
    return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def read(path):
    return json.loads(path.read_bytes())


def node(path,name):
    tree=ast.parse(path.read_bytes())
    if '.' in name:
        cls,name=name.split('.')
        tree=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    return copy.deepcopy(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name))


def main():
    ready=read(ENG/'ENGINEERING_READY.json');checks={};refs={}
    def check(name,value):
        assert value,name
        checks[name]=True
    def checkref(row):
        rel=row['relative_path'];p=(ROOT/rel).resolve()
        assert p.is_relative_to(ROOT)
        result=record(p)
        assert all(result[k]==row[k] for k in ('bytes','sha256')),(rel,result)
        refs[rel]=result
    check('canonical_READY_final_known_SHA',record(ENG/'ENGINEERING_READY.json')==dict(bytes=22813,
        sha256='eaa0bba9a7f6c4be9c721bf7dac36b3fe67b722a18e989527b58cd2d64934885'))
    for row in ready['sources'].values():checkref(row)
    check('all5_current_runtime_source_bytes_match_final_READY',len(ready['sources'])==5)
    for key in ('accepted_proposal','accepted_contract_review','configuration_document','data_roles',
                'configuration_draft','original124_protection','attempt_history'):
        checkref(ready[key])
    for row in ready['engineering_receipts'].values():checkref(row)
    check('all_reused_and_focused_receipt_SHA_bindings_match',True)
    before=read(ENG/'PROTECTED_BEFORE.json')['files']
    after=read(ENG/'source_review_fix01/PROTECTED_AFTER_FIX01.json')['files']
    check('original124_before_and_after_manifest_identity',len(before)==124 and before==after)
    actual={rel:record(ROOT/rel) for rel in before}
    check('original124_actual_public_source_control_bytes_unchanged',actual==before)
    previous=read(OLD/'receipts/wp4_017_20261015/engineering_01/ENGINEERING_READY.json')
    for rel,row in previous['sources'].items():
        check('preserved_prior_source_'+rel,record(OLD/rel)==dict(bytes=row['bytes'],sha256=row['sha256']))

    driver='scripts/run_wp4_017.py'
    stable={}
    for name in ('Engine017.step','Runtime017.signals','Runtime017.training_window',
                 'budget_caps017','prepare_numeric017','energy_vectors017'):
        stable[name]=ast.dump(node(ROOT/driver,name))==ast.dump(node(OLD/driver,name))
    check('heavy_training_observation_loss_counter_and_energy_AST_exact',all(stable.values()))
    unchanged_stats={}
    for name in ('partition017','groups_weights017','weighted017','support_cell017','scalar_banks017',
                 'coverage017','covariance_cell017','timing_cell017','energy_cell017','scalar_features017',
                 'pooled_wave017','waveform_cells017','coverage_cells017'):
        unchanged_stats[name]=ast.dump(node(ROOT/'src/n01/stats017.py',name))==ast.dump(node(OLD/'src/n01/stats017.py',name))
    check('reused_statistical_formulas_AST_exact',all(unchanged_stats.values()))
    unchanged_bytes={s:(ROOT/s).read_bytes()==(OLD/s).read_bytes() for s in
        ('scripts/wp4_017_controller.py','src/n01/model017.py','src/n01/saved_array_kernels017.py')}
    check('model_controller_mature_kernel_exact_reused_bytes',all(unchanged_bytes.values()))
    metadata_only={}
    for name,field in (('build_frozen_cache017','T2_fixed_B0_scope'),('evaluate_kernel017','T2_mechanism_scope'),
                       ('Engine017.checkpoint','initial_stream_states')):
        current=node(ROOT/driver,name);prior=node(OLD/driver,name)
        for item in ast.walk(current):
            if isinstance(item,ast.ImportFrom) and item.module=='n01.stats017':
                item.names=[a for a in item.names if a.name!='T2_fixed_basis_scope017']
            elif isinstance(item,ast.Call):
                item.keywords=[k for k in item.keywords if k.arg!=field]
        metadata_only[name]=ast.dump(current)==ast.dump(prior)
    check('cache_eval_CP_only_explicit_new_reporting_metadata',all(metadata_only.values()))
    # Validate previous focused evidence on its exact preserved intermediate
    # sources; only metadata/preflight and T2 reporting followed that fixture.
    intermediate=ENG/'source_review_fix01/before_preflight_metadata'
    focused=read(ENG/'source_review_fix01/artificial_delta_01/ENGINEERING_RECEIPT.json')
    for s in ('scripts/run_wp4_017.py','src/n01/stats017.py'):
        check('focused27_exact_intermediate_'+s,record(intermediate/s)==focused['sources'][s])
    check('final29_actual_source_hashes_equal_READY',read(HERE/'FINAL_SEMANTIC_DELTA_CHECKS.json')['sources']==
          {s:dict(bytes=row['bytes'],sha256=row['sha256']) for s,row in ready['sources'].items()})
    check('no_source_or_metadata_mutation_by_this_audit',True)
    result=dict(schema='n01/wp4-017-independent-source-binding/v1',passed=True,checks=checks,
        check_count=len(checks),bound_references=refs,protected_count=len(actual),
        protected_source_control_records=actual,unchanged_heavy_AST=stable,
        unchanged_statistical_AST=unchanged_stats,unchanged_bytes=unchanged_bytes,
        metadata_only_AST=metadata_only,original_CP_SOURCE_REAL_DEV_reads=0,
        NN_AD_G_RNG_GPU_operations=0,SSH_SCP_status_deploy_start_calls=0,
        separate_review_layer=False)
    out=HERE/'SOURCE_BINDING_CHECKS.json'
    assert not out.exists(),'Preserve old independent evidence'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,checks=len(checks),protected=len(actual),bound=len(refs),output=str(out),**record(out))))


if __name__=='__main__':
    main()
