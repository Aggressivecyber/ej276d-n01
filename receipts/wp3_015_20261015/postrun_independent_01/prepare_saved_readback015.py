"""Create015's path/binding/no-probe adaptation of the completed014 readback, without execution."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import difflib
import hashlib
import json
import sys

ROOT = Path('E:/GAN/research/N01')
OLD = ROOT/'receipts/wp2_014_20261015/postrun_independent_01'
HERE = Path(__file__).resolve().parent
CONFIG_SHA = '54c096ad1a5e112493798e5c6c8db481cd3c79b78747200af0bbd47bccba5394'
MANIFEST_SHA = 'f1255dbff0e47edb7ec20e76c7d90057201f9a56f1184b7896f0c8cb0a67825a'
INVENTORY_SHA = '9d253a57bb79d9f25b112b4f5e0009a239c21360766c89cef7357d040adebe42'
KERNEL_SHA = 'fe1366f35f36e97974f1a4037a36ad858c03dd8453957c1b45b65080f0fee3ac'
ORIGINAL = '/home/rylan/projects/ej276d-n01/wp3_015_20261015'
SIDE = '/home/rylan/projects/ej276d-n01/wp3_015_saved_readback_20261015'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def record(path): return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def write_text(name, value):
    with (HERE/name).open('x', encoding='utf-8', newline='\n') as stream: stream.write(value)


def write_json(name, value): write_text(name, json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def replace_once(text, old, new):
    require(text.count(old)==1, 'adaptation anchor is not unique: '+old[:100])
    return text.replace(old,new,1)


def node(tree, cls, method):
    if cls is None: return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==method)
    item=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in item.body if isinstance(n,ast.FunctionDef) and n.name==method)


def main():
    old_path=OLD/'readback_saved_arrays.py';old=old_path.read_text(encoding='utf-8')
    require(sha(old_path)=='de071016fc926f8bc8c5cc475c4e4e6a230047166d9f43343deca1554d625caf','completed014 worker changed')
    require(sha(OLD/'saved_array_kernels.py')==KERNEL_SHA,'mature kernel changed')
    inventory_path=ROOT/'receipts/wp3_015_20261015/collection_01/REMOTE_FULL_INVENTORY.json'
    require(sha(inventory_path)==INVENTORY_SHA,'collected015 inventory changed')
    inv=json.loads(inventory_path.read_bytes());raw=inv['omitted_remote_inventory']
    require(len(raw)==161 and sum(v['bytes'] for v in raw.values())==3064431039,'actual015 raw envelope differs')
    new=old.replace('Bounded014','Bounded015').replace('wp2','wp3').replace('frozen014','frozen015').replace('frozen95source','frozen105source')
    new=replace_once(new,"run_id='wp3_014_saved_readback'","run_id='wp3_015_saved_readback_20261015'")
    new=new.replace('wp3_014_config.json','wp3_015_config.json').replace('WP2_SOURCE_MANIFEST.json','WP3_SOURCE_MANIFEST.json')
    new=replace_once(new,"CONFIG_SHA = 'abb1de68d5ca312d1dbaf14dc058d7514429321b0813aa00c70ff37b20e92e32'",f"CONFIG_SHA = '{CONFIG_SHA}'")
    new=replace_once(new,"MANIFEST_SHA = '14db81d3a08a0edede97f544ed8ec4c63147138223c00fc38d833bbc278e2684'",f"MANIFEST_SHA = '{MANIFEST_SHA}'\nINVENTORY_SHA = '{INVENTORY_SHA}'")
    new=replace_once(new,"TRIALS = ('seed0_C0', 'seed0_C1', 'seed1_C0', 'seed1_C1')","TRIALS = ('seed0_M0', 'seed0_M1', 'seed1_M0', 'seed1_M1')")
    # Remove the014-only mathematical gradient function and probe implementation/calls entirely.
    start=new.index('def add_three_gradients(');end=new.index('def require(',start)
    new=new[:start]+new[end:]
    start=new.index('    def probe(');end=new.index('\n\ndef main():',start)
    new=new[:start]+new[end:]
    block="""            for step in (0, 2000):
                audit.op('probe_saved_vectors_and_rows', dict(probe_raw_npz_files_decoded=1,
                    probe_gradient_npz_files_decoded=1, probe_snapshots_verified=1),
                    lambda: audit.probe(trial, step), trial=trial, G_updates=step)
"""
    new=replace_once(new,block,'')
    new=replace_once(new,"require(len(audit.pairs) == 36 and len(audit.probes) == 8, 'complete finite readback scope')",
        "require(len(audit.pairs) == 36 and not audit.probes, 'complete015 finite readback without probes')")
    new=replace_once(new,"require(len(mf['files']) == 95, 'frozen105source')","require(len(mf['files']) == 105, 'frozen105source')")
    new=replace_once(new,"        inv = load(args.inventory)","        require(sha(args.inventory) == INVENTORY_SHA, 'registered015 full inventory SHA')\n        inv = load(args.inventory)")
    new=replace_once(new,"        require(len(raw) == 176 and all(name.endswith('.npz') for name in raw), 'exact176 rawNPZ scope')\n        require(sum(row['bytes'] for row in raw.values()) <= 6160*1024**2, 'finite raw-byte envelope')",
        """        expected_evaluation = set()
        for trial in TRIALS:
            for snapshot, arms in [('initial_evaluation', ('A', 'initial')), ('final_evaluation', ('final',))]:
                directory = 'runs/primary/artifacts/' + trial + '/' + snapshot + '/'
                expected_evaluation.update(directory + f'bank{b}_BASELINE100.npz' for b in (0, 1))
                expected_evaluation.update(directory + f'bank{b}_{arm}_{h}_{kind}.npz'
                    for b in (0, 1) for arm in arms for h in H for kind in ('HARD_ROWS', 'REPORT'))
        summary_only = 'runs/primary/artifacts/MEAN_EVIDENCE_REAL_SUMMARY.npz'
        require(len(expected_evaluation) == 160 and set(raw) == expected_evaluation | {summary_only},
                'exact015 raw160 evaluation plus one SHA-only mean summary scope')
        require(sum(row['bytes'] for row in raw.values()) == 3064431039 and
                sum(row['bytes'] for row in raw.values()) <= 3072*1024**2, 'finite actual015 raw-byte envelope')""")
    new=replace_once(new,"== dict(raw_files_SHA_checked=176, hard_rows_files_decoded=72, report_npz_files_decoded=72,\n                        baseline_npz_files_decoded=16, probe_raw_npz_files_decoded=8, probe_gradient_npz_files_decoded=8,\n                        bank_pairs_verified=36, probe_snapshots_verified=8, waveform_rows_decoded=72*N)",
        "== dict(raw_files_SHA_checked=161, hard_rows_files_decoded=72, report_npz_files_decoded=72,\n                        baseline_npz_files_decoded=16, probe_raw_npz_files_decoded=0, probe_gradient_npz_files_decoded=0,\n                        bank_pairs_verified=36, probe_snapshots_verified=0, waveform_rows_decoded=72*N)")
    new=replace_once(new,"            preADC384_and_fullcarrier_independent_readback=False,",
        """            preADC384_and_fullcarrier_independent_readback=False,
            mean_evidence_REAL_summary_SHA_only_no_decode=True,
            mean_evidence_summary_record=audit.touched[summary_only],
            probe_or_gradient_readback_applicable=False,
            inherited015_reference_mean_module_not_causal_identification=True,""")
    require('audit.probe(' not in new and 'def probe(' not in new and 'add_three_gradients' not in new,'014 probe survived adaptation')
    require('wp2_014' not in new and 'wp3_014' not in new and 'WP2_SOURCE_MANIFEST.json' not in new and
        'seed0_C0' not in new and 'exact176' not in new,'old runtime binding survived')
    old_tree,new_tree=ast.parse(old),ast.parse(new)
    unchanged=[]
    for cls, method in [(None,n) for n in ('decode_window','lineage','wave_moments')]+[('Audit',n) for n in
        ('equal','member','identities','load_baselines','one_bank','pair','op','journal','state','guard')]:
        require(ast.dump(node(old_tree,cls,method),include_attributes=False)==ast.dump(node(new_tree,cls,method),include_attributes=False),
                'statistical/numeric/durable implementation changed: '+str(cls)+'.'+method)
        unchanged.append((cls+'.' if cls else '')+method)
    write_text('readback_saved_arrays.py',new)
    with (HERE/'saved_array_kernels.py').open('xb') as stream:stream.write((OLD/'saved_array_kernels.py').read_bytes())
    write_text('WORKER_ADAPTATION.patch',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
        fromfile='completed014/readback_saved_arrays.py',tofile='015/readback_saved_arrays.py')))
    fixture=OLD/'artificial_cpu_01/ENGINEERING_RECEIPT.json'
    fixture_data=json.loads(fixture.read_bytes())
    require(fixture_data['status']=='PASS','mature014 artificial receipt not passed')
    native_old=ROOT/'runs/wp2_014_20261015/independent_readback_collected_01/artifacts/ARRAY_READBACK.json'
    require(json.loads(native_old.read_bytes())['passed'] is True,'completed014 raw readback not passed')
    actual_preparation=dict(stdlib_only=True,actual_saved_NPZ_reads=0,original_SOURCE_REAL_DEV_waveform_reads=0,
        checkpoint_tensor_loads=0,NN_rows=0,AD_queries=0,optimizer_or_sampler_created=0,training_updates=0,
        new_random_draws=0,SSH_calls=0,numerical_fixtures_rerun=0)
    scope=dict(schema='n01/wp3-readback-scope-note/v1',UTC=datetime.now(timezone.utc).isoformat(),
        independently_recomputed=['codes+doublebaseline+optionalIEEE-XOR inverse and exact code roundtrip',
            'parent/seed/trial/bank/draw lineage, per-row Q/front/T/PSD/peak/rise/FWHM/baseline and reconstructible flags',
            'native r4 parent mean/sample variance/MCSE reconstruction and all9x9 pairwise-finite scalar moments/counts',
            'new r8 from two actual4draw banks: raw centered factor, diagonal/full9x9 and MCSE with sample denominator7;4+4 identity'],
        SHA_only_not_decoded=['MEAN_EVIDENCE_REAL_SUMMARY.npz: file role/hash only; no independent REAL mean/working-cluster-SE array replay'],
        producer_distance_evidence='Complete wave/QPSD/noise and other empirical distances remain frozen producer evidence plus separate savedJSON decision algebra; new CDF/loss replays0.',
        not_reconstructed=['fullpreADC384 signal or carrier, original-Y second copy, everyG sampler/RNG state, CP tensors, original C1 median fit'],
        no015_probe=True,native_r8_report_exists=False,dense256x256_covariance_materialized=False,
        scientific_acceptance=False,scientific_status='response_match_not_established')
    write_json('READBACK_SCOPE_NOTE.json',scope)
    limits=dict(CPU_threads=2,process_RSS_MiB=1536,virtual_memory_backstop_MiB=3072,wall_seconds=1800,output_MiB=64,
        expected_output_MiB_upper_estimate=8,raw_file_SHA_passes=1,raw_decode_passes=1,
        fixed_input_compressed_envelope_MiB=3072,input_IO_upper_envelope_MiB=6144,
        maximum_simultaneous_bank_pairs=1,covariance_parent_block=64,dense256x256_covariance_matrices=0,
        automatic_retry=False,empty_or_failed_readback='preserve FAILED/status/journal and stop; no tolerance/scope expansion')
    scientific_budget=dict(CP_loads=0,original_SOURCE_REAL_TRAIN_waveform_reads=0,original_DEV_waveform_reads=0,
        model_forwards=0,AD_queries=0,optimizer_or_sampler_created=0,training_updates=0,random_library_draws=0,
        new_CDF_or_loss_replays=0,bootstrap_replicates=0,GPU_work=0)
    expected=dict(raw_files_SHA_checked=161,raw_bytes_SHA_read=3064431039,hard_rows_files_decoded=72,
        report_npz_files_decoded=72,baseline_npz_files_decoded=16,probe_raw_npz_files_decoded=0,
        probe_gradient_npz_files_decoded=0,waveform_rows_decoded=1898784,bank_pairs_verified=36,probe_snapshots_verified=0)
    plan=dict(schema='n01/wp3-independent-readback-design/v1',UTC=datetime.now(timezone.utc).isoformat(),
        status='prepared_local_only_not_authorized_remote_execution',run_id='wp3_015_saved_readback_20261015',
        assigned_reviewer=dict(model='GPT-6.1-Sol',effort='Max',evidence='parent assigned continuing Sol Max role; runtime effort/model fields not independently exposed'),
        original_root=ORIGINAL,sidecar_root=SIDE,original_files_read_only=True,
        frozen_configuration=record(ROOT/'wp3_015_config.json'),frozen_manifest=record(ROOT/'WP3_SOURCE_MANIFEST.json'),
        inventory=record(inventory_path),worker=record(HERE/'readback_saved_arrays.py'),kernel=record(HERE/'saved_array_kernels.py'),
        scope_note=record(HERE/'READBACK_SCOPE_NOTE.json'),
        expected_raw_files=dict(total_npz_SHA161=161,evaluation_NPZ_decode160=160,hard_rows_npz=72,report_npz=72,
            baseline100_npz=16,mean_evidence_summary_npz_SHA_only=1,probe_or_gradient_NPZ=0,bank_pairs=36,
            response_rows_per_hard_file=26372,total_hard_rows=1898784,SOURCE_parents=6593,per_bank_draws=4,merged_draws=8,
            BASELINE100_rows=16*26372,raw_compressed_bytes=3064431039),
        expected_complete_counters=expected,expected_operations=285,
        operation_derivation='161 SHA +16 baseline decode +72 bank statistics +36 two-bank completed pairs =285 durable entered/completed operations.',
        finite_resource_budget=limits,scientific_budget=scientific_budget,
        planned_checks=[
            'Verify exact015 cfg54c/MFf125/source105 plus bound300/161 inventory before NumPy import or numeric decode; single SHA pass only on161 authorized NPZ.',
            'Decode exactly160 evaluation NPZ, sequential two-bank pair, no CP/NN/raw TRAIN/DEV/new random; additional mean summary is only opaque SHA checked.',
            'Keep unchanged014 code for uint16/16383-.5-doublemean+IEEE-XOR decoding, exact code roundtrip, baseline100, nine row features, reconstructible flags and preserved carrier flags/counts.',
            'For each of36 pairs verify parent/draw/bank/ID lineage in both HARD and REPORT, no row drops; report population6593x4 and REAL21047.',
            'Recompute native4draw per-parent mean/variance/MCSE and9x9 pairwise-finite covariance with counts. Build8draw factor and verify4+4 merge using sample denominator7; no6593 dense256x256 matrices.',
            'r8 is a new statistic derived from saved rows, not replay of a nonexistent native015 combined8 report. Wave sample rank<=7, not theoretical preADC covariance rank; pairwise-missing scalar covariance has no automatic PSD/rank claim.',
            'Reuse fixed014 morphology/finite covariance kernel byte-exact and mature engineering; no repeated macro/tiny numeric tests for path/binding/probe removal.',
            'First proof only after first real complete2bank pair and durable completed count; no tiny actual pair, unknown in-flight never counted complete.',
            'Independently selectedM0 development reference from savedJSON: dual-reference ALL6 positiveT noninferiority, mean12wave,1e-12 tieM0 and actual improvement over commoninitial. Selection remains pending this bounded array QA.'],
        reused_algorithms=dict(unchanged_AST_methods=unchanged,kernel_sha256=KERNEL_SHA,
            engineering_receipt=record(fixture),actual_completed014_receipt=record(native_old),
            new_numeric_test_execution=False),
        limitations=scope,actual_preparation=actual_preparation,
        execution_authority='Root must register this exact plan/worker/kernel/inventory/controller and explicitly authorize one CPU saved-array task; preparation makes0SSH/deploy/start/status calls.',
        scientific_acceptance=False,scientific_status='response_match_not_established')
    write_json('READBACK_PLAN.json',plan)
    ready=dict(schema='n01/wp3-independent-readback-worker-ready/v1',status='ready_for_root_registration_not_execution',
        UTC=datetime.now(timezone.utc).isoformat(),worker=record(HERE/'readback_saved_arrays.py'),
        kernel=record(HERE/'saved_array_kernels.py'),plan=record(HERE/'READBACK_PLAN.json'),scope_note=record(HERE/'READBACK_SCOPE_NOTE.json'),
        inventory=record(inventory_path),source105_frozen_config_sha256=CONFIG_SHA,manifest_sha256=MANIFEST_SHA,
        expected_complete_counters=expected,expected_operations=285,
        actual_schema='n01/wp3-readback-state/v1',first_proof_schema='n01/wp3-readback-first-formal-pair/v1',
        result_schema='n01/wp3-independent-raw-saved-array-readback/v1',
        CLI=f'python -B readback_saved_arrays.py --run-root {ORIGINAL} --inventory <bound inventory JSON> --out {SIDE}/artifacts',
        result_files=['status.json','operations.jsonl','FIRST_PAIR_PROOF.json','ARRAY_READBACK.json','COMPLETE.json','FAILED.json'],
        source_before_import=True,artificial_fixture_reused_not_rerun=record(fixture),
        completed014_real_readback_reused=record(native_old),unchanged_AST_statistical_and_durable_methods=unchanged,
        FIRST='first fully completed real two-bank pair; same-snapshot native health and PID/boot/source/kernel/inputSHA required by controller',
        no_probe_or_gradient_files_in015=True,mean_summary_sha_only_not_decode=True,
        actual_preparation=actual_preparation,registered_execution_performed=False,
        source105_unchanged=True,scientific_acceptance=False)
    write_json('READBACK_WORKER_READY.json',ready)
    selection=json.loads((HERE/'DEVELOPMENT_SELECTION.json').read_bytes())
    rows=[]
    for seed in (0,1):
        for h in ('fast','slow','common_level'):
            cells={a:next(x for x in selection['cells'] if x['candidate']==a and x['seed']==seed and x['hypothesis']==h) for a in ('M0','M1')}
            rows.append(f"|{seed}|{h}|{100*cells['M0']['wave_relative_improvement_vs_A']:.12f}%|{100*cells['M1']['wave_relative_improvement_vs_A']:.12f}%|")
    summary=('015四次短试已正常结束：每次2000G，共8000G/8000AD，CP4；全部6593 SOURCE/21047 REAL TRAIN均覆盖。原A/005及M1形状头保护证据通过，154518次进入与完成一致，无失败或在途操作。\n\n'
        '按冻结双参考ALL6非劣规则，两臂均合格；12格final wave均值M0=0.15012505860962605，M1=0.15243191496746586，共同C1初始=0.15042006897712382。M0确实优于共同初始，选作下一阶段研究参考，均值模块关闭；这不授予权重或科学晋级。\n\n'
        '|seed|假设|M0相对A wave改善|M1相对A wave改善|\n|---|---|---:|---:|\n'+'\n'.join(rows)+'\n\n'
        '用途门M0通过6/6，M1为0/6；两臂绝对T均未全通过，科学状态仍为response_match_not_established。原014 C1的5/6历史结论保留。表中百分比仅显示精度，判定使用保存JSON完整数值，未进行近阈值补训。\n\n'
        '本地139选收文件已逐SHA核验；300份完整远端库存中161份NPZ留远端。当前仅完成JSON/日志独审，未解码数组；下一次已准备的CPU读回将核160份评价NPZ及额外1份只做SHA的均值门摘要。完整wave/QPSD/noise经验距离、C1 median拟合及原始TRAIN波形均不重放。\n')
    write_text('DEVELOPMENT_SUMMARY.md',summary)
    require('numpy' not in sys.modules and 'torch' not in sys.modules,'preparation imported numeric package')
    for name in ('READBACK_PLAN.json','readback_saved_arrays.py','saved_array_kernels.py','READBACK_WORKER_READY.json','DEVELOPMENT_SUMMARY.md'):
        print(json.dumps(record(HERE/name),ensure_ascii=False))


if __name__=='__main__':main()
