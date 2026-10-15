"""Prepare only: stdlib source extraction and a fixed saved-array audit budget."""
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT / 'receipts/wp2_014_20261015/postrun_independent_01'


def pin(p):
    return dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


cfg_path, mf_path = ROOT / 'wp2_014_config.json', ROOT / 'WP2_SOURCE_MANIFEST.json'
cfg, mf = read(cfg_path), read(mf_path)
assert pin(cfg_path)['sha256'] == 'abb1de68d5ca312d1dbaf14dc058d7514429321b0813aa00c70ff37b20e92e32'
assert pin(mf_path)['sha256'] == '14db81d3a08a0edede97f544ed8ec4c63147138223c00fc38d833bbc278e2684'
assert len(mf['files']) == 95
source_files = ('scripts/run_wp2_014.py', 'scripts/run_wp1_013.py', 'scripts/wp1_evaluation.py',
                'src/n01/diagnose_mean_random009.py', 'src/n01/losses014.py', 'src/n01/observation008.py')
for rel in source_files:
    assert {k: pin(ROOT / rel)[k] for k in ('bytes', 'sha256')} == mf['files'][rel]

# Copy only mature, pure NumPy statistical helpers. Do NOT import model modules.
kernel_sources = [
    (ROOT / 'src/n01/diagnose_mean_random009.py', ('_crossing', 'morphology')),
    (ROOT / 'receipts/wp1_013_20261015/postrun_independent_01/read_reference_arrays.py', ('scalar_moments',)),
]
parts = ['"""Mature NumPy-only morphology and pairwise-finite covariance kernels. No model imports."""',
         'import numpy as np', 'def require(value, message):\n    if not value:\n        raise ValueError(message)']
provenance = []
for path, names in kernel_sources:
    text = path.read_text(encoding='utf8')
    tree = ast.parse(text)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in names:
        node = functions[name]
        assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node))
        parts.append('\n'.join(text.splitlines()[node.lineno - 1:node.end_lineno]))
        provenance.append(dict(source=pin(path), function=name, first_line=node.lineno, final_line=node.end_lineno))
kernel = OUT / 'saved_array_kernels.py'
with kernel.open('x', encoding='utf8', newline='\n') as f:
    f.write('\n\n'.join(parts) + '\n')
ast.parse(kernel.read_bytes())

plan = dict(
    schema='n01/wp2-independent-readback-design/v1', UTC=datetime.now(timezone.utc).isoformat(),
    status='prepared_local_only_not_authorized_remote_execution',
    assigned_reviewer=dict(model='GPT-6.1-Sol', effort='Max',
        evidence='existing parent dispatch role; current list_agents provides no runtime model/effort fields',
        runtime_model_and_effort_independently_exposed=False),
    frozen_configuration=pin(cfg_path), frozen_manifest=pin(mf_path),
    source_schema_files={rel: pin(ROOT / rel) for rel in source_files},
    mature_reference_readback=pin(ROOT / 'receipts/wp1_013_20261015/postrun_independent_01/read_reference_arrays.py'),
    pure_statistical_kernel=pin(kernel), kernel_provenance=provenance,
    expected_raw_files=dict(hard_rows_npz=72, report_npz=72, baseline100_npz=16,
        probe_raw_parent_npz=8, probe_gradient_vectors_npz=8, total_npz=176,
        bank_pairs=36, response_rows_per_hard_file=26372, SOURCE_parents=6593,
        per_bank_draws=4, merged_draws=8, source_population='same original TRAIN parents; no new independent sources'),
    finite_resource_budget=dict(CPU_threads=2, process_RSS_MiB=1536,
        virtual_memory_backstop_MiB=3072, wall_seconds=1800, output_MiB=64,
        expected_output_MiB_upper_estimate=8, raw_file_SHA_passes=1,
        raw_decode_passes=1, fixed_input_compressed_envelope_MiB=6160,
        input_IO_upper_envelope_MiB=12320, maximum_simultaneous_bank_pairs=1,
        covariance_parent_block=64, dense256x256_covariance_matrices=0,
        automatic_retry=False, empty_or_failed_readback='preserve partial receipt and stop, no scope or tolerance expansion'),
    scientific_budget=dict(CP_loads=0, original_TRAIN_waveform_reads=0, original_DEV_waveform_reads=0,
        model_forwards=0, AD_queries=0, optimizer_or_sampler_created=0, training_updates=0,
        random_library_draws=0, new_CDF_or_loss_replays=0, bootstrap_replicates=0, GPU_work=0),
    planned_checks=[
        'Before raw decode, verify frozen config/MF/six used schema sourceSHA and full-remote inventory binding. Hash only176 authorized NPZ plus relevant smallJSON/logs; no checkpoint decode.',
        'For each of36 fixed trial/snapshot/arm/hyp pairs, verify two banks:26372rows=6593original parents x4, exact parent/bank/draw/ID order, seed/trial address and no row loss or duplicates.',
        'Decode ADCuint16/16383-.5-exactdoublemean, then optional uint64IEEE XOR. Re-encode codes exactly; check saved actual baseline mean, baseline100 lineage and native writer roundtrip claim separately.',
        'Recompute all nine scalar rows from decoded window+actual baseline where available, morphology with copied frozen NumPy-only helper, and NaN/guard/censor/selected-peak flags. Cross-check all preserved flags and reportcounts; identify unreconstructible full-carrier/preADC flags honestly.',
        'Recompute every parent r4 mean, centered factor, variance diagonal, MCSE and full9x9 pairwise-finite scalar covariance/joint counts, compare actual saved REPORT members using013 established tolerances.',
        'Derive r8 from actual two-bank rows; compare direct centered-factor/9x9 computations with the4+4 merge formula, retain sample denominator7. Native014 has no saved combined8 report: mark these as new readback statistics derived from saved rows, not native replay.',
        'Do not materialize6593 dense256x256 matrices. Wave covariance is represented by the8x256 centered factor, whose rank is<=7. Pairwise-missing scalar covariance need not be PSD or obey a single complete-case rank7 claim.',
        'Check 8 probe RAW identities/flags and stored scalar-AD vectors, length1288, total=sum3, norms/cosines/empty strata; no new autograd. Compare saved G0/G2000 strata with explicit boundaries and changed masks.',
        'Locally audit JSON/logs: native PID/exit chain, fourG2000 trials, entered/completed/inflight, kernel and AD equalities, same calibrated-head SHA chain, explicit paired streams, periodic parent rows, complete TRAIN visit counts, G/update budgets and failure flags.',
        'Fit statistics: hash INTEGRAL_TRAIN_STATISTICS.json, fit21047/TRAIN/sixfeatures/floor formula and scale positivity/metadata use chain. Do not independently refit median or read original REAL TRAIN.',
        'Recompute frozen development selection from unrounded saved scalars: mean2bank perseed/hyp, ALL6 positive-definedT noninferiority; score mean12 finalwave; tie1e-12 C0; undefinedpending. Separately test ALL6 wave>=2%, absoluteT and response acceptance. Never round-to-pass or extend training.',
    ],
    limitations=[
        'Local142 selected files do not include176 rawNPZ. Remote inventory SHA verification is not array numerical readback; record numeric raw statuspending until separately registered bounded execution.',
        'No second originalY copy is saved. Native writer checks originalY bytes; independent code and scalar/moment roundtrip does not create such a copy.',
        'No raw preADC384 signals or full carrier are saved by014 evaluation. Signal-only covariance/canonicalzero or preADCrail/halftie/search-complete diagnostics cannot be independently reconstructed; preserve native guards and saved flags as bounded evidence.',
        'The8draw conditional response spread is finiteMC within a fixed TRAIN system, not true paired physical target or new source/group uncertainty; no bootstrap or external claim.',
        'Initialhead paired bytes are checked through saved state-hash/calibration/code chain only. No .pt tensor load in this bounded raw task. Sampling order can be checked on saved periodic rows and counters, not asserted independently reconstructed for everyG.',
        'Original C1 median scales are not independently refit; root explicitly chose JSON/SHA/fitcount/floor/use-chain verification only.',
    ],
    execution_authority='Root must register sourceSHA and this fixed resource scope, then explicitly authorize one rawCPU execution. This preparation calls no SSH/deploy/status.',
    actual_preparation=dict(stdlib_only=True, numeric_package_imports=0, scientific_array_decodes=0,
        CP_reads=0, NN_rows=0, AD_queries=0, training_updates=0, SSH_calls=0),
    scientific_status='response_match_not_established', scientific_acceptance=False,
)
assert not {'numpy', 'torch', 'n01'} & set(sys.modules)
target = OUT / 'READBACK_PLAN.json'
with target.open('x', encoding='utf8', newline='\n') as f:
    json.dump(plan, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
design = OUT / 'DESIGN_ZH.txt'
with design.open('x', encoding='utf8', newline='\n') as f:
    f.write('014独立保存数组读回：固定36个bank pair，72 HARD_ROWS+72REPORT+16baseline+8probeRAW+8gradient，共176NPZ。逐pair解码，64父分块；CPU2/RSS1536MiB/虚拟内存3072MiB后备/墙钟1800s/输出64MiB。\n')
    f.write('复用013已通过的解码、pairwise covariance与冻结NumPy形态核；不导入神经模块。r4逐父矩与REPORT核对，r8由保存4+4行独立派生并核merge公式，不展开dense256cov。\n')
    f.write('拟合统计只核JSON/SHA、fit21047、ADC积分floor和使用链；不重拟合median。不读CP、原TRAIN/DEV，不NN/AD/随机/CDF/bootstrap/GPU/训练。\n')
    f.write('先做本地已收小JSON/log核查，raw数值审核保持pending。根登记并授权后才可单次执行CPU读回；本准备没有SSH/状态轮询。\n')
print(json.dumps(dict(status=plan['status'], plan=pin(target), design=pin(design), kernel=pin(kernel)), ensure_ascii=False))
