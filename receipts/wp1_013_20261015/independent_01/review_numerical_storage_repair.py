"""Independent necessary storage-repair judgment; saved JSON and headers only."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

ROOT=Path('E:/GAN/research/N01')
OUT=ROOT/'receipts/wp1_013_20261015/independent_01/native_repeat_fix01'
FAILED=ROOT/'runs/wp1_013_20261015/failed_attempt_02'
REPORT=ROOT/'receipts/wp1_013_20261015/native_repeat_readback01/saved_01/SAVED_REPEAT_READBACK.json'
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def headers(p):
    result={}
    with zipfile.ZipFile(p) as z:
        for name in z.namelist():
            with z.open(name) as f:
                magic=f.read(8);assert magic[:6]==b'\x93NUMPY'
                size=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0]
                h=ast.literal_eval(f.read(size).decode('latin1' if magic[6]<3 else 'utf8').strip())
            result[name[:-4]]=dict(shape=list(h['shape']),dtype=h['descr'])
    return result
report=read(REPORT)
failure=read(FAILED/'runs/primary/artifacts/FAILED.json')
oldcfg=read(FAILED/'wp1_013_config.json')
assert report['failed_error']==failure['error']=='A repeat pure-signal bytes differ'
assert report['A']['parent_count']==6593 and report['A']['draws']==4
assert report['A']['positive_variance_parents']==1 and report['A']['positive_variance_cells']==283
assert report['A']['maximum_variance_location']==dict(source_parent_row=6592,sample=303)
assert all(r['parent_major_four_repeats_exact'] and r['no_parent_spans_chunk_boundary'] for r in report['lineage'])
assert all(r['rows']==6593*4 and r['complete_chunks']==206 and r['last_chunk_rows']==4 for r in report['lineage'])
bound=math.sqrt(6*report['A']['sample_variance_max_V2'])
assert abs(bound-report['A']['maximum_pairwise_repeat_difference_upper_bound_V'])<1e-23
assert bound<1e-7
assert abs(report['A']['centered_RMS_V']/report['theta']['centered_RMS_V']-report['A_centered_RMS_ratio_to_theta'])<1e-22
old_driver=ROOT/'receipts/wp1_013_20261015/independent_01/native_import_fix01/isolated_source84/scripts/run_wp1_013.py'
assert sha(old_driver)=='faa2eb81c60d4ab6ff6507513782450eb1ba08059d8643aaed504e63fa0a1680'
driver=old_driver.read_text('utf8')
assert 'float(base_error.max())<=1e-7' in driver
bank_headers={str(b):headers(FAILED/f'runs/primary/artifacts/BANK_{b}.npz') for b in (0,1)}
assert all(h['source_parent_rows']['shape']==[26372] for h in bank_headers.values())
full_headers={}
for arm in ('A','theta'):
    for h in ('fast','slow','common_level'):
        key=f'FULL_0_{arm}_{h}'
        doc=read(FAILED/f'runs/primary/evaluation/{key}.json')
        header=headers(FAILED/f'runs/primary/evaluation/{key}.npz')
        assert doc['population']['fake_rows']==26372 and doc['population']['real_rows']==21047
        assert header['root/repeated_parent/parent_rows']['shape']==[6593]
        assert header['root/repeated_parent/draw_grid']['shape']==[6593,4]
        assert header['root/source_input_trends/source_ids']['shape']==[6593]
        assert not any(r['shape']==[26372,256] for r in header.values())
        full_headers[key]=dict(fake_rows=26372,real_rows=21047,parent_count=6593,draws=4,
            raw_individual_hard_wave_saved=False,keys=sorted(header))
assert not (FAILED/'runs/primary/artifacts/REFERENCE_0.npz').exists()
assert not (FAILED/'runs/primary/artifacts/REFERENCE_1.npz').exists()
assert not (FAILED/'runs/primary/artifacts/SIGNAL_MOMENTS_1_A.npz').exists()
spent=failure['counters']
assert spent['old_A_rows']==spent['head_actual_parameter_forward_rows']==spent['noise005_rows']==52744
assert spent['hard_ADC_rows']==spent['all_path_gather_rows']==316464 and spent['checkpoint_loads']==3
assert failure['operations_entered']==failure['operations_completed']==3729 and not failure['inflight']
first=read(ROOT/'runs/wp1_013_20261015/failed_attempt_01/runs/primary/artifacts/FAILED.json')
prior={k:spent[k]+first['counters'][k] for k in spent}
aggregate={k:oldcfg['budget_caps'][k]+prior[k] for k in spent}
assert aggregate['old_A_rows']==aggregate['head_actual_parameter_forward_rows']==403992
assert aggregate['head_total_kernel_rows']==444952 and aggregate['head_linear_direction_map_rows']==40960
assert aggregate['noise005_rows']==158232 and aggregate['hard_ADC_rows']==2202768
assert aggregate['all_path_gather_rows']==3677328 and aggregate['checkpoint_loads']==9
record=dict(schema='n01/wp1-independent-numerical-storage-repair/v1',status='PASS_NECESSARY_REPAIR_PENDING_FINAL_SOURCE_REVIEW',passed=True,
    UTC=datetime.now(timezone.utc).isoformat(),run_id='wp1_013_20261015',saved_repeat_readback=pin(REPORT),
    failed_attempt2=pin(FAILED/'runs/primary/artifacts/FAILED.json'),
    quantitative_saved_evidence=dict(A_centered_RMS_V=report['A']['centered_RMS_V'],theta_centered_RMS_V=report['theta']['centered_RMS_V'],
        A_to_theta_centered_RMS_ratio=report['A_centered_RMS_ratio_to_theta'],nonzero_A_variance_parents=1,
        nonzero_A_variance_cells=283,last_parent_row=6592,last_chunk_rows=4,complete_chunk_rows=128,
        A_sample_variance_max_V2=report['A']['sample_variance_max_V2'],pairwise_repeat_difference_bound_V=bound,
        pairwise_bound_LSB=bound*16383,observed_exact_raw_max='unknown_not_saved',crossbank_actual_A_equal='unknown_not_saved'),
    independent_metadata_checks=dict(BANK_headers=bank_headers,FULL_0_headers_and_layout=full_headers,
        grouped_identity='np.repeat(arange6593,4), draw tile[0,1,2,3]; 128%4=0; 26372=206*128+4; only last group falls in the4-row chunk',
        waveform_payloads_decoded_by_this_review=0,identity_payloads_decoded_by_this_review=0,
        independent_identity_assertions_read_from_worker_receipt=True),
    mechanism_evidence=dict(baseline='model010._baseline_once uses explicit zero phys/local arrays and no_grad original A',
        deterministic='model008 deterministic sets physical_head,basis_head,envelope_head,local_filter toNone; no shape z feeds the baseline',
        normalization='ResidualFiLM uses GroupNorm per sample, not BatchNorm mixing parents',
        precision='learned convolution/linear features and heads use float32, physical normalization uses float64',
        supported_interpretation='Tiny numerical batch/layout variation is supported by the last4-row concentration and saved bounds; it is not evidence of physical A randomness.',
        unsupported='Specific CUDA kernel, hardware defect or exact crossbank equivalence cannot be identified from saved mean/variance.',
        np_array_equal='The old array_equal gate required exact numeric equality; its error label said bytes. It supplied no tolerance for these finite numerical differences.'),
    engineering_threshold=dict(existing_before_failure_V=1e-7,source='original scripts/run_wp1_013.py:414 baseline replay absolute max comparator',
        fitted_from_current_output=False,scientific_acceptance_T_changed=False,
        semantics='Numerical reproduction of the same frozen A across repeated/batch layouts, separate from hardware/response acceptance. Retain finite/exact prefix/canonical and frozen-state checks.'),
    necessary_patch=[
        'Use the already registered1e-7 absolute numerical reproduction threshold for within-parent repeats and across-bank A, compute/save actual residual magnitudes and reject any excess.',
        'Retain actual float64 A pre384 for every parent/draw/bank, never replace by first-draw broadcast or zero/averaged covariance; keep all original hard observations unchanged.',
        'Correct REFERENCE_ENCODING claims so they point to actual A per-row arrays; reconstruct complete covariance from actual saved draws, including tiny numerical variance labeled numerical rather than physiology.',
        'Explicit artificial1ULP fixtures must preserve original A rows and covariance, exact parent lineage and signedzero-safe numeric comparison; above1e-7 and crossbank failures must reject.',
        'Preserve both prior failed source/control/output packages and actual charged work. One newly registered fresh attempt only, no automatic fourth attempt.'
    ],
    reuse_boundary=dict(reusable='Bank0 six frozen full-population/outputQ/input-trend/repeat statistics, all-row scalar Q/tail/PSD/morphology/guards, two saved MC BANK identities/latents, source-information and saved012-boundary artifacts.',
        not_saved='REFERENCE0/1 actual per-draw hard wave, original A/shape signal draws, selected decoded/logits/noise caches, bank1 full evaluation and moments.',
        no_zero_NN_completion='Mean/variance and scalar descriptors cannot uniquely reconstruct missing raw waves, sorted projection/CDF data, actual nonlinear calibration or selected frozen-head feature/noise inputs. Completing the registered two-bank/30-state plan cannot be asserted as zero NN from these artifacts.',
        no_rule_changes='Do not substitute bank0 alone, fewer parents/banks/scales/hypotheses, conditional summaries for full wave, or artificial undefined directions merely because raw arrays were lost.'),
    prior_two_attempt_charged_work=prior,next_attempt_original_caps=oldcfg['budget_caps'],maximum_aggregate_caps=aggregate,
    resource='Existing caps remain; full actual A storage must fit the existing output cap. Native CUDA graph and peak resources are not validated by this metadata review.',
    reviewer_scope=dict(saved_JSON_and_NPZ_headers_only=True,original_data_decodes=0,raw_saved_wave_decodes=0,
        checkpoint_loads=0,Torch_imported=False,NN_rows=0,AD_queries=0,optimizer_created=0,training_updates=0,DEV_values_read=0,SSH_calls=0,source_edits=0),
    conclusion_zh='同意最小数值保存修复的必要性：末父4行批次A差异与浮点舍入解释相容，上界7.66nV低于既有100nV数值回放门；这不是已证明的具体CUDA/hardware原因，更不是A生理随机。以旧门保存真实逐draw A并保留微小协方差，不修改科学T/规则。现存bank0统计可复用，缺raw/decoded/noise不能零NN补全全013；两次完整参考生成成本必须保留，下一必要一次重生成最大总A/head403992、headTotal444952、noise158232、ADC2202768、gather3677328、CP9，AD/训练/DEVmodel仍0。')
OUT.mkdir(parents=True,exist_ok=True)
target=OUT/'NUMERICAL_STORAGE_REPAIR_REVIEW.json'
with target.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
assert read(target)['passed']
print(json.dumps(dict(status=record['status'],receipt=str(target),sha256=sha(target),NN_rows=0,AD_queries=0)))
