"""Audit saved legal-TRAIN geometry and traceable archive headers, with no waveform decode."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'receipts/response_007_20261013/deadline01/quiet_evidence'
PY='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    origin=ROOT/'runs/noise_origin_audit_003_20261012/runs/science/origin'
    complete=json.loads((origin/'AUDIT_COMPLETE.json').read_text(encoding='utf8'))
    names=['ORIGIN_REPORT.json','EVENT_FEATURES.npz']
    for n in names:assert sha(origin/n)==complete['output_hashes'][n],n
    report=json.loads((origin/'ORIGIN_REPORT.json').read_text(encoding='utf8'))
    with np.load(origin/'EVENT_FEATURES.npz',allow_pickle=False) as z:
        metadata={n:z[n].copy() for n in z.files if n in ['original_index','artificial_group','metadata_tp_post','metadata_left_post','metadata_peak_index_q32','metadata_pad_left','metadata_pad_right']}
    assert len(metadata['original_index'])==21047
    tp=metadata['metadata_tp_post'];left=metadata['metadata_left_post'];peaks=metadata['metadata_peak_index_q32']
    assert np.array_equal(tp-left,np.full(21047,120))
    assert not metadata['metadata_pad_left'].any() and not metadata['metadata_pad_right'].any()
    keys,counts=np.unique(left,return_counts=True)
    hist={str(int(k)):int(v) for k,v in zip(keys,counts)}
    assert hist==report['geometry']['actual_left_histogram']
    # The shortest/longest prefix before half-rise is a geometric upper bound,
    # not a claim that the rising edge or its entire prefix is quiet.
    geometry=dict(legal_TRAIN_rows=len(tp),original_artificial_groups=int(len(np.unique(metadata['artificial_group']))),
        processed_trace_samples=500,processed_sample_interval_ns=1,aligned_trace_samples=480,
        half_rise_index_range=[int(tp.min()),int(tp.max())],peak_index_range=[int(peaks.min()),int(peaks.max())],
        aligned_half_rise_index=120,aligned_crop_left_histogram=hist,
        largest_potential_prefix_before_half_rise_samples=int(tp.max()),
        rows_with_480_samples_before_half_rise=int(np.sum(tp>=480)),
        post_peak_sample_count_range=[int((500-1-peaks).min()),int((500-1-peaks).max())],
        independently_validated_quiet_480_rows=0,
        quiet_480_status='not_established_in_the_traceable_archive_and_saved_legal_TRAIN_audit',
        validated_baseline_convention_samples=100,
        baseline_convention_is_not_proof_of_pure_electronics=True)
    remote='''from pathlib import Path
import json,zipfile,numpy as np,datetime as dt
p=Path('/home/rylan/psd_results/EJ276D_AmBe_neutron_waveforms.npz')
s=p.stat();headers=[]
with zipfile.ZipFile(p) as z:
 for info in z.infolist():
  with z.open(info) as f:
   version=np.lib.format.read_magic(f)
   if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
   elif version==(2,0): shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
   else: raise ValueError(version)
  headers.append(dict(name=info.filename,shape=list(shape),dtype=str(dtype),fortran=fortran,compressed_bytes=info.compress_size,uncompressed_bytes=info.file_size,CRC=hex(info.CRC)))
inventory=[dict(name=x.name,is_file=x.is_file(),bytes=x.stat().st_size if x.is_file() else None) for x in sorted(p.parent.iterdir())]
print(json.dumps(dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),archive=str(p),stat=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino),headers=headers,directory_inventory=inventory,waveform_numeric_decodes=0,other_members_numeric_decodes=0,header_only=True)))
'''
    command=' '.join(['env','PYTHONDONTWRITEBYTECODE=1',shlex.quote(PY),'-B','-c',shlex.quote(remote)])
    live=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ForwardX11=no','rylan@100.82.231.102',command],text=True,encoding='utf8'))
    saved=complete['source_io']['fstat_after']
    stat_unchanged=all(live['stat'][a]==saved[b] for a,b in [('size','st_size'),('mtime_ns','st_mtime_ns'),('ctime_ns','st_ctime_ns'),('inode','st_ino')])
    assert stat_unchanged,'Raw archive stat changed; old provenance needs a fresh bound audit'
    references=[origin/'AUDIT_COMPLETE.json',origin/'ORIGIN_REPORT.json',origin/'EVENT_FEATURES.npz',
        ROOT/'docs/NOISE_ORIGIN_SOURCE_CHAIN.md',ROOT/'docs/NOISE_ORIGIN_003_PROTOCOL.md',ROOT/'docs/NOISE_ORIGIN_METHOD.md',
        ROOT/'DATA_REFERENCES.json',
        ROOT.parent/'sim2real_two_papers_20261008_0305/work/P0_METADATA_AUDIT/LABEL_AND_CONDITION_GAPS.md',
        ROOT.parent/'sim2real_two_papers_20261008_0305/reviews/R68_SOURCE_PROVENANCE_AUDIT_001/REPORT.md']
    receipt=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='bounded_metadata_audit_complete',
        geometry=geometry,software_time_grid=report['time_grid'],live_archive_headers=live,
        raw_archive_stat_matches_prior_complete_audit=stat_unchanged,
        previous_full_archive_sha256=complete['source_io']['container_sha256'],full_archive_hash_recomputed_this_audit=False,
        prior_21047_reconstruction_bitwise_verified=complete['same_target_bitwise_passes'],
        source_files={str(p):sha(p) for p in references},implementation_sha256=sha(Path(__file__)),
        new_waveform_numeric_decodes=0,new_model_forward_calls=0,new_training_updates=0,DEV_decodes=0,
        independent_acquisitions_proven=1,physical_hardware_timing_and_calibration='unknown',
        disposition='Keep all three explicit extension hypotheses. Existing traces do not supply a defensible 480-point quiet target.',
        scope_limit='Inventory covers the known psd_results directory and existing bound project references, not every file on the server.',
        measurement_needed=['same recorded acquisition settings and verified units/sample timing',
            'independent 480-point-or-longer quiet windows on the processed 1 ns grid, including uninterrupted autocovariance support',
            'baseline/trigger selection and upstream processing logs; no tail relabeling or stitching of short windows',
            'independent acquisition identity for physical validation'])
    (OUT/'QUIET_EVIDENCE_AUDIT.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    text='''# 007 长静默证据核对

当前已绑定资料不能提供可辩护的480点同采集静默噪声真值，因此保留fast、slow、common_level三种显式延拓假设。

003保存的21047条合法TRAIN几何已重新核对：20点均值得到的整条记录只有500点（软件1 ns网格），半高上升沿位于120–140点，峰值位于127–145点。没有一条记录在半高前具有480点前缀；即使把上升沿前全部样点算作潜在前缀，上限也只有140点，这还不是纯电子学噪声的证明。峰后也仅有354–372点，且脉冲尾部不能重标为静默。

已有100点基线是一项处理约定。003的配对重建、不同窗口位置和协方差统计为该约定提供可复核描述，但不证明其物理纯净性，也不能识别未经观测的长程相关。500点总长度不等于500点静默窗口。

本次重新读取了已保存的TRAIN几何与源文件，并对最早NPZ做ZIP/NPY头检查及其所在目录文件清单；没有解码任何新波形、DEV或其他成员数值。当前原文件大小、mtime、ctime、inode与003完整审计一致；本次没有重新计算3GB原件全SHA，因此不将stat一致写成新全量哈希验证。

原文件的0.05 ns软件时间格经20点块均值成为1 ns。原始硬件采样、带宽、增益、温度、偏压、触发选择、上游处理及NPZ创建链仍未知；人工分组和文件名不构成独立采集。已有可确认的独立采集数仍为1。

后续约束长程噪声需要：相同且有记录的工况、可核对单位/采样链、连续不少于480 ns的独立静默窗口，以及触发与上游处理记录。不得用挑选最安静事件、拼接100点窗口或脉冲尾部来替代该证据。此次核对范围仅限已绑定来源和已知目录，未声称全服务器不存在其他资料。

完整路径、SHA、窗口直方图、当前目录清单和审计范围见QUIET_EVIDENCE_AUDIT.json。
'''
    (OUT/'QUIET_EVIDENCE.md').write_text(text,encoding='utf8')
    reopened=json.loads((OUT/'QUIET_EVIDENCE_AUDIT.json').read_text(encoding='utf8'))
    assert reopened['geometry']==geometry
    print(json.dumps(dict(status=receipt['status'],geometry=geometry,archive_stat_unchanged=stat_unchanged,
        headers=live['headers'],directory_entries=len(live['directory_inventory']),report=str(OUT/'QUIET_EVIDENCE.md')),ensure_ascii=False))

if __name__=='__main__':main()
