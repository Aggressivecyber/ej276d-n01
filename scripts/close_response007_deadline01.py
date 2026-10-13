"""Close the saved bounded comparison; no model calls or new scientific data reads."""
from pathlib import Path
from html.parser import HTMLParser
import datetime as dt
import hashlib
import json
import re
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/response_007_20261013/deadline01'
SCI = RUN / 'runs/science'
REC = ROOT / 'receipts/response_007_20261013/deadline01'
REPORT = ROOT / 'reports/response_007_20261013/deadline01'
POINTER = ROOT.parents[1] / 'CURRENT_SIM2REAL_PROGRAM.json'
POPPLER = Path('C:/Users/10629/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')
PDF_PYTHON = Path('C:/Users/10629/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
ARMS = ('deterministic', 'full')
HYPOTHESES = ('fast', 'slow', 'common_level')
COMPONENTS = ('wave', 'QPSD', 'noise')


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and 'href' in attrs:
            self.hrefs.append(attrs['href'])
        if tag == 'img' and 'src' in attrs:
            self.images.append(attrs['src'])


def main():
    assert not (REC / 'FINAL_RECEIPT.json').exists(), 'Preserve existing closeout'
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    complete = read(SCI / 'COMPLETE.json')
    summary = read(SCI / 'TRAIN_diagnostics/SUMMARY.json')
    verify = read(REC / 'ARTIFACT_VERIFICATION.json')
    replay_path = RUN / 'runs/verification_deadline01/REPLAY_VERIFICATION.json'
    replay = read(replay_path)
    native = read(RUN / 'runs/supervisor/exit_receipt.json')
    replay_native = read(RUN / 'runs/verification_supervisor/exit_receipt.json')
    assert complete['G_updates'] == 8000 and complete['new_execution_G_updates'] == 7999
    assert complete['D_updates'] == 6000 and complete['C2ST_updates'] == complete['noise_updates'] == 0
    assert all(x['G_updates'] == 4000 for x in complete['arms'])
    assert native['exit_code'] == replay_native['exit_code'] == 0
    assert verify['status'] == replay['status'] == 'passed'
    assert not complete['original_19000_per_arm_endpoint_reached']
    assert replay['new_training_updates'] == replay['new_unique_random_draws'] == replay['DEV_decodes'] == 0
    for relative, digest in complete['files'].items():
        assert sha(SCI / relative) == digest, relative
    manifest = ROOT / 'RESPONSE007_DEADLINE01_SOURCE_MANIFEST.json'
    assert sha(manifest) == complete['source_manifest_sha256']
    for relative, digest in read(manifest)['files'].items():
        assert sha(ROOT / relative) == digest, relative
    build = read(REPORT / 'BUILD_RECEIPT.json')
    for relative, digest in build['files'].items():
        assert sha(REPORT / relative) == digest, relative

    pdf = REPORT / 'N01_RESPONSE007_STAGE_REPORT.pdf'
    info = subprocess.check_output([str(POPPLER / 'pdfinfo.exe'), str(pdf)], encoding='utf8', errors='replace')
    assert re.search(r'Pages:\s+3\b', info)
    pdf_read_code = 'from pypdf import PdfReader; import json,sys; print(json.dumps([len(p.extract_text()) for p in PdfReader(sys.argv[1]).pages]))'
    page_characters = json.loads(subprocess.check_output([str(PDF_PYTHON), '-B', '-c', pdf_read_code, str(pdf)], encoding='utf8'))
    assert len(page_characters) == 3 and min(page_characters) > 300
    parser = Links()
    parser.feed((REPORT / 'N01_RESPONSE007_STAGE_REPORT.html').read_text(encoding='utf8'))
    for relative in parser.hrefs + parser.images:
        assert (REPORT / relative).resolve().is_file(), relative
    renders = []
    for n in (1, 2, 3):
        p = RUN / f'visual_QA/final-page-{n}.png'
        header = p.read_bytes()[:24]
        assert header[:8] == b'\x89PNG\r\n\x1a\n'
        width, height = struct.unpack('>II', header[16:24])
        assert width > 1000 and height > 800
        renders.append(dict(page=n, path=p.relative_to(ROOT).as_posix(), sha256=sha(p), width=width, height=height))
    visual = dict(UTC=now, status='passed', reviewer='primary_agent_visual_self_review',
        PDF_path=pdf.relative_to(ROOT).as_posix(), PDF_sha256=sha(pdf), pages=3,
        renderer='Poppler pdftoppm at 105 dpi', rendered_pages=renders,
        all_three_rendered_pages_visually_inspected=True,
        observations=['Chinese labels are legible', 'No page clipping or overlapping legends',
                      'Three noise hypotheses, REAL-REAL reference, TRAIN-only and incomplete-endpoint limits visible'],
        reopened_pdf_text_characters=page_characters,
        HTML_path=(REPORT / 'N01_RESPONSE007_STAGE_REPORT.html').relative_to(ROOT).as_posix(),
        HTML_sha256=sha(REPORT / 'N01_RESPONSE007_STAGE_REPORT.html'),
        HTML_href_targets_verified=len(parser.hrefs), HTML_image_targets_verified=len(parser.images),
        supersedes_build_visual_QA_status='pending_rendered_inspection',
        scientific_data_changed=False, new_model_calls=0, new_training_updates=0)
    write(REC / 'VISUAL_QA.json', visual)

    latest_live = sorted((RUN / 'live').glob('*.json'))[-1]
    live = read(latest_live)
    assert live['owned_processes'] == []
    assert not live['GPU_compute_processes']
    assert (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(live['UTC'])).total_seconds() < 1800
    postflight = dict(UTC=live['UTC'], status='no_owned_science_process_and_GPU_compute_list_empty',
        snapshot=latest_live.relative_to(ROOT).as_posix(), snapshot_sha256=sha(latest_live),
        GPU=live['GPU'], GPU_compute_processes=live['GPU_compute_processes'], owned_processes=live['owned_processes'],
        science_native_exit_code=native['exit_code'], replay_native_exit_code=replay_native['exit_code'])
    write(REC / 'POSTFLIGHT_RECEIPT.json', postflight)
    native_copies = {}
    for source, label in [(RUN / 'runs/supervisor/exit_receipt.json', 'science_exit.json'),
                          (RUN / 'runs/verification_supervisor/exit_receipt.json', 'replay_exit.json'),
                          (replay_path, 'REPLAY_VERIFICATION.json')]:
        target = REC / 'native' / label
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        native_copies[label] = dict(original=source.relative_to(ROOT).as_posix(),
                                    retained_copy=target.relative_to(ROOT).as_posix(), sha256=sha(target))

    dist = summary['matched_distances']
    components = {a: {h: dist[a][h]['components'] for h in HYPOTHESES} for a in ARMS}
    worst = {a: {k: max(components[a][h][k] for h in HYPOTHESES) for k in COMPONENTS} for a in ARMS}
    assert worst == summary['worst_hypothesis_distances']
    improvement = {k: 1 - worst['full'][k] / worst['deterministic'][k] for k in COMPONENTS}
    real_real = dist['REAL_REAL']['components']
    control_ratios = {k: worst['full'][k] / real_real[k] for k in COMPONENTS}
    sensitivity = summary['sensitivity']
    full_index = sensitivity['arms'].index('full')
    lowq_index = sensitivity['statistic_names'].index('fixed_bin_0_PSD_SD_ratio')
    next_action = ('先核实后续Astra/Max/Fast运行设置；再登记长窗噪声约束与响应差异的最小可判别诊断。'
                   '当前不追加训练，原007完整末点尚未执行。')
    final = dict(UTC=now, status='bounded_stage_comparison_complete_reviewed',
        scientific_status='response_match_not_established',
        reviewer='primary_agent_self_review_no_subagents_no_second_party_review',
        endpoint=dict(per_arm_G=4000, mean_stage_G=3000, joint_stage_G=1000,
            cumulative_G=8000, previously_completed_G=1, new_execution_G=7999, D=6000, noise=0,
            C2ST=0, population_weight_updates=0, original_endpoint_per_arm_G=19000,
            original_endpoint_reached=False, remaining_original_G=30000, remaining_original_D=90000),
        update_accounting='Original 77000 + noise repairs 5000 + 5000 + 5000 + response007 cumulative 8000 = 100000 model updates; training D and evaluation C2ST counted separately.',
        comparison=dict(split='TRAIN', equal_arm_endpoint_G=4000, matched_rows_per_side=2048,
            hypotheses=list(HYPOTHESES), all_hypothesis_components=components,
            worst_hypothesis_components=worst, relative_reduction_full_vs_deterministic=improvement,
            reduction_formula='1 - max_h(distance_full[h]) / max_h(distance_deterministic[h])',
            reduction_comparator='same 4000-G arm endpoints, identical fixed matched selections and projection context',
            reduction_not_relative_to=['REAL_REAL', 'integration006', 'physical equivalence'],
            worst_hypothesis_can_differ_between_arms=True,
            REAL_REAL_components=real_real, full_worst_to_REAL_REAL_ratio=control_ratios,
            scientific_pass_threshold=None),
        low_signal_Q_sensitivity=dict(full_PSD_SD_max_over_min=sensitivity['point'][full_index][lowq_index],
            parent_bootstrap_95_interval=sensitivity['percentile_95_interval'][full_index][lowq_index],
            bootstrap_draws=1000, different_arm_bin_parent_populations=True,
            use='within-arm noise-hypothesis sensitivity, not arm superiority'),
        data_scope=dict(SOURCE_TRAIN_parents=6593, generated_draws_per_parent=4, REAL_TRAIN_rows=21047,
            all_TRAIN_parents_covered=True, DEV_waveforms_read=False, independent_acquisition_available=False,
            validated_480_sample_quiet_rows=0, quiet_evidence_scope='traceable archive metadata and saved legal TRAIN origin audit only'),
        native=dict(science_exit_code=0, replay_exit_code=0, science_ended_UTC=native['ended_utc'],
            science_peak_RSS_bytes=native['peak_rss_bytes'], replay= native_copies,
            replay_bitwise_array_members=sum(replay['array_members_bitwise_verified'].values()),
            replay_generation_chunks=207, replay_bootstrap_draws=1000, replay_matched_comparisons=7,
            replay_new_training_updates=0, replay_new_unique_draws=0, replay_DEV_decodes=0),
        verification=dict(source_files_verified=verify['source_files_verified'],
            completed_payloads_verified=verify['completed_files_sha256_verified'],
            shared_mean_stage_state_keys_bitwise_equal=178, paired_mean_stage_metric_steps=3000,
            full_random_branch_gradients_nonzero=True, conditional_values_recomputed=1152,
            visual_QA='receipts/response_007_20261013/deadline01/VISUAL_QA.json',
            final_PDF_sha256=sha(pdf)),
        source_commit='f27521a4f8c7721b46df540eeb3b034b1e35e44d',
        source_manifest_sha256=sha(manifest), COMPLETE_sha256=sha(SCI / 'COMPLETE.json'),
        scientific_SUMMARY_sha256=sha(SCI / 'TRAIN_diagnostics/SUMMARY.json'),
        report_HTML=(REPORT / 'N01_RESPONSE007_STAGE_REPORT.html').relative_to(ROOT).as_posix(),
        report_PDF=pdf.relative_to(ROOT).as_posix(),
        next_action=next_action, active_scientific_jobs=0, published=False,
        limits=['TRAIN-only finite-stage comparison; original fixed endpoint remains incomplete',
                'Frozen baseline noise unchanged; output-Q conditional widths still differ from REAL',
                'Long-range noise and independent-acquisition evidence remain unavailable',
                'Native replay proves reproduction, not physical correctness'],
        implementation_sha256=sha(Path(__file__)))
    bind_paths = [REC / 'VISUAL_QA.json', REC / 'POSTFLIGHT_RECEIPT.json', REC / 'ARTIFACT_VERIFICATION.json',
                  REC / 'PAIRED_MEAN3000_AND_RANDOM_GRADIENTS.json', REC / 'quiet_evidence/QUIET_EVIDENCE_AUDIT.json',
                  REPORT / 'BUILD_RECEIPT.json', ROOT / 'docs/RESPONSE_007_PROTOCOL.md',
                  ROOT / 'docs/RESPONSE_007_DEADLINE01.md', ROOT / 'response007_deadline01_config.json']
    final['evidence_sha256'] = {p.relative_to(ROOT).as_posix(): sha(p) for p in bind_paths}
    write(REC / 'FINAL_RECEIPT.json', final)

    backup = RUN / 'intermediate_backups/before_final_state'
    assert not backup.exists()
    backup.mkdir(parents=True)
    for p in [ROOT / 'PROGRAM.json', ROOT / 'jobs.json', ROOT / 'README.md', POINTER]:
        shutil.copy2(p, backup / p.name)
    readme = (ROOT / 'README.md').read_text(encoding='utf8')
    lines = readme.splitlines()
    assert lines[2].startswith('当前推进007')
    lines[2] = ('**007有限阶段已完成并核验：两臂各4,000 G，累计8,000 G（本执行新增7,999）及6,000训练D；'
                '原每臂19,000 G末点尚未完成，响应匹配未建立，当前科学作业0。** '
                '[三页报告](reports/response_007_20261013/deadline01/N01_RESPONSE007_STAGE_REPORT.html) · '
                '[完成回执](receipts/response_007_20261013/deadline01/FINAL_RECEIPT.json) · '
                '[比较口径与下一步](reports/response_007_20261013/deadline01/README.md)。本轮由主代理实施与自检。')
    lines[4] = lines[4].replace('当前主线 006', '此前 006')
    (ROOT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf8', newline='\n')
    report_readme = f'''# N01-007 有限阶段结果

两臂各4,000 G，累计8,000 G（本执行新增7,999）及6,000训练判别器更新；原每臂19,000 G终点未完成。科学状态为 `response_match_not_established`。

完整随机臂相对确定性臂的三假设最坏波形距离降低{improvement['wave']:.4%}，Q/PSD距离降低{improvement['QPSD']:.4%}。计算式为 `1 - max_h(distance_full[h]) / max_h(distance_deterministic[h])`，两臂都是4,000 G末点，使用同一固定匹配选择（每侧2,048条）和投影上下文。不同指标、不同臂的最坏假设可以不同。这不是相对REAL–REAL、旧006或物理等价性的改善率。

完整随机臂的最坏wave/QPSD/noise距离仍为同采集REAL–REAL对照的{control_ratios['wave']:.4f}/{control_ratios['QPSD']:.4f}/{control_ratios['noise']:.4f}倍。冻结基线未改善；输出Q条件宽度仍有偏差。低信号Q箱的三假设PSD宽度最大/最小为1.1006，父级95%区间为[1.0893, 1.1128]。各臂自身固定信号Q箱父人口不同，敏感性比值不能用作臂优劣排名。

在已追溯归档及合法TRAIN原记录审计中，没有得到480点静默窗口的证据。处理后的500点记录在120–140点出现半高上升沿；100点基线只是处理约定，不能证明纯电子学噪声。硬件时间标定、长程噪声和独立采集仍需补证。

[HTML报告](N01_RESPONSE007_STAGE_REPORT.html) · [PDF报告](N01_RESPONSE007_STAGE_REPORT.pdf) · [各假设完整摘要](SUMMARY.json) · [完成回执](../../../receipts/response_007_20261013/deadline01/FINAL_RECEIPT.json) · [本地原生重放回执](../../../runs/response_007_20261013/deadline01/runs/verification_deadline01/REPLAY_VERIFICATION.json)。服务器原生重放位置为 `/home/rylan/projects/ej276d-n01/response_007_20261013_deadline01/runs/verification_deadline01/REPLAY_VERIFICATION.json`。

训练和完整重放均原生退出0。76个冻结源文件、22个完成产物、178个共同状态项与前3,000步配对指标核验通过；207个生成分块、213个数组成员、1,000次bootstrap和7个分布比较复现一致。复现不增加训练或独立随机样本，没有读取DEV。

三页PDF已全部渲染、视觉检查并重新打开。最终PDF SHA256：`{sha(pdf)}`。视觉检查绑定见 [VISUAL_QA](../../../receipts/response_007_20261013/deadline01/VISUAL_QA.json)。全部审核由主代理完成，不称第二方独立审核。

下一步：{next_action} 先前的截止约束后来被取消，已完成的有限阶段及其冻结预算仍原样保留。当前没有新公开发布或上传共享。
'''
    (REPORT / 'README.md').write_text(report_readme, encoding='utf8', newline='\n')
    p = read(ROOT / 'PROGRAM.json')
    p.update(status='response007_bounded_stage_complete_reviewed', scientific_status='response_match_not_established',
        updated_UTC=now, last_verified_UTC=live['UTC'], active_scientific_jobs=0, active_repair=None,
        new_training_updates=100000, current_run_training_updates=8000,
        training_update_accounting=final['update_accounting'], current_arm=None,
        current_stage='bounded_stage_comparison_complete_reviewed', next_action=next_action,
        response007_G_updates=8000, response007_D_updates=6000, response007_C2ST_updates=0,
        response007_noise_updates=0, response007_training_pid=None, response007_supervisor_pid=None,
        response007_native_exit_code=0, response007_replay_native_exit_code=0,
        response007_original_endpoint_reached=False, response007_report=final['report_HTML'],
        response007_completion_receipt=(REC / 'FINAL_RECEIPT.json').relative_to(ROOT).as_posix(),
        response007_ended_UTC=native['ended_utc'], response007_publication_status='local_only_not_published')
    write(ROOT / 'PROGRAM.json', p)
    jobs = read(ROOT / 'jobs.json')
    jobs.update(updated_UTC=now, active_scientific_jobs=0)
    job = next(j for j in jobs['jobs'] if j['job_id'] == 'N01_RESPONSE007_DEADLINE01_20261013')
    job.update(status='bounded_stage_comparison_complete_reviewed', pid=None, supervisor_pid=None,
        ended_UTC=native['ended_utc'], native_exit_code=0, observed_cumulative_G_updates=8000,
        observed_new_execution_G_updates=7999, observed_D_updates=6000,
        original_endpoint_reached=False, completion_receipt=p['response007_completion_receipt'],
        report=final['report_HTML'], last_verified_UTC=live['UTC'], scientific_status='response_match_not_established')
    assert not any(j['job_id'] == 'N01_RESPONSE007_DEADLINE01_REPLAY' for j in jobs['jobs'])
    jobs['jobs'].append(dict(job_id='N01_RESPONSE007_DEADLINE01_REPLAY', status='completed_engineering_replay',
        host='100.82.231.102', pid=None, supervisor_pid=None, scientific_job=False,
        historical_supervisor_pid=replay_native['supervisor_pid'], started_UTC=replay_native['started_utc'],
        ended_UTC=replay_native['ended_utc'], native_exit_code=0, new_training_updates=0,
        new_unique_random_draws=0, DEV_decodes=0, engineering_forward_counts=replay['actual_engineering_forward_counts'],
        receipt=replay_path.relative_to(ROOT).as_posix()))
    write(ROOT / 'jobs.json', jobs)
    pointer = read(POINTER)
    pointer.update(status=p['status'], scientific_status=p['scientific_status'], active_scientific_jobs=0,
        active_repair=None, updated_UTC=now, last_verified_UTC=live['UTC'], plan_sha256=sha(ROOT / 'README.md'),
        next_action=next_action, response007_original_endpoint_reached=False,
        response007_completion_receipt=str(REC / 'FINAL_RECEIPT.json'), response007_report=str(REPORT / 'N01_RESPONSE007_STAGE_REPORT.html'),
        response007_native_exit_code=0, response007_replay_native_exit_code=0,
        response007_publication_status='local_only_not_published')
    write(POINTER, pointer)
    print(json.dumps(dict(status=final['status'], PDF_sha256=sha(pdf), G=8000, D=6000,
        reduction=improvement, report=final['report_HTML'], final_receipt=p['response007_completion_receipt']), ensure_ascii=False))


if __name__ == '__main__':
    main()
