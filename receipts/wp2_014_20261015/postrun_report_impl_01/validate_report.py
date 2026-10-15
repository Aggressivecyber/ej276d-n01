"""Reopen report text/tables/manifest; compare saved JSON facts without scientific replay."""
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime, timezone
import csv
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'reports/wp2_014_20261015'
checks = []


def test(name, passed):
    assert passed, name
    checks.append(dict(name=name, passed=True))


def sha(path):
    assert path.suffix.lower() not in {'.npz', '.pt', '.pth', '.npy'}
    return hashlib.sha256(path.read_bytes()).hexdigest()


def j(path):
    assert path.suffix == '.json'
    return json.loads(path.read_bytes())


def table_blocks(text):
    return [[line.strip().strip('|').split('|') for line in block.splitlines()]
            for block in re.findall(r'(?:^\|.*\|\n)+', text, flags=re.M)]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.links, self.headings, self.tables = [], [], 0, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'img':
            self.images.append(a['src'])
        if tag == 'a':
            self.links.append(a['href'])
        self.headings += tag == 'h2'
        self.tables += tag == 'table'


def main():
    manifest = j(OUT / 'report_manifest.json')
    receipt = j(HERE / 'REPORT_BUILD_RECEIPT.json')
    data = j(OUT / 'REPORT_DATA.json')
    text = (OUT / 'N01_WP2_014_REPORT.md').read_text(encoding='utf-8')
    page = Page()
    page.feed((OUT / 'N01_WP2_014_REPORT.html').read_text(encoding='utf-8'))
    for rel, meta in manifest['outputs'].items():
        path = OUT / rel
        assert path.stat().st_size == meta['bytes'] and sha(path) == meta['sha256'], rel
    test('all15_report_outputs_size_SHA_match_manifest_and_build_receipt',
         len(manifest['outputs']) == 15 and receipt['outputs'] == manifest['outputs']
         and receipt['manifest_sha256'] == sha(OUT / 'report_manifest.json')
         and receipt['source_sha256'] == sha(HERE / 'build_report.py'))
    for rel, meta in manifest['inputs'].items():
        path = ROOT / rel
        assert path.stat().st_size == meta['bytes'] and sha(path) == meta['sha256'], rel
    test('all_report_input_bytes_current_unchanged', True)
    test('HTML12_sections_3_images_and_all_relative_links_exist',
         page.headings == 12 and len(page.images) == 3
         and all((OUT / name).resolve().exists() for name in page.links + page.images))
    test('HTML_is_self_contained_no_remote_assets_or_scripts',
         all(not name.startswith(('http:', 'https:', '//')) for name in page.links + page.images)
         and '<script' not in (OUT / 'N01_WP2_014_REPORT.html').read_text(encoding='utf-8').lower())
    blocks = table_blocks(text)
    wave_table = next(rows for rows in blocks if any('相对A改善/%' in x for x in rows[0]))
    rows = [[x.strip() for x in row] for row in wave_table[2:]]
    test('twelve_unrounded_source_wave_cells_present_in_report', len(rows) == len(data['cells']) == 12)
    for row, cell in zip(rows, data['cells']):
        expected = [cell['candidate'], str(cell['seed']), cell['hypothesis'], f"{cell['A_wave']:.9f}",
                    f"{cell['initial_wave']:.9f}", f"{cell['final_wave']:.9f}", f"{cell['wave_improvement_percent']:.12f}",
                    'PASS' if cell['usefulness_at_least_2percent'] else 'FAIL']
        assert row == expected
    test('wave_table_all_A_initial_final_percent_and_frozen_gate_labels_match_saved_JSON', True)
    test('critical_threshold_full_precision_and_failure_remain',
         '1.9887788589535327%' in text and '0.15042006897712382' in text
         and '0.15255321144257886' in text
         and sum(c['usefulness_at_least_2percent'] for c in data['cells'] if c['candidate'] == 'C1') == 5)
    q_table = next(rows for rows in blocks if any('A QPSD' in x for x in rows[0]))
    for row, cell in zip(q_table[2:], data['cells']):
        actual = [v.strip() for v in row]
        A, initial, final = [cell[k] for k in ['A', 'initial', 'final']]
        expected = [cell['candidate'], str(cell['seed']), cell['hypothesis'], f"{A['QPSD']:.9f}",
                    f"{initial['QPSD']:.9f}", f"{final['QPSD']:.9f}", f"{100*(final['QPSD']/A['QPSD']-1):+.5f}", f"{final['noise']:.9f}"]
        assert actual == expected
    test('all12_native_QPSD_noise_table_rows_match_saved_evidence', len(q_table[2:]) == 12)
    test('all15_positiveT_by_ALL6_noninferiority_and_absolute_fail_preserved',
         len(data['all_components']) == 264
         and sum(r['noninferiority'] == 'PASS' for r in data['all_components']) == 180
         and sum(r['noninferiority'] == 'pending_not_calibrated' for r in data['all_components']) == 84
         and all(not c['absolute_T_all_cells'] for c in data['development_choice']['candidates'].values()))
    for filename, expected in [('WP2_ALL6_WAVE.csv', 12), ('WP2_ALL6_COMPONENTS.csv', 264),
                               ('WP2_GRADIENT_STRATA.csv', 104), ('WP2_R8_MOMENTS.csv', 36),
                               ('WP2_OUTPUT_Q_CONDITIONS.csv', 576), ('WP2_SOURCE_INPUT_TRENDS.csv', 10368)]:
        with (OUT / filename).open(encoding='utf-8-sig', newline='') as f:
            assert len(list(csv.DictReader(f))) == expected
    test('six_CSV_tables_have12_264_104_36_576_10368_complete_rows', True)
    for stem, width, height in [('01_wave_usefulness', 2360, 1020), ('02_absolute_tolerance', 2360, 1600),
                               ('03_conditional_gradients', 2440, 920)]:
        png = (OUT / 'figures' / (stem + '.png')).read_bytes()
        actual_width, actual_height = struct.unpack('>II', png[16:24])
        assert png[:8] == b'\x89PNG\r\n\x1a\n' and abs(actual_width-width) <= 1 and abs(actual_height-height) <= 1
        svg = ET.parse(OUT / 'figures' / (stem + '.svg'))
        svg_text = ''.join(svg.getroot().itertext())
        assert not re.search('[\u4e00-\u9fff]', svg_text)
    test('three200dpi_English_figures_SVG_XML_and_PNG_dimensions_valid', True)
    test('raw4_vs_derived8_finite_support_and_limit_definitions_present',
         all(term in text for term in ['r4原报告', 'r8派生', '分母7', '逐对共同有限',
             '原REAL TRAIN median未重拟合', '完整wave/QPSD/noise经验距离沿用冻结producer',
             '完整preADC384', 'probe完整窗/peak_index', '原始Y没有第二份独立副本']))
    test('gradient_norm_ratio_is_nonadditive_not_event_contribution',
         '子集梯度不能相加解释为事件贡献、损失比例或因果占比' in text
         and len(data['gradient_rows']) == 104 and all(not r['additive_event_contribution'] for r in data['gradient_rows']))
    test('all_four_evaluation_suites_reported_without_conditional_T_claim',
         len(data['output_Q_conditions']) == 576 and len(data['source_input_trends']) == 10368
         and all(r['supported'] for r in data['output_Q_conditions'])
         and len(data['output_Q_final_worst_by_cell']) == 12 and len(data['source_input_final_summary']) == 12
         and '条件T未标定' in text and '真实条件均值和物理源权重继续保持未识别/unknown' in text
         and '13种子集' not in text and '银行' not in text)
    for summary in data['output_Q_final_worst_by_cell']:
        values = [r for r in data['output_Q_conditions'] if r['trial'] == summary['trial']
                  and r['hypothesis'] == summary['hypothesis'] and r['arm'] == 'final']
        assert len(values) == 16
        assert summary['worst_PSD']['PSD_W1_defined'] == max(r['PSD_W1_defined'] for r in values)
        assert abs(summary['worst_rise']['rise_mean_difference_ns']) == max(abs(r['rise_mean_difference_ns']) for r in values)
        assert abs(summary['worst_FWHM']['FWHM_mean_difference_ns']) == max(abs(r['FWHM_mean_difference_ns']) for r in values)
    test('all12_final_cells_preserve_each_metric_worst_bank_bin', True)
    test('actual_science_cost_G8000_AD9872_head8653376_and_audit_zero',
         data['native_cost']['G_updates'] == 8000 and data['native_cost']['new_AD_queries'] == 9872
         and data['native_cost']['head_total_kernel_rows'] == 8653376
         and data['new_NPZ_CP_NN_AD_random_CDF_training'] == 0
         and data['native_exit']['exit_code'] == data['raw_CPU_exit']['exit_code'] == 0)
    test('codepath_C1_weights_false_science_false',
         data['development_choice']['development_codepath'] == 'C1'
         and data['weights_promoted'] is False and data['scientific_acceptance'] is False
         and 'response_match_not_established' in text)
    record = dict(schema='n01/wp2-report-content-figure-QA/v1', UTC=datetime.now(timezone.utc).isoformat(),
        status='PASS_CONTENT_AND_FIGURES_HTML_VISUAL_PENDING', passed=True, checks=checks,
        report={name:dict(bytes=(OUT/name).stat().st_size, sha256=sha(OUT/name)) for name in
                ['N01_WP2_014_REPORT.md','N01_WP2_014_REPORT.html','report_manifest.json']},
        source_sha256=sha(HERE / 'build_report.py'), QA_script_sha256=sha(Path(__file__)),
        figure_visual_inspection='Three generated PNGs viewed locally; labels, legends, threshold and failed-cell annotation checked. Updated near-one heatmap values use three decimals.',
        HTML_layout='Uses the 013 blue-white style and relative local SVGs. Root browser/render review remains pending.',
        new_NPZ_CP_NN_AD_random_CDF_training_SSH=0, scientific_acceptance=False, weights_promoted=False,
        root_render_candidate=str(OUT / 'N01_WP2_014_REPORT.html'))
    path = HERE / 'REPORT_QA.json'
    with path.open('x',encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(passed=True, checks=len(checks), QA_path=str(path), QA_sha256=sha(path)), ensure_ascii=False))


if __name__ == '__main__':
    main()
