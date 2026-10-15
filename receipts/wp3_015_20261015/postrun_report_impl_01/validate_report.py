"""Reopen local report and compare saved JSON; never replay scientific operations.

Reuse WP2/014 HTML, SHA and Markdown-table validation helpers. This validator
independently reaggregates the twelve saved wave cells and checks export coverage.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import importlib.util
import json
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET

ROOT=Path('E:/GAN/research/N01')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'reports/wp3_015_20261015'
ART=ROOT/'runs/wp3_015_20261015/selected_collected_01/runs/primary/artifacts'
TMP=Path('E:/GAN/runtime_tmp/wp3_015_report')
for name in ['TMP','TEMP','TMPDIR']:
    os.environ[name]=str(TMP)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
HELPER=ROOT/'receipts/wp2_014_20261015/postrun_report_impl_01/validate_report.py'
spec=importlib.util.spec_from_file_location('wp2_validation_helpers',HELPER)
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
j,sha,test,Page=base.j,base.sha,base.test,base.Page


def rows(name):
    path=OUT/'tables'/name
    assert path.read_bytes()[:3]==b'\xef\xbb\xbf',name+' UTF8 BOM'
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def equal(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-14)


def main():
    manifest=j(OUT/'report_manifest.json')
    sources=j(OUT/'SOURCES_SHA256.json')
    data=j(OUT/'REPORT_DATA.json')
    receipt=j(HERE/'REPORT_BUILD_RECEIPT.json')
    final=j(ROOT/'receipts/wp3_015_20261015/postrun_independent_02/FINAL_DEVELOPMENT_SELECTION.json')
    review=j(ROOT/'receipts/wp3_015_20261015/postrun_independent_02/FINAL_INDEPENDENT_REVIEW.json')
    text=(OUT/'REPORT.md').read_text(encoding='utf-8')
    html=(OUT/'REPORT.html').read_text(encoding='utf-8')
    assert len(manifest['outputs'])==22
    for name,meta in manifest['outputs'].items():
        path=OUT/name
        assert path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256'],name
    test('all22_report_outputs_reopened_size_and_SHA_match',True)
    test('build_receipt_and_source_hash_match',receipt['outputs']==manifest['outputs'] and
        receipt['manifest_sha256']==sha(OUT/'report_manifest.json') and
        sources['report_source']['sha256']==sha(HERE/'build_report.py'))
    for name,meta in sources['inputs'].items():
        path=ROOT/name
        assert path.suffix=='.json' and path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256'],name
    test('all_local_saved_JSON_sources_current_bytes_unchanged',sources['inputs']==manifest['inputs'])
    for helper in [sources['reused_helper'],sources['report_source']]:
        assert sha(ROOT/helper['path'])==helper['sha256']
    test('reused_WP2_helper_and_WP3_builder_hashes_current',True)
    page=Page()
    page.feed(html)
    test('HTML12_sections_4_images_and_relative_links_exist',page.headings==12 and len(page.images)==4 and
        all((OUT/name).resolve().exists() or name=='VALIDATION.json' for name in page.images+page.links))
    test('HTML_no_remote_assets_or_scripts',all(not name.startswith(('http:','https:','//')) for name in page.images+page.links) and '<script' not in html.lower())
    test('UTF8_report_has_no_replacement_glyph', '\ufffd' not in text+html)

    wave=rows('ALL6_WAVE.csv')
    component=rows('ALL6_COMPONENTS.csv')
    test('12_wave_and264_component_CSV_rows',len(wave)==12 and len(component)==264)
    for c,w in zip(data['cells'],wave):
        initial=j(ART/f"seed{c['seed']}_{c['candidate']}"/'initial_evaluation/METRICS.json')
        end=j(ART/f"seed{c['seed']}_{c['candidate']}"/'final_evaluation/METRICS.json')
        for arm,doc,key in [('A',initial,'A_wave'),('initial',initial,'common_initial_wave'),('final',end,'final_wave')]:
            expected=sum(doc['metrics'][f"bank{bank}_{arm}_{c['hypothesis']}"]['values']['wave'] for bank in [0,1])/2
            assert equal(c[key],expected) and equal(w[key],expected)
        percent=100*(c['A_wave']-c['final_wave'])/c['A_wave']
        assert equal(percent,w['wave_improvement_vs_A_percent'])
        assert (w['usefulness_ALL6_cell']=='True')==(percent>=2)
        reference=next(f for f in final['cells'] if (f['candidate'],f['seed'],f['hypothesis'])==(c['candidate'],c['seed'],c['hypothesis']))
        for r in [r for r in component if (r['candidate'],int(r['seed']),r['hypothesis'])==(c['candidate'],c['seed'],c['hypothesis'])]:
            d=reference['components'][r['component']]
            assert equal(r['A'],d['A']) and equal(r['common_initial'],d['common_initial']) and equal(r['final'],d['final'])
            assert r['status']==d['status']
    test('all12_wave_cells_independently_reaggregate_original_METRICS_and264_export_values_match_final',True)
    test('usefulness_M0_6of6_M1_0of6',sum(w['usefulness_ALL6_cell']=='True' for w in wave if w['candidate']=='M0')==6 and
        sum(w['usefulness_ALL6_cell']=='True' for w in wave if w['candidate']=='M1')==0)
    test('180_noninferior84_pending_and_absolute_fail_preserved',sum(r['status']=='noninferior_to_both_references' for r in component)==180 and
        sum(r['status']=='pending_not_calibrated' for r in component)==84 and all(not c['absolute_T_ALL6'] for c in final['candidates'].values()))
    for candidate in ['M0','M1']:
        score=sum(float(w[k]) for w in wave if w['candidate']==candidate for k in ['final_bank0_wave','final_bank1_wave'])/12
        assert equal(score,final['candidates'][candidate]['final_wave_mean12'])
    test('frozen_mean12_scores_and_development_M0_selection',final['selected_research_reference']=='M0' and
        final['selected_mean_adapter_enabled'] is False and final['independent_raw_saved_array_readback_complete'] is True and
        review['passed'] is True and data['development_choice']==final)
    test('full_precision_scores_and014_historical5of6_remain',all(term in text for term in
        ['0.15012505860962605','0.15243191496746586','0.15042006897712382','历史用途5/6']))
    blocks=base.table_blocks(text)
    block=next(b for b in blocks if any('相对A改善/%' in h for h in b[0]))
    test('Markdown12_wave_rows_match_export',len(block[2:])==12)
    for r,c in zip(block[2:],data['cells']):
        expected=[c['candidate'],str(c['seed']),c['hypothesis'],f"{c['A_wave']:.9f}",f"{c['common_initial_wave']:.9f}",
            f"{c['final_wave']:.9f}",f"{c['wave_improvement_vs_A_percent']:.9f}",'PASS' if c['usefulness_ALL6_cell'] else 'FAIL']
        assert [x.strip() for x in r]==expected
    test('Markdown_all_A_initial_final_percent_gate_labels_correct',True)

    expected_counts={'OUTPUT_Q_CONDITIONS.csv':576,'SOURCE_INPUT_TRENDS.csv':10368,'POPULATION_FEATURES.csv':1152,
        'DERIVED8_MOMENTS.csv':36,'MEAN_RESIDUAL_GATE.csv':6,'OUTPUT_Q_FINAL_SUMMARY.csv':12,'SOURCE_INPUT_FINAL_SUMMARY.csv':12}
    for name,count in expected_counts.items():
        assert len(rows(name))==count,(name,count)
    flag_rows=rows('ALL_FLAGS.csv')
    assert len(flag_rows)==len(data['all_flags'])
    test('ten_CSV_tables_BOM_and_complete_four_axis_coverage',len(list((OUT/'tables').glob('*.csv')))==10)
    for r in data['output_Q_final_worst_by_cell']:
        subset=[x for x in data['output_Q_conditions'] if (x['trial'],x['hypothesis'],x['arm'])==(r['trial'],r['hypothesis'],'final')]
        assert len(subset)==16 and all(x['supported'] for x in subset)
        assert equal(r['PSD_W1_max'],max(x['PSD_W1_defined'] for x in subset))
        assert equal(abs(r['rise_signed_worst']),max(abs(x['rise_mean_difference_ns']) for x in subset))
        assert equal(abs(r['FWHM_signed_worst']),max(abs(x['FWHM_mean_difference_ns']) for x in subset))
    test('all12_final_conditional_worst_bank_bin_values_preserved',True)
    # Reopen all 72 source reports to verify every flag has the original denominator.
    original_flag_count=0
    for p in data['derived8_moments']:
        for bank in [0,1]:
            doc=j(ART/p['trial']/p['snapshot']/f"bank{bank}_{p['arm']}_{p['hypothesis']}_REPORT.json")
            matches=[r for r in data['all_flags'] if (r['trial'],r['snapshot'],r['arm'],r['hypothesis'],r['bank'])==
                (p['trial'],p['snapshot'],p['arm'],p['hypothesis'],bank)]
            expected=sum(len(doc['population'][domain+'_flags']) for domain in ['fake','real'])+len(doc['population']['real_unknown_flags'])
            assert len(matches)==expected
            original_flag_count+=expected
            for domain in ['fake','real']:
                for flag,c in doc['population'][domain+'_flags'].items():
                    item=next(r for r in matches if r['domain']==domain and r['flag']==flag and r['status']=='saved_report_value')
                    assert all(item[k]==v for k,v in c.items())
    test('all_flags_reopen72_REPORTs_original_counts_denominators_and_unknowns',len(flag_rows)==original_flag_count)

    for stem,width,height in [('01_wave_selection',2680,1020),('02_absolute_tolerance',2760,1600),
        ('03_mean_residual_gate',2320,900),('04_repeated_parent_moments',2520,940)]:
        binary=(OUT/'figures'/(stem+'.png')).read_bytes()
        w,h=struct.unpack('>II',binary[16:24])
        assert binary[:8]==b'\x89PNG\r\n\x1a\n' and abs(w-width)<=1 and abs(h-height)<=1
        svg=ET.parse(OUT/'figures'/(stem+'.svg'))
        assert not re.search('[\u4e00-\u9fff]',''.join(svg.getroot().itertext()))
    test('four_English_PNG_SVG_reopened_dimensions_and_XML_valid',True)
    test('native4_derived8_and_covariance_scope_precise',all(term in text for term in
        ['native4','derived8','分母7','经验秩≤7','理论协方差秩≤8','逐对共同有限','半正定','未物化每父稠密矩阵','原REAL TRAIN median未重拟合','REAL mean摘要NPZ仅SHA未decode']))
    test('distance_evidence_layers_and_unseen_scope_preserved',all(term in text for term in
        ['数值距离沿用冻结producer','未重播完整经验CDF或loss','纯信号完整preADC384','CP tensor未解码','原Y无第二份独立副本','条件T未标定','真实条件响应','跨采集物理复用']))
    test('science_cost8000G8000AD_and_cumulative175667_not_readded',data['native_cost']['G_updates']==8000 and data['native_cost']['new_AD_queries']==8000 and
        data['native_cost']['checkpoint_loads']==4 and data['native_cost']['old_A_rows']==8613952 and
        data['native_cost']['mean_adapter_rows']==4306976 and data['native_cost']['hard_ADC_rows']==26474784 and data['cumulative_training_unchanged']==175667)
    test('CPU_saved_readback161SHA160decode36pairs_no_science',data['readback_actual_counts']['raw_files_SHA_checked']==161 and
        data['readback_actual_counts']['waveform_rows_decoded']==1898784 and data['readback_actual_counts']['bank_pairs_verified']==36 and
        data['new_scientific_or_remote_operations']==0 and data['readback_operations']==285)
    test('two_distinct_RSS_measurements_preserved',data['readback_resources']['worker_getrusage_peak_RSS_bytes']==489336832 and
        data['readback_resources']['supervisor_sampled_peak_owned_RSS_bytes']==402255872 and all(term in text for term in
        ['worker getrusage峰值489336832','supervisor采样owned-tree峰值402255872','不同测量']))
    test('weights_false_science_false_next_qsig_single_factor',data['weights_promoted'] is False and data['scientific_acceptance'] is False and
        all(term in text for term in ['response_match_not_established','剩余最多2次只检验单因素q_sig','不承诺同时修复peak、rise或电子学DC']))
    result=dict(schema='n01/wp3-015-report-validation/v1',UTC=datetime.now(timezone.utc).isoformat(),passed=True,
        status='PASS_CONTENT_AND_STRUCTURE_VISUAL_PENDING',checks=base.checks,checks_passed=len(base.checks),
        report=dict(path=str(OUT/'REPORT.html'),sha256=sha(OUT/'REPORT.html')),source_sha256=sha(HERE/'build_report.py'),
        validation_script_sha256=sha(Path(__file__)),reused_validation_helper_sha256=sha(HELPER),CSV_tables=10,
        flag_rows=original_flag_count,English_figures=4,figure_visual_inspection='pending direct PNG review',
        HTML_layout='root representative browser rendering pending',new_NPZ_CP_NN_AD_G_random_TRAINDEV_GPU_CDF_loss_SSH=0,
        weights_promoted=False,scientific_acceptance=False)
    with (OUT/'VALIDATION.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    with (HERE/'REPORT_QA.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(dict(passed=True,checks=len(base.checks),flag_rows=original_flag_count,validation=str(OUT/'VALIDATION.json')),ensure_ascii=False))


if __name__=='__main__':
    main()
