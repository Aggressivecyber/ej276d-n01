"""Seal the reviewed015 report and a small SHA-verified D-drive backup."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import struct

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
OUT = HERE.parent
REPORT = ROOT / 'reports/wp3_015_20261015'
CURRENT = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value, new=False):
    with path.open('x' if new else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), bytes=path.stat().st_size)


def png(path):
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    w, h = struct.unpack('>II', data[16:24])
    assert w >= 1400 and h >= 700
    return dict(**ref(path), width=w, height=h)


def main():
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = read(REPORT / 'report_manifest.json')
    for name, row in manifest['outputs'].items():
        path = REPORT / name
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], name
    qa_path = OUT / 'postrun_report_impl_01/REPORT_QA.json'
    qa = read(qa_path)
    assert qa['passed'] and qa['checks_passed'] == 25
    assert qa['new_NPZ_CP_NN_AD_G_random_TRAINDEV_GPU_CDF_loss_SSH'] == 0
    render_path = HERE / 'html_qa_01/RENDER_RECEIPT.json'
    renders = read(render_path)
    assert renders['report_sha256'] == sha(REPORT / 'REPORT.html')
    for row in renders['renders'].values():
        assert row['Edge_native_exit'] == 0 and sha(Path(row['png'])) == row['png_sha256']
    visual_path = HERE / 'REPORT_VISUAL_QA.json'
    visual = dict(schema='n01/wp3-root-representative-render-QA/v1', UTC=stamp,
        passed=True, status='PASS_REPRESENTATIVE_NATIVE_VIEWS', report=ref(REPORT/'REPORT.html'),
        native_render_receipt=ref(render_path), original_HTML_top_viewport=png(HERE/'html_qa_01/top.png'),
        exact_later_section_previews={n: png(HERE/('html_qa_01/'+n+'.png')) for n in ('wave','conditional','readback')},
        original_English_figures={Path(n).stem: png(REPORT/n) for n in manifest['outputs'] if n.startswith('figures/') and n.endswith('.png')},
        observations=['All four original English PNGs viewed; 2% line,6/6 versus0/6, initial score and final scores readable.',
            'The absolute-T heatmap preserves1.00005 and the large peak/rise/FWHM/baseline failures.',
            'Original HTML top and exact wave/conditional/readback sections viewed; Chinese text, all12 wave rows and table columns fit.',
            'The saved-array scope and distinct RSS measures are legible; all4 native Edge exits were0.'],
        scope='Original top viewport plus exact later section subtrees with original CSS/assets; representative rendering, not full interactive scrolling.',
        original_report_modified=False, new_scientific_CP_NN_AD_G=0)
    write(visual_path, visual, new=True)
    native_path = OUT/'postrun_root_01/NATIVE_COMPLETION_READBACK.json'
    native = read(native_path)
    cpu_path = OUT/'readback_execution_01/WAKE_COMPLETION_SUMMARY.json'
    cpu = read(cpu_path)
    review_path = OUT/'postrun_independent_02/FINAL_INDEPENDENT_REVIEW.json'
    selection_path = OUT/'postrun_independent_02/FINAL_DEVELOPMENT_SELECTION.json'
    decision = read(selection_path)
    assert read(review_path)['passed'] and decision['passed'] and not decision['provisional']
    assert decision['selected_research_reference'] == 'M0' and not decision['selected_mean_adapter_enabled']
    assert native['actual_counts']['G_updates'] == native['actual_counts']['new_AD_queries'] == 8000
    assert cpu['operations_entered'] == cpu['operations_completed'] == 285
    backup = Path('D:/GAN_backups/N01/wp3_015_20261015/closeout_01')
    backup.mkdir(parents=True, exist_ok=False)
    backup_files = [REPORT/n for n in manifest['outputs']] + [REPORT/'report_manifest.json', REPORT/'VALIDATION.json',
        qa_path, visual_path, render_path, review_path, selection_path, native_path, cpu_path,
        OUT/'postrun_independent_02/FINAL_CONCLUSION_ZH.md', OUT/'postrun_independent_02/FINAL_OUTPUT_RECEIPT.json',
        OUT/'postrun_report_impl_01/FIGURE_VISUAL_QA.json', OUT/'postrun_report_impl_01/build_report.py',
        OUT/'postrun_report_impl_01/validate_report.py', OUT/'readback_collection_01/COLLECTION_RECEIPT.json',
        OUT/'collection_01/COLLECTION_RECEIPT.json', OUT/'collection_01/REMOTE_FULL_INVENTORY.json',
        HERE/'AGENT_MODEL_POLICY_SOLMAX.json']
    backup_files += [p for p in (ROOT/'runs/wp3_015_20261015/independent_readback_collected_01').rglob('*') if p.is_file()]
    backup_files += [HERE/('html_qa_01/'+n+'.png') for n in ('top','wave','conditional','readback')]
    backup_files = list(dict.fromkeys(backup_files))
    assert sum(p.stat().st_size for p in backup_files) < 32*1024**2
    inventory = {}
    for path in backup_files:
        assert path.suffix.lower() not in ('.pt','.npz','.tar','.gz')
        name = path.relative_to(ROOT).as_posix()
        target = backup/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert sha(path) == sha(target)
        inventory[name] = dict(sha256=sha(target), bytes=target.stat().st_size)
    write(backup/'MANIFEST.json', dict(UTC=stamp, files=inventory, original_raw_or_checkpoint_included=False), new=True)
    (backup/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(inventory.items())), encoding='utf-8')
    backup_receipt = OUT/'D_CLOSEOUT_BACKUP_RECEIPT.json'
    write(backup_receipt, dict(UTC=stamp, status='all_copied_files_SHA256_verified', destination=str(backup),
        files=len(inventory), bytes=sum(v['bytes'] for v in inventory.values()), manifest_sha256=sha(backup/'MANIFEST.json'),
        original_raw_or_checkpoint_included=False, all_scientific_data_on_E_D=True), new=True)
    final_path = OUT/'FINAL_RESULT_RECEIPT.json'
    final = dict(schema='n01/wp3-015-final-result/v1', UTC=stamp,
        status='015_engineering_experiment_bounded_audit_and_report_complete', run_id='wp3_015_20261015',
        training_source_commit='15b60f3fbe02641bcd18693486698f4fed0875bf',
        saved_array_CPU_source_commit='56aafccbb88b206aaf2ce68a6df8126477078756',
        all_four_trials_native0=True, actual_counts=native['actual_counts'],
        cumulative_training_before015=167667, cumulative_training_after015=175667,
        additional_scientific_G_AD_NN_CP_after_original015=0,
        global_short_trials_completed=8, WP3_short_trials_completed=4, WP3_remaining_maximum=2,
        original_training_collection=dict(files=139,total=300,bytes=76908267,thin_checkpoints=12,NPZ=0),
        saved_array_audit=dict(actual_counts=cpu['actual_counts'], operations_entered=285, operations_completed=285,
            native_child_exit=0,native_supervisor_exit=0,collected_small_files=19,collected_bytes=513764),
        raw161_NPZ_retained_remote=True, independent_complete_CDF_loss_replay=False,
        independent_C1_median_refit=False, full_preADC384_independent_readback=False,
        REAL_mean_summary_SHA_only_no_decode=True,
        independent_review=ref(review_path), development_selection=ref(selection_path),
        selected_research_reference='M0', mean_adapter_enabled=False,
        M0_mean12_final_wave=.15012505860962605, M1_mean12_final_wave=.15243191496746586,
        common_initial_mean12_wave=.15042006897712382, M0_useful_cells=6, M1_useful_cells=0,
        dual_reference_positive_T_noninferiority='Both candidatesALL6; every15 calibrated component passes',
        absolute_T_ALL6_passed=False, uncalibrated_T='pending_never_automatic_PASS',
        original014_usefulness_5of6_preserved=True, weights_promoted=False, scientific_acceptance=False,
        scientific_status='response_match_not_established', cross_acquisition_reuse='not_verified',
        report_MD=ref(REPORT/'REPORT.md'), report_HTML=ref(REPORT/'REPORT.html'),
        report_manifest=ref(REPORT/'report_manifest.json'), report_content_QA=ref(qa_path), report_visual_QA=ref(visual_path),
        D_closeout_backup=ref(backup_receipt), model_policy=ref(HERE/'AGENT_MODEL_POLICY_SOLMAX.json'),
        next_stage='016 two-coefficient signal-charge implementation, independent review and frozen bounded launch',
        whole_WP3_complete=False, whole_six_workpackage_project_complete=False)
    write(final_path, final, new=True)
    paths = [ROOT/n for n in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')] + [CURRENT]
    snap = HERE/'metadata_before_report_closeout'
    snap.mkdir(exist_ok=False)
    for p in paths:
        shutil.copy2(p, snap/p.name)
    master, program, jobs, current = [read(p) for p in (paths[0],paths[1],paths[2],CURRENT)]
    assert program['new_training_updates'] == 175667
    assert all(d['active_scientific_jobs'] == 0 for d in (master,program,jobs,current))
    result_ref = dict(**ref(final_path), status=final['status'], scientific_acceptance=False, weights_promoted=False)
    report_ref = dict(MD=ref(REPORT/'REPORT.md'), HTML=ref(REPORT/'REPORT.html'), visual_QA=ref(visual_path))
    for value in (master,program,current):
        value.update(wp3_015_completed=True,wp3_015_result=result_ref,wp3_015_report=report_ref,
            wp3_015_report_visual_QA_complete=True,updated_UTC=stamp,
            status='WP3_015_closed_016_implementation_in_progress',current_stage='WP3_016_implementation_pending_review',
            active_research=None,active_audit=None,
            next_action='Complete016 implementation and independent review, freeze exact source/config and start the registered two-trial charge test.')
        if 'wp3_execution' in value:
            value['wp3_execution']['status']='completed_native0_selectively_collected_independently_reviewed_reported'
    wp3=next(w for w in master['work_packages'] if w['id']=='WP3')
    wp3.update(status='015_closed_016_implementation_in_progress',result015=result_ref,report015=report_ref)
    wp3['execution']['status']='completed_native0_selectively_collected_independently_reviewed_reported'
    for job in jobs['jobs']:
        if job['job_id']=='N01_WP3_015_20261015':
            job.update(status='completed_native0_collected_independently_reviewed_reported_closed',result=result_ref,report=report_ref)
            job['execution']['status']=wp3['execution']['status']
        elif job['job_id']=='N01_WP3_015_SAVED_ARRAY_READBACK_20261015':
            job['status']='completed_native0_collected_independently_reviewed_closed'
    jobs['updated_UTC']=stamp
    text=paths[3].read_text(encoding='utf-8-sig')
    text=text.replace('# N01 当前执行：WP3/015 独审闭合，报告与016设计正在完成', '# N01 当前执行：WP3/015 报告完成，016正在实现',1)
    text=text.replace('[015最终独审]', '[015中文报告](reports/wp3_015_20261015/REPORT.md) · [HTML](reports/wp3_015_20261015/REPORT.html) · [最终回执](receipts/wp3_015_20261015/FINAL_RESULT_RECEIPT.json) · [015最终独审]',1)
    paths[3].write_text(text,encoding='utf-8')
    current['plan_sha256']=sha(paths[3])
    write(paths[0],master)
    program['master_plan_sha256']=current['master_plan_sha256']=sha(paths[0])
    for p,value in ((paths[1],program),(paths[2],jobs),(CURRENT,current)):
        write(p,value)
    write(HERE/'FINAL_REPORT_CLOSEOUT_STATE.json', dict(UTC=stamp,final_result=ref(final_path),
        metadata={str(p):sha(p) for p in paths}, report_complete=True,cumulative_training_updates=175667,
        whole_project_complete=False,new_scientific_operations=0),new=True)
    supplement=[final_path,backup_receipt,HERE/'FINAL_REPORT_CLOSEOUT_STATE.json']+paths[:-1]
    supplement_inventory={}
    for p in supplement:
        name=p.relative_to(ROOT).as_posix()
        target=backup/'delivery'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
        assert sha(p)==sha(target)
        supplement_inventory[name]=dict(sha256=sha(target),bytes=target.stat().st_size)
    write(backup/'DELIVERY_MANIFEST.json',dict(UTC=stamp,base_manifest_sha256=sha(backup/'MANIFEST.json'),files=supplement_inventory),new=True)
    print(json.dumps(dict(status=final['status'],final_result_sha256=sha(final_path),
        D_backup_files=len(inventory),D_backup_bytes=sum(v['bytes'] for v in inventory.values()),
        report_visual_QA='PASS_REPRESENTATIVE_NATIVE_VIEWS',cumulative_training_updates=175667,new_scientific_operations=0)))


if __name__=='__main__':
    main()
