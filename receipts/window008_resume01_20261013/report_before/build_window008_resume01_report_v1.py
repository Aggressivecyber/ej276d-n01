"""Render frozen endpoint artifacts; never load a model or original waveforms."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import html
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']='E:/GAN/runtime_tmp/matplotlib_window008_resume01'
os.environ['TMP']=os.environ['TEMP']='E:/GAN/runtime_tmp'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
sys.path.insert(0,str(ROOT/'runtime/plotting_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,
    'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8.5,'axes.unicode_minus':False,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42})
ARMS=('deterministic','full','simple');HYP=('fast','slow','common_level');METRICS=('wave','QPSD','noise')
LABEL={'deterministic':'确定性 U-Net','full':'完整随机 U-Net','simple':'简单随机响应',
       'fast':'短相关','slow':'慢相关','common_level':'恒定公共分量',
       'wave':'256点波形','QPSD':'电荷与PSD','noise':'100点基线'}
COLORS={'fast':'#007f8b','slow':'#cc7626','common_level':'#7958a0','REAL':'#26394b',
        'deterministic':'#22718b','full':'#b15a28','simple':'#687248'}
STYLES={'fast':'-','slow':'--','common_level':':'}
FLAGS=('Q_nonpositive','PSD_guarded','PSD_undefined','peak_at_search_boundary',
       'global_peak_outside_search','ADC_rail_hit','pre_ADC_out_of_range')
FLAG_LABEL={'Q_nonpositive':'Q≤0','PSD_guarded':'PSD分母保护','PSD_undefined':'raw PSD未定义',
    'peak_at_search_boundary':'寻峰边界','global_peak_outside_search':'全局峰在搜索窗外',
    'ADC_rail_hit':'384点载体ADC碰轨','pre_ADC_out_of_range':'ADC前超范围'}
SPLITS=('TRAIN','DEV')
_ALLOWED_NPZ=set()


def require(condition,message):
    if not condition:raise ValueError(message)
def read(path):
    def unique(pairs):
        value={}
        for key,item in pairs:
            require(key not in value,'Duplicate JSON key: '+key);value[key]=item
        return value
    return json.loads(Path(path).read_text(encoding='utf8'),object_pairs_hook=unique)
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda:stream.read(1024**2),b''):digest.update(piece)
    return digest.hexdigest()
def load(path,keys=None):
    path=Path(path).resolve()
    require(path in _ALLOWED_NPZ,'Only hash-verified saved endpoint artifacts may be decoded: '+str(path))
    with np.load(path,allow_pickle=False) as archive:
        names=archive.files if keys is None else keys
        require(set(names)<=set(archive.files),'Saved artifact members missing: '+str(path))
        return {key:archive[key].copy() for key in names}
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,Path):return str(value)
    return value
def write(path,value):
    with path.open('x',encoding='utf8') as stream:stream.write(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def fmt(value):return '未定义' if value is None or not np.isfinite(float(value)) else f'{value:.4g}'
def foot(fig,value):fig.text(.065,.028,value,va='bottom',fontsize=10,color='#26394b',linespacing=1.4)
def title(fig,main,sub):
    fig.suptitle(main,y=.963,fontsize=17)
    fig.text(.5,.921,sub,ha='center',fontsize=10.5)


def _artifact_gate(directory,schema,status,source_hashes):
    directory=Path(directory).resolve();summary=read(directory/'SUMMARY.json')
    require(summary.get('schema')==schema and summary.get('status')==status
        and summary.get('scientific_status')=='response_match_not_established','Incomplete/wrong scientific artifact summary')
    require(not (directory/'FAILED.json').exists(),'A failed artifact attempt cannot enter the final report')
    inventory=summary.get('artifact_sha256')
    require(isinstance(inventory,dict) and bool(inventory),'Missing artifact hashes')
    require({p.name for p in directory.iterdir()}==set(inventory)|{'SUMMARY.json'},'Artifact inventory differs')
    for name,digest in inventory.items():
        path=directory/name
        require(Path(name).name==name and path.is_file() and not path.is_symlink() and sha(path)==digest,
            'Saved artifact changed: '+name)
        source_hashes[str(path.resolve())]=digest
        if path.suffix=='.npz':_ALLOWED_NPZ.add(path.resolve())
    source_hashes[str((directory/'SUMMARY.json').resolve())]=sha(directory/'SUMMARY.json')
    return summary


def _job_gate(job,summary,module,kind,remote,manifest_field,manifest_sha,status,source_hashes):
    paths={name:Path(job)/name for name in ('supervisor/exit_receipt.json','supervisor/launch.json',
        'EVALUATOR_NATIVE.json','JOB_COMPLETE.json')}
    values={name:read(path) for name,path in paths.items()}
    native,launch,core,complete=(values[k] for k in paths)
    require(type(native.get('exit_code')) is int and native['exit_code']==0 and native.get('reason') is None,
        'Clean evaluator supervisor native0 is required')
    require(type(core.get('exit_code')) is int and core['exit_code']==0 and core.get('module')==module and core.get('kind')==kind,
        'Core evaluator native0/module/kind is required')
    require(all(type(native.get(k)) is int and native[k]>0 and native[k]==launch.get(k) for k in ('supervisor_pid','child_pid'))
        and core.get('wrapper_pid')==native['child_pid'] and core.get('supervisor_pid')==native['supervisor_pid'],
        'Evaluator native PID roles differ')
    command=core.get('command',[])
    require(isinstance(command,list) and command[1:4]==['-B','-m',module]
        and command.count('--destination')==1 and command[command.index('--destination')+1]==remote+'/runs/'+
        ('diagnostic_jobs' if module.endswith('diagnose_window008_resume01') else 'evaluation_jobs')+'/'+kind+'/artifacts',
        'Evaluator command/module/output binding differs')
    require(native.get('started_utc')==launch.get('started_utc') and complete.get('status')==status
        and complete.get('kind')==kind and complete.get('optimizer_updates')==0
        and core.get(manifest_field)==complete.get(manifest_field)==manifest_sha
        and complete.get('evaluator_native_sha256')==sha(paths['EVALUATOR_NATIVE.json'])
        and complete.get('summary_sha256')==sha(Path(job)/'artifacts/SUMMARY.json'), 'Evaluator completion hashes differ')
    for path in paths.values():source_hashes[str(path.resolve())]=sha(path)
    return dict(native=native,core=core,complete=complete,paths={k:str(v.resolve()) for k,v in paths.items()})


def validate_inputs(run,evaluation,diagnostics):
    """JSON/byte gates precede every saved NPZ decode. No model is imported."""
    run,evaluation,diagnostics=map(lambda p:Path(p).resolve(),(run,evaluation,diagnostics))
    require(evaluation==run/'runs/evaluation_jobs/primary/artifacts'
        and diagnostics==run/'runs/diagnostic_jobs/primary/artifacts','Saved result roots differ from the selected run')
    source_hashes={};complete_path=run/'runs/science/TRAINING_COMPLETE.json';native_path=run/'runs/supervisor/exit_receipt.json'
    complete,native=read(complete_path),read(native_path)
    require(complete.get('schema')=='n01/window008-resume01-training-complete/v1'
        and complete.get('status')=='fixed_resumed_endpoints_complete_pending_native_exit_and_evaluation'
        and complete.get('effective_comparison_G_updates')==57000 and complete.get('effective_comparison_D_updates')==144000
        and complete.get('new_execution_G_updates')==32000 and complete.get('new_execution_D_updates')==87000
        and type(native.get('exit_code')) is int and native['exit_code']==0 and native.get('reason') is None,
        'Report requires the completed fixed resume01 training run')
    require([r.get('arm') for r in complete['arms']]==list(ARMS)
        and all(r.get('G_updates')==19000 and r.get('D_updates')=={h:16000 for h in HYP}
            and r['coverage']['source_unique']==6593 and r['coverage']['real_unique']==21047 for r in complete['arms']),
        'Report training endpoint budget/coverage differs')
    for row in complete['arms']:
        path=ROOT/row['endpoint_path'] if row['arm']=='deterministic' else run/row['endpoint_path']
        require(sha(path)==row['final_sha256'],'Report endpoint bytes changed: '+row['arm']);source_hashes[str(path.resolve())]=sha(path)
    for path in (complete_path,native_path):source_hashes[str(path.resolve())]=sha(path)
    summary=_artifact_gate(evaluation,'n01/window008-resume01-evaluation/v1','fixed_resumed_endpoint_evaluation_complete',source_hashes)
    replay_dir=evaluation.parent.parent/'replay01/artifacts'
    replay_summary=_artifact_gate(replay_dir,'n01/window008-resume01-evaluation/v1','fixed_resumed_endpoint_evaluation_complete',source_hashes)
    addon=_artifact_gate(diagnostics,'n01/window008-resume01-diagnostics/v1','registered_frozen_endpoint_diagnostics_complete',source_hashes)
    require(summary.get('fixed_training_G_updates')==57000 and summary.get('fixed_training_D_updates')==144000
        and summary.get('frozen_inputs_rechecked') is True and summary.get('DEV_numeric_decoded') is True
        and summary.get('endpoint_map')==complete['endpoint_map'] and summary.get('recovery')==complete['recovery'],
        'Main evaluation endpoint/freeze identity differs')
    require(all(summary.get(k)==0 for k in ('evaluation_G_updates','evaluation_D_updates','evaluation_noise_updates','population_updates','C2ST_updates'))
        and all(addon.get(k)==0 for k in ('optimizer_updates','G_updates','D_updates','noise_updates','population_updates','C2ST_updates')),
        'Report may consume only zero-optimization evaluation/diagnostics')
    require(addon.get('artificial_arrays_only') is False and addon.get('DEV_numeric_decoded') is True
        and addon.get('frozen_states_rechecked') is True and addon.get('physical_response_match_established') is False
        and addon.get('all_selected_parent_draw_rows_preserved') is True
        and (addon.get('repeat_selection_parent_count'),addon.get('repeat_draws_per_parent'),
             addon.get('intervention_parent_count'),addon.get('intervention_draws_per_parent'))==(128,64,32,8),
        'Scientific addon scope/freeze is incomplete')
    counts=read(diagnostics/'ACTUAL_COUNTERS.json')
    require(counts==addon['actual_forward_counts'] and counts['primary']==dict(signal_forward_rows=28416,observation_rows=85248,noise005_forward_rows=8704)
        and counts['total_signal_forward_rows']==29952 and counts['total_observation_rows']==89856
        and counts['total_noise005_forward_rows']==9216,'Addon actual row counters differ')
    main_freeze,addon_freeze=read(evaluation/'FROZEN_INPUTS.json'),read(diagnostics/'FROZEN_INPUTS.json')
    require(main_freeze['checkpoint_state_sha256']==addon_freeze['checkpoint_state_sha256']
        and main_freeze['noise_state_sha256']==addon_freeze['noise_state_sha256'], 'Main/addon frozen endpoint states differ')
    resume=read(ROOT/'window008_resume01_config.json');remote=resume['remote_root'].rstrip('/')
    eval_manifest=ROOT/'WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json'
    addon_manifest=ROOT/'WINDOW008_RESUME01_DIAGNOSTICS_SOURCE_MANIFEST.json'
    jobs={}
    for kind,result in (('primary',summary),('replay01',replay_summary)):
        jobs[kind]=_job_gate(evaluation.parent.parent/kind,result,'n01.evaluate_window008_resume01',kind,remote,
            'evaluation_source_manifest_sha256',sha(eval_manifest),'evaluation_job_complete',source_hashes)
    jobs['diagnostics']=_job_gate(diagnostics.parent,addon,'n01.diagnose_window008_resume01','primary',remote,
        'diagnostics_source_manifest_sha256',sha(addon_manifest),'diagnostics_job_complete',source_hashes)
    replay_path=evaluation.parent.parent/'replay01/REPLAY_VERIFICATION.json';replay=read(replay_path)
    require(replay.get('status')=='exact_replay_verified' and replay.get('mismatches')==[]
        and replay.get('reference_summary_sha256')==sha(evaluation/'SUMMARY.json')
        and replay.get('candidate_summary_sha256')==sha(replay_dir/'SUMMARY.json')
        and jobs['replay01']['complete']['replay_verification_sha256']==sha(replay_path), 'Exact replay/hash verification is incomplete')
    require(replay.get('NPZ_files_checked')==13 and replay.get('NPZ_members_checked',0)>0, 'Replay file/member evidence is missing')
    source_hashes[str(replay_path.resolve())]=sha(replay_path)
    for path in (eval_manifest,addon_manifest,ROOT/'window008_resume01_config.json',ROOT/'window008_resume01_diagnostics_policy.json'):
        source_hashes[str(path.resolve())]=sha(path)
    identity=read(diagnostics/'REPLAY_IDENTITY_ORDER.json')
    require(identity['rewritten_saved_ids']['status']==identity['reversed_parent_storage_order']['status']=='passed',
        'Addon identity/order engineering replay failed')
    for split,parents in (('TRAIN',6593),('DEV',512)):
        population=addon['population'][split]
        require(population['parents']==parents and population['rows']==parents*4 and population['response_forward_rows']==0,
            'Population sensitivity is not complete and zero-forward: '+split)
    return dict(complete=complete,native=native,summary=summary,addon=addon,replay=replay,counts=counts,
        jobs=jobs,source_hashes=source_hashes,identity_order=identity)


def validate_marker(path,pdfpath):
    value=read(path)
    require(type(value.get('exit_code')) is int and value['exit_code']==0
        and value.get('operation_kind')=='create' and value.get('output_format')=='pdf'
        and type(value.get('expected_output_count')) is int and value['expected_output_count']==1,
        'PDF authoring requires the parent-captured successful artifact marker')
    require(Path(value['outputpath']).resolve()==pdfpath.resolve(),'Artifact marker is bound to another PDF')
    return value


def finite_mean(value):
    value=np.asarray(value,float);defined=np.isfinite(value)
    return float(value[defined].mean()) if defined.any() else np.nan


def model_legend(fig,y=.890):
    handles=[Line2D([0],[0],color=COLORS[a],lw=1.7,label=LABEL[a]) for a in ARMS]
    handles.extend(Line2D([0],[0],color='#26394b',ls=STYLES[h],label=LABEL[h]) for h in HYP)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,y),ncol=6,frameon=False,fontsize=9.5)


def executive_findings(summary):
    distances=summary['distances'];comparisons=summary['full_vs_reference']
    higher=sum(distances[s]['full'][h]['components']['wave']>distances[s][a][h]['components']['wave']
        for s in SPLITS for h in HYP for a in ('deterministic','simple'))
    wave='TRAIN/DEV三种假设下，完整随机的波形距离均高于两个对照。' if higher==12 else f'12项角色×假设×对照比较中，完整随机的波形距离有{higher}项更高。'
    dev='DEV波形：full '+ ' / '.join(fmt(distances['DEV']['full'][h]['components']['wave']) for h in HYP)+ \
        '；确定性 '+ ' / '.join(fmt(distances['DEV']['deterministic'][h]['components']['wave']) for h in HYP)+'。'
    train_changes=[100*comparisons['TRAIN']['deterministic'][h]['QPSD']['relative_distance_reduction'] for h in HYP]
    dev_changes=[-100*comparisons['DEV']['deterministic'][h]['QPSD']['relative_distance_reduction'] for h in HYP]
    qpsd='QPSD相对确定性：TRAIN减少 '+ ' / '.join(f'{v:.3g}%' for v in train_changes)+ \
        '；DEV增加 '+ ' / '.join(f'{v:.3g}%' for v in dev_changes)+'。'
    return dict(wave_comparison=wave,DEV_wave_values=dev,QPSD_split_tradeoff=qpsd,
        hypothesis_order=list(HYP),wave_comparisons_full_higher=higher,wave_comparisons_total=12,
        values_source='SUMMARY.distances and full_vs_reference, per split/hypothesis/reference; no selection or refitting')


def load_training_journals(run):
    def rows(p):return [json.loads(line) for line in p.read_text(encoding='utf8').splitlines()]
    old=ROOT/'runs/window008_20261013/runs/science/train'
    fix=ROOT/'runs/window008_fix01_20261013/runs/science/train'
    new=run/'runs/science/train'
    full=[v for v in rows(old/'full/training.jsonl') if v['G_updates']<=3000]
    full.extend(v for v in rows(fix/'full/training.jsonl') if v['G_updates']<=6000)
    full.extend(rows(new/'full/training.jsonl'))
    result={'deterministic':rows(old/'deterministic/training.jsonl'),'full':full,'simple':rows(new/'simple/training.jsonl')}
    for arm,values in result.items():
        if [v['G_updates'] for v in values]!=list(range(1,19001)):raise ValueError('Effective training journal lineage differs: '+arm)
    return result


def evidence_page(pdf,save,evidence,coverage):
    complete,replay,counts=(evidence[k] for k in ('complete','replay','counts'))
    ended=dt.datetime.fromisoformat(evidence['native']['ended_utc']).astimezone(dt.timezone(dt.timedelta(hours=8)))
    fig=plt.figure(figsize=(12.4,8.8))
    findings=executive_findings(evidence['summary'])
    title(fig,'N01-008：固定终点评价与输入诊断','以下三值顺序为短相关 / 慢相关 / 公共分量；物理响应匹配尚未建立')
    ax=fig.add_axes([.07,.51,.86,.35]);ax.axis('off')
    lines=[
        ('主要结果',findings['wave_comparison']),
        ('DEV / 波形',findings['DEV_wave_values']),
        ('QPSD / 角色',findings['QPSD_split_tradeoff']),
        ('训练与覆盖','三臂各19,000 G / 每假设16,000 D；覆盖6,593 / 21,047父；有效57,000 / 144,000。'),
        ('执行与账本',f"历史实际{complete['historical_actual_G_updates']:,} G / {complete['historical_actual_D_updates']:,} D；resume01 native0，{ended:%m-%d %H:%M}（UTC+8）。"),
        ('数值证据',f"{replay['NPZ_files_checked']}个NPZ / {replay['NPZ_members_checked']}成员精确回放；诊断含重放{counts['total_signal_forward_rows']:,}信号行。"),
    ]
    for i,(label,value) in enumerate(lines):
        y=.98-i*.165
        ax.text(0,y,label,fontsize=11,fontweight='bold',va='top',color='#22718b')
        ax.text(.175,y,value,fontsize=10.5,va='top')
    table_ax=fig.add_axes([.07,.18,.86,.28]);table_ax.axis('off')
    rows=[[v['item'],v['report_status'],v['scope']] for v in coverage]
    table=table_ax.table(cellText=rows,colLabels=['原提案项目','实际状态','证据范围'],cellLoc='left',colLoc='left',
        colWidths=[.23,.18,.59],bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(9.5)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d9e4ec');cell.set_facecolor('#e7f0f6' if r==0 else ('#f6f9fb' if r%2==0 else 'white'))
        if r==0:cell.set_text_props(weight='bold')
    foot(fig,'原数值失败native1与用户停机native1保留；full从完整6,000步接续。观测256点、内部384点、原100点基线。\n单次已暴露采集；主体/诊断G/D/噪声/人口/C2ST更新均为0。独立C2ST、配对重复REAL和跨采集标定尚缺。')
    save(fig,'00_execution_and_replay',pdf)


def proposal_coverage():
    return [
        dict(item='波形、Q/PSD与分箱',report_status='主体完成',scope='TRAIN/DEV；三臂×三假设；父bootstrap1,000次'),
        dict(item='同输入条件随机变化',report_status='诊断完成',scope='128父×64次；Var(Q|x)、Var(PSD|x)、Cov及波形协方差'),
        dict(item='固定Q输入形状替换',report_status='诊断完成',scope='32父×8次同latent成对比较；实际干预强度保留'),
        dict(item='同形状Q节点趋势',report_status='诊断完成',scope='32父×8次×3个冻结TRAIN节点；qbar结构单列'),
        dict(item='冻结响应后人口变化',report_status='诊断完成',scope='TRAIN/DEV三套正权重；新增前向0；ESS与缺失质量'),
        dict(item='归一形状与时间量',report_status='定义子集已报',scope='Q>q_floor的归一曲线；rise/FWHM定义次数保留'),
        dict(item='独立C2ST',report_status='未执行',scope='更新0；独立域分类器的raw/条件AUC未获得'),
        dict(item='可复用物理响应',report_status='尚未建立',scope='缺少配对重复REAL、独立采集与能量/硬件标定'),
    ]


def collect_flags(generations,diagnostic_statistics):
    rows=[]
    def append(scope,split,arm,hyp,variant,pop):
        n=int(pop['Q']['rows']);flags={}
        for flag in FLAGS:
            alias={'peak_at_search_boundary':'peak_search_edge','global_peak_outside_search':'global_peak_outside_search'}
            entry=pop.get(flag,pop.get(alias.get(flag,'')))
            flags[flag]=dict(rows=n,true_rows=entry.get('true_rows') if isinstance(entry,dict) else None,
                saved=isinstance(entry,dict),meaning=FLAG_LABEL[flag])
        rows.append(dict(scope=scope,split=split,arm=arm,hypothesis=hyp,variant=variant,rows=n,flags=flags,
            raw_PSD_defined_rows=pop['PSD_raw'].get('numeric_defined_rows'),raw_PSD_undefined_rows=pop['PSD_raw'].get('undefined_rows'),
            rise10_90_undefined_rows=pop.get('rise10_90_ns',{}).get('undefined_rows'),
            FWHM_undefined_rows=pop.get('FWHM_ns',{}).get('undefined_rows'),
            normalized_waveform_undefined_rows=pop.get('normalized_waveform_defined',{}).get('false_rows')))
    for split in SPLITS:
        for arm in ARMS:
            for h in HYP:append('main',split,arm,h,'four_draw_population',generations[split]['summary'][arm][h]['population'])
        append('main',split,'REAL','existing_ADC','reference',generations[split]['summary']['REAL']['population'])
    for arm in ARMS:
        for h in HYP:
            append('repeat','DEV',arm,h,'64_draw',diagnostic_statistics['repeat'][arm][h]['population'])
            for family in ('shape','Q_trend'):
                for variant,pop in diagnostic_statistics[family][arm][h]['variants'].items():append(family,'DEV',arm,h,variant,pop)
    return rows


def load_addon_views(diagnostics):
    """Decode selected saved summary members, avoiding 720MiB raw/covariance copies."""
    result=dict(repeat={},shape={},trend={})
    repeat_keys=['source_parent_ids','Var_Q_given_x','Var_PSD_given_x','Cov_Q_PSD_given_x',
        'waveform256_covariance_mean_over_parents','scalar_fields','scalar_mean','scalar_sample_covariance',
        'scalar_defined_draws','scalar_covariance_defined_draws','normalized_waveform_defined_draws',
        'decomposition_variance_Q_identity_residual','decomposition_variance_tail_identity_residual']
    for arm in ARMS:
        for h in HYP:
            key=arm+'_'+h
            repeat=load(diagnostics/('REPEAT_'+key+'.npz'),repeat_keys)
            require(repeat['source_parent_ids'].shape==(128,) and repeat['Var_Q_given_x'].shape==(128,)
                and repeat['waveform256_covariance_mean_over_parents'].shape==(256,256), 'Repeat conditional artifact shape differs')
            result['repeat'][key]=repeat
            tag='donor_same_Q_minus_original'
            names=['source_parent_ids',tag+'_delta_waveform_mean']
            for metric in ('Q','tail','PSD','qsig','qbar','time_scale','rise10_90_ns','FWHM_ns'):
                names.extend(tag+'_delta_'+metric+'_'+kind for kind in ('mean','MonteCarlo_SE','defined_draws','undefined_draws'))
            shape=load(diagnostics/('SHAPE_'+key+'.npz'),names)
            require(shape['source_parent_ids'].shape==(32,) and shape[tag+'_delta_waveform_mean'].shape==(32,256),
                'Shape paired artifact dimensions differ');result['shape'][key]=shape
            names=['source_parent_ids']
            for node in ('Q25','Q50','Q75'):
                for metric in ('qbar','qsig','Q','tail','PSD','time_scale','rise10_90_ns','FWHM_ns'):
                    names.extend(node+'_'+metric+'_'+kind for kind in ('mean','MonteCarlo_SE','defined_draws'))
            for pair in ('Q50_minus_Q25','Q75_minus_Q50'):
                names.extend(pair+'_delta_'+metric+'_'+kind for metric in ('qbar','Q','PSD') for kind in ('mean','MonteCarlo_SE','defined_draws'))
            trend=load(diagnostics/('Q_TREND_'+key+'.npz'),names)
            require(trend['source_parent_ids'].shape==(32,), 'Q-node parent count differs');result['trend'][key]=trend
    result['selection']=load(diagnostics/'SELECTIONS.npz',['repeat_selected_ids','selected_ids','donor_ids','SOURCE_Q',
        'donor_SOURCE_Q','abs_log_charge_gap','unit_shape_l2_difference','unit_shape_absmax_difference','Q_nodes','Q_node_scale'])
    require(result['selection']['repeat_selected_ids'].shape==(128,) and result['selection']['selected_ids'].shape==(32,)
        and result['selection']['Q_nodes'].shape==(3,), 'Saved intervention selections differ')
    result['common']=load(diagnostics/'REPEAT_COMMON.npz',['baseline_RMS_V','baseline_floor_RMS_V',
        'baseline_ACF_lag0_99','baseline_power_fraction5','baseline_variance_floor_used','S'])
    return result


def _small_stats(values):
    values=np.asarray(values,float);valid=np.isfinite(values);selected=values[valid]
    return dict(rows=int(values.size),defined_rows=int(valid.sum()),undefined_rows=int((~valid).sum()),
        mean=float(selected.mean()) if len(selected) else None,
        minimum=float(selected.min()) if len(selected) else None,maximum=float(selected.max()) if len(selected) else None,
        quantile_levels=[.05,.5,.95],quantiles=np.quantile(selected,[.05,.5,.95]).tolist() if len(selected) else None)


def conditional_page(pdf,save,views):
    fig,axes=plt.subplots(2,3,figsize=(12.4,8.8));fig.subplots_adjust(left=.075,right=.97,bottom=.15,top=.83,hspace=.46,wspace=.32)
    title(fig,'同一个输入的条件随机变化：128父事件×64次','上排显示父事件均值与5%–95%分位范围；下排为平均条件协方差对角线的平方根，ddof=1。')
    metrics=[('Var_Q_given_x','Var(Q|x)','(V ns)²'),('Var_PSD_given_x','Var(PSD|x)','无量纲²'),('Cov_Q_PSD_given_x','Cov(Q,PSD|x)','V ns')]
    for j,(field,name,unit) in enumerate(metrics):
        ax=axes[0,j]
        for k,h in enumerate(HYP):
            data=[np.asarray(views['repeat'][a+'_'+h][field],float) for a in ARMS]
            means=np.asarray([finite_mean(v) for v in data]);bounds=np.asarray([np.quantile(v,[.05,.95]) for v in data])
            x=np.arange(3)+(k-1)*.20
            ax.vlines(x,bounds[:,0],bounds[:,1],color=COLORS[h],lw=1.8)
            ax.scatter(x,means,color=COLORS[h],s=23,label=LABEL[h],zorder=3)
        ax.axhline(0,color='#84929c',lw=.7);ax.set_xticks(np.arange(3),['确定性','完整随机','简单随机'])
        ax.set_title(name);ax.set_ylabel(unit);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    for j,arm in enumerate(ARMS):
        ax=axes[1,j]
        for h in HYP:
            cov=views['repeat'][arm+'_'+h]['waveform256_covariance_mean_over_parents']
            require(np.isfinite(cov).all() and (np.diag(cov)>=0).all(),'Invalid saved mean conditional covariance')
            ax.plot(np.arange(256)-16,1000*np.sqrt(np.diag(cov)),color=COLORS[h],label=LABEL[h])
        ax.set_title(LABEL[arm]+' / 条件逐点RMS宽度');ax.set_xlabel('相对选中峰时间 / ns');ax.set_ylabel('√平均条件Var / mV')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.897),ncol=3,frameon=False)
    foot(fig,'固定选择128个DEV SOURCE父事件；三臂和三假设共享显式随机量。Q与guarded PSD保留全部64次。\n完整256×256协方差、同窗signal/noise/ADC分量协方差及定义次数保存在原NPZ和报告证据JSON。')
    save(fig,'06_repeat_conditional_variance_covariance',pdf)


def paired_shape_page(pdf,save,views):
    fig,axes=plt.subplots(2,3,figsize=(12.4,8.8));fig.subplots_adjust(left=.075,right=.97,bottom=.15,top=.83,hspace=.45,wspace=.32)
    strength=views['selection']['unit_shape_l2_difference'];tag='donor_same_Q_minus_original'
    title(fig,'固定Q、固定随机量的输入形状替换：32父×8次',
        '实际输入归一形状L2差：中位数 '+fmt(np.median(strength))+'，最大值 '+fmt(np.max(strength))+' / ns⁻¹')
    for j,(metric,name,unit) in enumerate((('PSD','成对ΔPSD','无量纲'),('rise10_90_ns','成对Δrise10–90','ns'),('FWHM_ns','成对ΔFWHM','ns'))):
        ax=axes[0,j]
        for k,h in enumerate(HYP):
            for i,arm in enumerate(ARMS):
                values=views['shape'][arm+'_'+h][tag+'_delta_'+metric+'_mean'];valid=np.isfinite(values)
                if valid.any():
                    low,high=np.quantile(values[valid],[.05,.95]);x=i+(k-1)*.2
                    ax.vlines(x,low,high,color=COLORS[h],lw=1.7);ax.scatter(x,finite_mean(values),color=COLORS[h],s=23,zorder=3)
        ax.axhline(0,color='#84929c',lw=.8);ax.set_xticks(np.arange(3),['确定性','完整随机','简单随机'])
        ax.set_title(name);ax.set_ylabel(unit);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    for j,arm in enumerate(ARMS):
        ax=axes[1,j]
        for h in HYP:
            delta=views['shape'][arm+'_'+h][tag+'_delta_waveform_mean']
            ax.plot(np.arange(256)-16,1000*delta.mean(0),color=COLORS[h],label=LABEL[h])
        ax.axhline(0,color='#84929c',lw=.8);ax.set_title(LABEL[arm]+' / 均值波形成对差')
        ax.set_xlabel('相对选中峰时间 / ns');ax.set_ylabel('donor−original / mV')
    handles,labels=axes[1,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.897),ncol=3,frameon=False)
    foot(fig,'每个donor按原SOURCE电荷选取，缩放后canonical Q相同；两种形状逐draw共用8组latent。\n竖线为32父的5%–95%范围；Monte Carlo SE、共同定义次数与实际Q/形状差详列证据表，近零干预照常保留。')
    save(fig,'07_paired_shape_intervention',pdf)


def trend_page(pdf,save,views,diagnostic_statistics):
    fig,axes=plt.subplots(2,3,figsize=(12.4,8.8));fig.subplots_adjust(left=.075,right=.97,bottom=.15,top=.815,hspace=.46,wspace=.33)
    nodes=views['selection']['Q_nodes'];names=('Q25','Q50','Q75')
    title(fig,'保持输入形状的Q节点趋势：32父×8次×3节点','节点直接来自冻结SOURCE TRAIN边界索引2/4/6；不同节点共享同一份32×8显式latent。')
    metrics=(('qbar','条件均值参数 qbar','V ns'),('qsig','随机signal目标 qsig','V ns'),('Q','最终观测 Q','V ns'),
             ('PSD','最终guarded PSD','无量纲'),('rise10_90_ns','rise10–90','ns'),('FWHM_ns','FWHM','ns'))
    for ax,(metric,name,unit) in zip(axes.flat,metrics):
        for arm in ARMS:
            for h in HYP:
                entry=views['trend'][arm+'_'+h]
                y=[finite_mean(entry[node+'_'+metric+'_mean']) for node in names]
                ax.plot(nodes,y,marker='o',ms=3.5,color=COLORS[arm],ls=STYLES[h],lw=1.3)
        ax.set_xticks(nodes,[fmt(v) for v in nodes]);ax.set_xlabel('SOURCE canonical Q / V ns');ax.set_ylabel(unit);ax.set_title(name)
    model_legend(fig,y=.883)
    negative=sum(diagnostic_statistics['Q_trend'][arm][h]['paired_comparisons'][pair]['qbar_structure']['negative_draws']
        for arm in ARMS for h in HYP for pair in ('Q50_minus_Q25','Q75_minus_Q50'))
    foot(fig,f'qbar相邻负差的实际记录：{negative:,}个臂×假设×父×draw比较；结构记录与带噪声观测Q分开。\n曲线为32父的条件均值概括；时间量按定义子集统计，全部缺失次数、成对差和Monte Carlo SE保留。')
    save(fig,'08_same_shape_Q_trend',pdf)


def population_page(pdf,save,populations):
    fig,axes=plt.subplots(2,3,figsize=(12.4,8.8));fig.subplots_adjust(left=.075,right=.97,bottom=.15,top=.815,hspace=.47,wspace=.33)
    title(fig,'冻结响应下的人口倾斜：TRAIN与DEV分别统计','由固定SOURCE Q8箱定义uniform、tilt+、tilt−正权重；全部父事件保留，新增信号/噪声前向为0。')
    schemes=('uniform','tilt_plus','tilt_minus');scheme_labels=('uniform','tilt+','tilt−');scheme_colors=('#26394b','#cc7626','#007f8b')
    for i,split in enumerate(SPLITS):
        reports=populations[split]['schemes'];ax=axes[i,0]
        for scheme,color in zip(schemes,scheme_colors):
            report=reports[scheme]
            ax.plot(np.arange(1,9),report['bin_parent_weight_fraction'],'o-',ms=3.3,color=color,
                label=scheme.replace('tilt_plus','tilt+').replace('tilt_minus','tilt−')+' ESS='+fmt(report['parent_ESS']))
        ax.set_title(split+' / 父权重质量');ax.set_xlabel('固定SOURCE Q8箱');ax.set_ylabel('父权重份额');ax.set_xticks(np.arange(1,9));ax.legend(fontsize=8.7)
        for j,(metric,kind,name,unit) in enumerate((('PSD','std','人口PSD SD','无量纲'),('Q','mean','人口Q均值','V ns')),start=1):
            ax=axes[i,j]
            for arm in ARMS:
                for h in HYP:
                    y=[reports[s]['arms'][arm][h]['population'][metric][kind][0] for s in schemes]
                    ax.plot(np.arange(3),y,marker='o',ms=3.5,color=COLORS[arm],ls=STYLES[h],lw=1.3)
            ax.set_xticks(np.arange(3),scheme_labels);ax.set_ylabel(unit);ax.set_title(split+' / '+name)
    model_legend(fig,y=.883)
    foot(fig,'每父四draw同权重，父权重和等于原父数；Q/PSD采用ddof0人口矩，REAL保持原权重。\n全局波形矩由固定SOURCE-bin矩精确合成；output-Q条件加权波形矩未提供，主产物缺少联合分箱波形矩。')
    save(fig,'09_population_weight_sensitivity',pdf)


def html_table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='runs/window008_resume01_20261013')
    parser.add_argument('--evaluation',default='runs/window008_resume01_20261013/runs/evaluation_jobs/primary/artifacts')
    parser.add_argument('--diagnostics',default='runs/window008_resume01_20261013/runs/diagnostic_jobs/primary/artifacts')
    parser.add_argument('--out',default='reports/window008_resume01_20261013')
    parser.add_argument('--marker-receipt',required=True,help='Parent-captured native0 PDF operation marker, bound to the final PDF path')
    args=parser.parse_args();run=(ROOT/args.run).resolve();evaluation=(ROOT/args.evaluation).resolve()
    diagnostics=(ROOT/args.diagnostics).resolve();out=(ROOT/args.out).resolve()
    require(not (os.name=='nt' and out.drive.upper() not in ('E:','D:')),'All report outputs must stay on E/D')
    if out.exists():raise FileExistsError('Preserve an earlier report: '+str(out))
    pdfpath=out/'N01_WINDOW008_RESUME01_REPORT.pdf'
    marker_path=(ROOT/args.marker_receipt).resolve();marker=validate_marker(marker_path,pdfpath)
    evidence=validate_inputs(run,evaluation,diagnostics)
    complete,summary,replay=(evidence[k] for k in ('complete','summary','replay'))
    distributions=read(evaluation/'DISTRIBUTION_DISTANCES.json')['distances']
    generations={split:read(evaluation/(split+'_generation.json')) for split in SPLITS}
    bootstrap={split:read(evaluation/(split+'_PARENT_BOOTSTRAP.json')) for split in SPLITS}
    curves={split:load(evaluation/(split+'_curves.npz')) for split in SPLITS}
    diagnostic_statistics=read(diagnostics/'DIAGNOSTIC_STATISTICS.json')
    diagnostic_generation=read(diagnostics/'GENERATION.json')
    populations={split:read(diagnostics/('POPULATION_'+split+'.json')) for split in SPLITS}
    views=load_addon_views(diagnostics)
    flags=collect_flags(generations,diagnostic_statistics);coverage=proposal_coverage()
    journals=load_training_journals(run)
    if any(len(rows)!=19000 or rows[-1]['G_updates']!=19000 for rows in journals.values()):raise ValueError('Training logs are incomplete')
    for path in (ROOT/'runs/window008_20261013/runs/science/train/deterministic/training.jsonl',
        ROOT/'runs/window008_20261013/runs/science/train/full/training.jsonl',
        ROOT/'runs/window008_fix01_20261013/runs/science/train/full/training.jsonl',
        run/'runs/science/train/full/training.jsonl',run/'runs/science/train/simple/training.jsonl'):
        evidence['source_hashes'][str(path.resolve())]=sha(path)
    out.mkdir(parents=True);(out/'figures').mkdir()
    figures=[]
    def save(fig,name,pdf):
        fig.text(.968,.027,str(len(figures)+1)+' / 10',ha='right',fontsize=9.5,color='#26394b')
        fig.savefig(out/'figures'/(name+'.png'),dpi=175,facecolor='white')
        pdf.savefig(fig,facecolor='white');plt.close(fig);figures.append(name)
    with PdfPages(pdfpath) as pdf:
        evidence_page(pdf,save,evidence,coverage)
        fig,axes=plt.subplots(2,3,figsize=(12.4,8.8));fig.subplots_adjust(left=.075,right=.97,bottom=.14,top=.825,hspace=.46,wspace=.32)
        title(fig,'256点固定终点的匹配分布距离','三臂×三种噪声假设全部保留；TRAIN与DEV分别报告。')
        for i,split in enumerate(('TRAIN','DEV')):
            for j,metric in enumerate(METRICS):
                ax=axes[i,j];positions=np.arange(3)
                for k,h in enumerate(HYP):
                    ax.bar(positions+(k-1)*.24,[distributions[split][a][h]['components'][metric] for a in ARMS],width=.22,color=COLORS[h],label=LABEL[h])
                ax.axhline(distributions[split]['REAL_REAL']['components'][metric],color=COLORS['REAL'],lw=1.5,ls='--',label='REAL-REAL')
                ax.set_xticks(positions,['确定性','完整随机','简单随机']);ax.set_ylabel('固定投影距离（越小越接近）')
                ax.set_title(split+' / '+LABEL[metric])
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.902),ncol=4,frameon=False)
        foot(fig,'TRAIN每域匹配2,048个父事件，DEV每域256个；每个生成父事件只取第0次响应。距离为标准化固定投影口径。\nREAL A/B按原软件组平衡且不交叠；每个假设使用同一冻结投影。表中保留各假设原值与full的成对差。')
        save(fig,'01_fixed_endpoint_distances',pdf)

        fig,axes=plt.subplots(2,2,figsize=(12.4,8.8));fig.subplots_adjust(left=.08,right=.97,bottom=.15,top=.815,hspace=.44,wspace=.3)
        title(fig,'TRAIN与DEV：全人口256点波形矩','每父4次响应；颜色表示模型，线型表示噪声假设，黑色点划线为REAL。')
        times=np.arange(256)-16
        for j,split in enumerate(SPLITS):
            for i,stat in enumerate(('mean','std')):
                ax=axes[i,j];values=curves[split]
                for arm in ARMS:
                    for h in HYP:ax.plot(times,1000*values[arm+'_'+h+'_'+stat],color=COLORS[arm],ls=STYLES[h],lw=1.25)
                ax.plot(times,1000*values['REAL_'+stat],color='#17232d',lw=1.7,ls='-.')
                ax.set_title(split+(' / 人口均值' if stat=='mean' else ' / 人口逐点SD'))
                ax.set_xlabel('相对选中峰时间 / ns');ax.set_ylabel('电压 / mV' if stat=='mean' else '人口逐点SD / mV')
                ax.axvline(0,color='#84929c',lw=.8,ls=':')
        model_legend(fig,y=.883)
        foot(fig,'TRAIN生成26,372行/臂/假设、REAL21,047行；DEV生成2,048行/臂/假设、REAL599行。\n人口逐点SD同时包含输入父事件差异和随机响应；条件Var另在128×64重复页呈现。')
        save(fig,'02_population_waveforms_TRAIN_DEV',pdf)

        fig,axes=plt.subplots(2,3,figsize=(12.4,8.8));fig.subplots_adjust(left=.075,right=.975,bottom=.14,top=.86,hspace=.45,wspace=.3)
        title(fig,'电荷条件下的PSD宽度','各生成臂和REAL使用相同的输出Q定义与TRAIN分箱边界；箱内父事件人口仍可能不同。')
        for i,split in enumerate(('TRAIN','DEV')):
            for j,arm in enumerate(ARMS):
                ax=axes[i,j]
                for h in HYP:
                    rows=generations[split]['summary'][arm][h]['output_Q_in_REAL_TRAIN_bins']
                    vals=[row['PSD'].get('std',np.nan) for row in rows]
                    ax.plot(np.arange(1,9),vals,'o-',ms=3.5,color=COLORS[h],label=LABEL[h])
                rows=generations[split]['summary']['REAL']['output_Q_in_REAL_TRAIN_bins']
                ax.plot(np.arange(1,9),[row['PSD'].get('std',np.nan) for row in rows],'s--',ms=3.5,color=COLORS['REAL'],label='REAL')
                ax.set_xticks(np.arange(1,9));ax.set_xlabel('输出Q箱（低 → 高）');ax.set_ylabel('PSD SD');ax.set_title(split+' / '+LABEL[arm])
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.900),ncol=4,frameon=False)
        foot(fig,'输出Q使用固定REAL TRAIN边界。每箱保留父事件数、行数、支持标记、PSD均值/SD和raw PSD缺失数。\nPSD为带符号分母保护口径；所有负值和guard行保留。完整条件均值、宽度和迁移矩阵见HTML/证据JSON。')
        save(fig,'03_output_charge_conditional_psd',pdf)

        fig,axes=plt.subplots(2,2,figsize=(12.4,8.8));fig.subplots_adjust(left=.08,right=.97,bottom=.14,top=.83,hspace=.46,wspace=.29)
        title(fig,'相同SOURCE父人口：随机响应的PSD展宽','固定SOURCE电荷箱；完整随机臂 / 对照臂。带状区域为1,000次父事件重采样的95%百分位区间。')
        unsupported={}
        for i,split in enumerate(('TRAIN','DEV')):
            unsupported[split]={}
            for j,reference in enumerate(('deterministic','simple')):
                ax=axes[i,j];entry=bootstrap[split]['comparisons'][reference]['std_full_over_reference']
                point=np.asarray(entry['point'],float)[1,:,:,1];ci=np.asarray(entry['percentile95'],float)[:,1,:,:,1]
                support=np.asarray(bootstrap[split]['comparison_supported'][reference],bool)[1]
                require(point.shape==(3,8) and ci.shape==(2,3,8) and support.shape==(3,8),'Parent bootstrap interval axes differ')
                unsupported[split][reference]=int((~support).sum())
                for k,h in enumerate(HYP):
                    ax.plot(np.arange(1,9),point[k],'o-',ms=3.5,color=COLORS[h],label=LABEL[h])
                    ax.fill_between(np.arange(1,9),ci[0,k],ci[1,k],where=support[k],color=COLORS[h],alpha=.13)
                    if (~support[k]).any():ax.scatter(np.arange(1,9)[~support[k]],point[k,~support[k]],marker='x',s=45,color=COLORS[h])
                ax.axhline(1,color='#7f8a95',ls=':',lw=1);ax.set_xticks(np.arange(1,9));ax.set_xlabel('固定SOURCE Q箱');ax.set_ylabel('PSD SD比值')
                ax.set_title(split+' / 完整随机 ÷ '+LABEL[reference])
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.9),ncol=3,frameon=False)
        foot(fig,'父事件的4次响应共同重采样；三臂与三假设共享父权重，REAL固定。至少32父才绘制区间，低支持点用叉号。\n带为1,000次父bootstrap的95%百分位区间；每格定义replicate数和零参考宽度对应的未定义状态均保留。')
        save(fig,'04_paired_parent_width_comparisons',pdf)

        fig,axes=plt.subplots(2,2,figsize=(12.4,8.8));fig.subplots_adjust(left=.08,right=.97,bottom=.15,top=.86,hspace=.44,wspace=.3)
        title(fig,'优化记录与独立100点基线','3,000均值 + 12,000联合 + 4,000低学习率；电子学噪声005与人口权重始终冻结。')
        x=np.arange(50,19001,50)
        for arm in ARMS:
            rows=journals[arm]
            axes[0,0].plot(x,np.array([row['weighted_total'] for row in rows]).reshape(-1,50).mean(1),label=LABEL[arm],color=COLORS[arm])
            axes[0,1].plot(x,np.array([np.mean([row['hypotheses'][h]['components']['QPSD'] for h in HYP]) for row in rows]).reshape(-1,50).mean(1),label=LABEL[arm],color=COLORS[arm])
        for ax in axes[0]:
            ax.axvline(3000,color='#7f8a95',ls=':',lw=1);ax.axvline(15000,color='#7f8a95',ls=':',lw=1)
            ax.set_xlabel('每臂G更新');ax.legend(fontsize=8)
        axes[0,0].set_title('加权训练目标 / 50步均值');axes[0,0].set_ylabel('损失（跨阶段含义不同）')
        axes[0,1].set_title('三种假设的QPSD项均值');axes[0,1].set_ylabel('QPSD损失')
        for j,split in enumerate(('TRAIN','DEV')):
            ax=axes[1,j];c=curves[split]
            ax.plot(np.arange(100),1000*c['generated_baseline100_std'],label='生成 / 三臂共享',color=COLORS['full'])
            ax.plot(np.arange(100),1000*c['REAL_baseline100_std'],label='REAL',color=COLORS['REAL'],ls='--')
            ax.set_title(split+' / 实际基线段逐点SD');ax.set_xlabel('内部前100点基线样本');ax.set_ylabel('基线SD / mV');ax.legend()
        foot(fig,'训练曲线按有效19,000步串接：原full1–3,000、fix01的3,001–6,000、resume01的6,001–19,000。\n基线为载体原100点ADC；005冻结且三臂/假设共享。RMS、ACF0–99、频带和S的诊断统计见证据表。')
        save(fig,'05_training_and_baseline',pdf)
        conditional_page(pdf,save,views)
        paired_shape_page(pdf,save,views)
        trend_page(pdf,save,views,diagnostic_statistics)
        population_page(pdf,save,populations)

    require(len(figures)==10,'The documented report page count differs')
    # A final byte recheck protects the figures/tables against concurrent result changes.
    for path,digest in evidence['source_hashes'].items():require(sha(Path(path))==digest,'Report input changed during plotting: '+path)
    repeat_detail={}
    for key,entry in views['repeat'].items():
        fields=[str(v) for v in entry['scalar_fields']]
        repeat_detail[key]=dict(source_parent_ids=entry['source_parent_ids'],Var_Q_given_x=entry['Var_Q_given_x'],
            Var_PSD_given_x=entry['Var_PSD_given_x'],Cov_Q_PSD_given_x=entry['Cov_Q_PSD_given_x'],
            scalar_fields=fields,scalar_mean=entry['scalar_mean'],scalar_sample_covariance=entry['scalar_sample_covariance'],
            scalar_defined_draws=entry['scalar_defined_draws'],scalar_covariance_defined_draws=entry['scalar_covariance_defined_draws'],
            normalized_waveform_defined_draws=entry['normalized_waveform_defined_draws'],
            full_waveform_covariance_shape=[128,256,256],full_waveform_covariance_source=str(diagnostics/('REPEAT_'+key+'.npz')),
            decomposition_variance_Q_identity_residual=entry['decomposition_variance_Q_identity_residual'],
            decomposition_variance_tail_identity_residual=entry['decomposition_variance_tail_identity_residual'])
    common=views['common']
    baseline=dict(RMS=_small_stats(common['baseline_RMS_V']),floor_RMS=_small_stats(common['baseline_floor_RMS_V']),
        S=_small_stats(common['S']),variance_floor_rows=int(common['baseline_variance_floor_used'].sum()),
        rows=int(common['S'].size),ACF_mean_lag0_99=common['baseline_ACF_lag0_99'].mean((0,1)),
        power_fraction5_mean=common['baseline_power_fraction5'].mean((0,1)),
        FFT_bins=[[1,4],[4,9],[9,17],[17,33],[33,51]],original_baseline100=True)
    source_evidence=dict(schema='n01/window008-resume01-report-evidence/v1',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        units=dict(waveform='V',charge='V ns',PSD='dimensionless',time='ns',Var_Q='(V ns)^2',Cov_Q_PSD='V ns'),
        row_retention='All saved parent/draw rows retained; undefined numerical entries remain null with explicit counts.',
        source_files_sha256=evidence['source_hashes'],source_role='Only saved completed endpoint/diagnostic artifacts and metadata journals',
        training=complete,main_summary=summary,main_distances=distributions,executive_findings=executive_findings(summary),
        main_full_vs_reference=summary['full_vs_reference'],main_parent_bootstrap=bootstrap,
        main_generation_summary={s:dict(parents=generations[s]['parents'],rows=generations[s]['rows'],
            rows_removed=generations[s]['rows_removed'],summary=generations[s]['summary'],engineering=generations[s]['engineering']) for s in SPLITS},
        diagnostic_summary=evidence['addon'],diagnostic_statistics=diagnostic_statistics,actual_counters=evidence['counts'],
        diagnostic_generation=diagnostic_generation,main_exact_replay=replay,identity_order_replay=evidence['identity_order'],
        proposal_coverage=coverage,flags_and_definition_counts=flags,repeat_conditional_by_parent=repeat_detail,
        paired_shape_by_parent=views['shape'],same_shape_Q_trend_by_parent=views['trend'],
        intervention_selection=views['selection'],population=populations,baseline100=baseline,
        bootstrap_interval_axes=['lower_upper','scheme','hypothesis','Q_bin','metric'],
        C2ST=dict(status='unexecuted',updates=0,AUC=None,reason='No independent C2ST job was registered or executed.'),
        new_model_calls=0,new_training_updates=0,new_original_waveform_decodes=0,
        scientific_status='response_match_not_established',PDF_visual_QA='pending_all_page_render_and_inspection')
    write(out/'EVIDENCE.json',source_evidence)
    headline=dict(schema='n01/window008-resume01-report/v1',UTC=source_evidence['UTC'],status='report_built_pending_visual_QA',
        training_completed=True,training_native_exit_code=evidence['native']['exit_code'],main_evaluation_completed=True,
        exact_numerical_replay_verified=True,diagnostics_completed=True,physical_response_match_established=False,
        G_updates=57000,D_updates=144000,historical_actual_G_updates=complete['historical_actual_G_updates'],
        historical_actual_D_updates=complete['historical_actual_D_updates'],new_execution_G_updates=complete['new_execution_G_updates'],
        new_execution_D_updates=complete['new_execution_D_updates'],exact_replay=replay['status'],
        NPZ_files_checked=replay['NPZ_files_checked'],NPZ_members_checked=replay['NPZ_members_checked'],
        C2ST_status='unexecuted',C2ST_updates=0,C2ST_AUC=None,noise_updates=0,population_updates=0,
        effective_samples=256,internal_carrier_samples=384,report_pages=len(figures),scientific_status='response_match_not_established',
        per_hypothesis_distances=distributions,full_vs_reference=summary['full_vs_reference'],executive_findings=executive_findings(summary),
        unsupported_width_comparison_cells=unsupported,actual_diagnostic_counts=evidence['counts'],
        proposal_coverage=coverage,new_model_calls=0,new_training_updates=0,new_original_waveform_decodes=0,
        independent_acquisition_available=False,paired_event_response_truth_available=False,DEV_previously_exposed=True,
        evidence_json_sha256=sha(out/'EVIDENCE.json'),evaluation_summary_sha256=sha(evaluation/'SUMMARY.json'),
        diagnostics_summary_sha256=sha(diagnostics/'SUMMARY.json'),training_complete_sha256=sha(run/'runs/science/TRAINING_COMPLETE.json'))
    write(out/'SUMMARY.json',headline)

    distance_rows=[[split,LABEL[arm],LABEL[h]]+[fmt(distributions[split][arm][h]['components'][metric]) for metric in METRICS]
        for split in SPLITS for arm in ARMS for h in HYP]
    distance_rows.extend([[split,'REAL-REAL','原分组内A/B']+[fmt(distributions[split]['REAL_REAL']['components'][metric]) for metric in METRICS] for split in SPLITS])
    relative_rows=[]
    for split in SPLITS:
        for reference in ('deterministic','simple'):
            for h in HYP:
                for metric in METRICS:
                    entry=summary['full_vs_reference'][split][reference][h][metric]
                    fraction=entry['relative_distance_reduction']
                    relative_rows.append([split,LABEL[reference],LABEL[h],LABEL[metric],fmt(entry['full_minus_reference']),
                        '未定义' if fraction is None else f'{100*fraction:.3g}%'])
    conditional_rows=[]
    bootstrap_rows=[]
    for split in SPLITS:
        for si,scheme in enumerate(('output_Q_in_REAL_TRAIN_bins','fixed_SOURCE_Q_in_SOURCE_TRAIN_bins')):
            scheme_label='输出Q / REAL TRAIN箱' if si==0 else 'SOURCE Q / SOURCE TRAIN箱'
            for arm in ARMS:
                for h in HYP:
                    for row in generations[split]['summary'][arm][h][scheme]:
                        conditional_rows.append([split,scheme_label,LABEL[arm],LABEL[h],row['bin']+1,row['parents'],row['rows'],
                            '支持' if row['parents']>=32 else '低支持',fmt(row['Q'].get('mean')),fmt(row['Q'].get('std')),
                            fmt(row['PSD'].get('mean')),fmt(row['PSD'].get('std')),row['raw_PSD'].get('undefined_rows'),
                            row['guarded_PSD_rows'],row['nonpositive_Q_rows']])
            for reference in ('deterministic','simple'):
                support=np.asarray(bootstrap[split]['comparison_supported'][reference],bool)
                for hi,h in enumerate(HYP):
                    for b in range(8):
                        for mi,metric in enumerate(('Q','PSD')):
                            record=[split,scheme_label,LABEL[reference],LABEL[h],b+1,metric,
                                '支持' if support[si,hi,b] else '低支持']
                            for name in ('mean_full_minus_reference','std_full_minus_reference','std_full_over_reference'):
                                item=bootstrap[split]['comparisons'][reference][name]
                                point=np.asarray(item['point'],float);interval=np.asarray(item['percentile95'],float)
                                defined=np.asarray(item['defined_replicates'])
                                record.extend([fmt(point[si,hi,b,mi]),fmt(interval[0,si,hi,b,mi]),fmt(interval[1,si,hi,b,mi]),
                                    int(defined[si,hi,b,mi])])
                            bootstrap_rows.append(record)
    flag_rows=[]
    for entry in flags:
        for key,count in entry['flags'].items():
            flag_rows.append([entry['scope'],entry['split'],LABEL.get(entry['arm'],entry['arm']),
                LABEL.get(entry['hypothesis'],entry['hypothesis']),entry['variant'],FLAG_LABEL[key],
                '未保存' if count['true_rows'] is None else count['true_rows'],count['rows'],
                '未保存' if count['true_rows'] is None else f"{100*count['true_rows']/count['rows']:.4g}%"])
        for key,label in (('raw_PSD_undefined_rows','raw PSD数值未定义'),('rise10_90_undefined_rows','rise10-90未定义'),
                           ('FWHM_undefined_rows','FWHM未定义'),('normalized_waveform_undefined_rows','归一波形未定义')):
            if entry[key] is not None:flag_rows.append([entry['scope'],entry['split'],LABEL.get(entry['arm'],entry['arm']),
                LABEL.get(entry['hypothesis'],entry['hypothesis']),entry['variant'],label,entry[key],entry['rows'],f"{100*entry[key]/entry['rows']:.4g}%"])
    repeat_rows=[];shape_rows=[];trend_rows=[];population_rows=[];weight_rows=[]
    for arm in ARMS:
        for h in HYP:
            key=arm+'_'+h;entry=views['repeat'][key]
            for field,label,unit in (('Var_Q_given_x','Var(Q|x)','(V ns)^2'),('Var_PSD_given_x','Var(PSD|x)','无量纲^2'),('Cov_Q_PSD_given_x','Cov(Q,PSD|x)','V ns')):
                stats=_small_stats(entry[field]);repeat_rows.append([LABEL[arm],LABEL[h],label,unit,fmt(stats['mean']),
                    fmt(stats['quantiles'][0]),fmt(stats['quantiles'][1]),fmt(stats['quantiles'][2]),stats['defined_rows'],stats['undefined_rows']])
            for metric in ('Q','tail','PSD','qsig','qbar','time_scale','rise10_90_ns','FWHM_ns'):
                tag='donor_same_Q_minus_original_delta_'+metric
                values=views['shape'][key][tag+'_mean'];counts=views['shape'][key][tag+'_defined_draws']
                shape_rows.append([LABEL[arm],LABEL[h],metric,fmt(finite_mean(values)),
                    fmt(finite_mean(views['shape'][key][tag+'_MonteCarlo_SE'])),int(np.isfinite(values).sum()),32,
                    int(counts.sum()),int(32*8-counts.sum())])
                for node in ('Q25','Q50','Q75'):
                    value=views['trend'][key][node+'_'+metric+'_mean'];counts=views['trend'][key][node+'_'+metric+'_defined_draws']
                    trend_rows.append([LABEL[arm],LABEL[h],node,metric,fmt(finite_mean(value)),
                        fmt(finite_mean(views['trend'][key][node+'_'+metric+'_MonteCarlo_SE'])),int(np.isfinite(value).sum()),32,
                        int(counts.sum()),int(32*8-counts.sum())])
    for split in SPLITS:
        for scheme,entry in populations[split]['schemes'].items():
            weight_rows.append([split,scheme,entry['retained_parents'],fmt(entry['weight_min']),fmt(entry['weight_max']),
                fmt(entry['parent_weight_sum']),fmt(entry['parent_ESS'])])
            for arm in ARMS:
                for h in HYP:
                    for metric,item in entry['arms'][arm][h]['population'].items():
                        population_rows.append([split,scheme,LABEL[arm],LABEL[h],metric,fmt(item['mean'][0]),fmt(item['std'][0]),
                            item['defined_rows'][0],item['undefined_rows'][0],fmt(item['defined_weight'][0]),fmt(item['undefined_weight'][0])])
    figures_html=''.join('<section class="figure"><img src="figures/'+name+'.png" alt="报告第'+str(i+1)+'页"></section>' for i,name in enumerate(figures))
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>N01-008固定终点评价</title>
<style>body{font-family:"Microsoft YaHei",sans-serif;background:#f1f5f8;color:#172e43;margin:0}main{max-width:1240px;margin:28px auto;padding:28px;background:white}h1{font-size:29px;margin-top:0}h2{margin-top:30px}p{line-height:1.75}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #dae3e9}th:first-child,td:first-child{text-align:left}th{background:#e7f0f6}img{width:100%;display:block}.figure{margin:25px 0}a{color:#006e87}.status{padding:16px 20px;background:#e7f2f5;border-left:5px solid #007f8b}details{margin:18px 0;padding:12px;border:1px solid #dce6ec;overflow:auto}summary{cursor:pointer;font-size:17px;font-weight:600}.method{color:#304c60}code{font-family:Consolas,monospace}</style><main>
<h1>N01-008：256点有效随机响应固定终点</h1><p class="status"><b>三臂训练、主体评价、精确数值回放与四项冻结诊断均已完成；物理响应匹配尚未建立。</b><br>
每臂19,000 G，比较预算57,000 G / 144,000 D；历史实际'''+f"{complete['historical_actual_G_updates']:,} G / {complete['historical_actual_D_updates']:,} D。"+'''主评价与诊断的G/D/噪声/人口/C2ST更新均为0。</p>
<p><b>'''+html.escape(executive_findings(summary)['wave_comparison'])+'''</b> '''+html.escape(executive_findings(summary)['DEV_wave_values'])+'''<br>'''+html.escape(executive_findings(summary)['QPSD_split_tradeoff'])+''' 三值顺序为短相关 / 慢相关 / 公共分量。</p>
<p>有效窗为峰前16点至峰后239点，1 ns/点；波形V，Q/tail为V ns，PSD无量纲。内部384点载体保留原100点ADC基线和固定寻峰上下文。SOURCE保持清洁，REAL保留已有ADC，生成载体数字化一次。所有合法父事件和原权重保留。</p>
<p><a href="N01_WINDOW008_RESUME01_REPORT.pdf">下载'''+str(len(figures))+'''页PDF</a> · <a href="SUMMARY.json">执行与结果摘要</a> · <a href="EVIDENCE.json">完整数值证据、定义次数与来源SHA256</a></p>
<h2>匹配分布距离</h2><p>TRAIN每域2,048、DEV每域256；SOURCE取第0次响应，REAL A/B按软件组平衡且不交叠。三个噪声假设分别报告，距离采用冻结标准化投影。</p>'''+html_table(['角色','模型','噪声假设','波形','QPSD','原100点基线'],distance_rows)+'''
<details><summary>full相对两个对照的各假设差异</summary><p>相对减少=1−full/reference；正值表示该固定距离降低，负值表示增加；参考距离为0时未定义。</p>'''+html_table(['角色','对照','假设','距离项','full−reference','相对减少'],relative_rows)+'''</details>
<h2>报告图表</h2>'''+figures_html+'''
<details><summary>两种分箱的Q/PSD均值与宽度</summary><p>Q/tail为V ns，PSD无量纲；每箱父数≥32为支持。输出Q箱与SOURCE Q箱属于两种不同人口条件。</p>'''+html_table(['角色','分箱','模型','假设','箱','父数','draw行','支持','Q均值','Q SD','PSD均值','PSD SD','raw未定义','guard','Q≤0'],conditional_rows)+'''</details>
<details><summary>全量成对父bootstrap差异与95%区间</summary><p>同一父的四draw共同重采样；区间端点分别索引lower/upper。每格保留有效replicate数量。Q的均值/SD差为V ns，PSD差为无量纲，SD比为无量纲。</p>'''+html_table(['角色','分箱','对照','假设','箱','量','支持','Δ均值','下限','上限','定义次','ΔSD','下限','上限','定义次','SD比','下限','上限','定义次'],bootstrap_rows)+'''</details>
<details><summary>128×64条件方差与协方差</summary><p>对每个固定输入计算64次样本统计，ddof=1。下表是128父的均值和分位数；完整逐父结果、17量协方差/共同定义次数与256×256波形协方差来源保存在EVIDENCE及原NPZ。</p>'''+html_table(['模型','假设','统计','单位','父均值','5%','中位数','95%','定义父','未定义父'],repeat_rows)+'''</details>
<details><summary>固定Q形状干预：成对差与Monte Carlo精度</summary><p>全部32父各8组latent成对复用。Δ为donor−original；概括均值按定义父等权。MCSE列是各父8draw差估计SE的均值，父间变化另由原逐父结果呈现。实际输入差与Q间隔固定保留。</p>'''+html_table(['模型','假设','量','定义父Δ均值','每父MCSE均值','定义父','全部父','共同定义draw','未定义draw'],shape_rows)+'''</details>
<details><summary>同形状Q节点：全部量与定义次数</summary><p>节点为冻结TRAIN Q8边界索引2/4/6，Q/tail单位V ns，rise/FWHM单位ns，PSD/time_scale无量纲。qbar结构相邻差、带噪声Q趋势和成对MCSE分别保存在原记录。</p>'''+html_table(['模型','假设','节点','量','定义父均值','每父MCSE均值','定义父','全部父','定义draw','未定义draw'],trend_rows)+'''</details>
<details><summary>TRAIN/DEV人口权重、ESS与全部加权标量</summary><p>父权重和等于原父数，父的四draw同权重，人口矩ddof=0。raw PSD按定义质量统计，未定义行数/权重质量同时保留。全局波形矩由固定SOURCE-bin矩合成；output-Q条件加权波形矩因缺联合矩而未提供。</p>'''+html_table(['角色','方案','保留父','权重最小','权重最大','父权重和','ESS'],weight_rows)+html_table(['角色','方案','模型','假设','量','加权均值','SD','定义行','未定义行','定义质量','未定义质量'],population_rows)+'''</details>
<details><summary>全部异常flags与未定义统计</summary><p>分母是对应角色/变体的全部观测行。ADC碰轨按原384点载体计；未保存字段明确显示“未保存”。全部Q≤0、guard、raw未定义、寻峰边界和ADC异常行保留。</p>'''+html_table(['任务','角色','模型','假设','变体','事件','计数','全部行','比例'],flag_rows)+'''</details>
<details><summary>原100点基线、latent与调用证据</summary><p>基线为原载体100点ADC。报告JSON包含RMS/floor RMS、lag0–99平均ACF、五频带份额、S统计与方差floor次数。三假设的384点/后284点噪声、同窗Q/tail分量协方差、工程恒等式残差和实际调用数也保留。</p><pre>'''+html.escape(json.dumps(clean(dict(baseline100=baseline,actual_counters=evidence['counts'])),ensure_ascii=False,indent=2))+'''</pre></details>
<h2>提案覆盖与当前证据范围</h2>'''+html_table(['原提案项目','实际状态','范围'],[[v['item'],v['report_status'],v['scope']] for v in coverage])+'''
<p>当前结果描述单次已暴露采集下，相对于解析SOURCE输入和既有软件电压口径的有效响应候选。三个电子学扩展属于敏感性假设，源Q尚未建立能量标定，人口倾斜只覆盖温和SOURCE-Q变化。条件协方差与输入干预有各自的数值证据；物理可复用性需要配对重复REAL、独立采集或标定约束。独立C2ST未执行，AUC保留为未获得。</p></main></html>'''
    with (out/'N01_WINDOW008_RESUME01_REPORT.html').open('x',encoding='utf8') as stream:stream.write(body)
    artifacts={path.relative_to(out).as_posix():sha(path) for path in sorted(out.rglob('*')) if path.is_file()}
    write(out/'BUILD_RECEIPT.json',dict(schema='n01/window008-resume01-report-build/v1',UTC=headline['UTC'],files=artifacts,
        report_pages=len(figures),PDF_operation_marker_receipt=str(marker_path),PDF_operation_marker_sha256=sha(marker_path),
        PDF_operation_marker=marker,source_files_sha256=evidence['source_hashes'],
        source_data_role='saved completed fixed TRAIN and exposed DEV endpoint/diagnostic outputs only',
        new_model_calls=0,new_training_updates=0,new_original_waveform_decodes=0,
        visual_QA='pending_all_pages_rendered_inspection',implementation_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status='report_built_pending_visual_QA',out=str(out),pages=len(figures),files=len(artifacts))))


if __name__=='__main__':main()
