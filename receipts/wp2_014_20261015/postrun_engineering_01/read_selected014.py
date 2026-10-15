"""Read only collected014 JSON/JSONL; reconstruct frozen development selection."""
from __future__ import annotations
import argparse
from collections import defaultdict
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys

HYPOTHESES=('fast','slow','common_level')
CANDIDATES=('C0','C1')
TRIALS=('seed0_C0','seed0_C1','seed1_C0','seed1_C1')
STATUS='provisional_pending_independent_raw_readback'


def require(test,message):
    if not test:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write_new(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def average(values):return sum(values)/len(values) if values and all(finite(v) for v in values) else None
def stats(values):
    a=[float(v) for v in values if finite(v)]
    return dict(n=len(a),minimum=min(a),median=statistics.median(a),mean=average(a),maximum=max(a)) if a else dict(n=0)


class Reader:
    def __init__(self,collected,collection):
        self.root=collected.resolve();self.collection=collection.resolve();self.checked={}
        self.remote=self.external_json(self.collection/'REMOTE_FULL_INVENTORY.json')
        self.receipt=self.external_json(self.collection/'COLLECTION_RECEIPT.json')
        require(self.receipt['status']=='native_inventory_and_selected_bytes_verified' and not self.receipt['partial_evidence_only'],'collection not complete')
        self.inventory=self.remote['inventory'];self.all_inventory=self.remote['whole_remote_inventory']
        require(self.receipt['whole_remote_inventory_verified'] and not self.receipt['all_raw_artifacts_collected'],'incorrect selective-collection scope')
    def check(self,path,expected=None):
        path=path.resolve();key=str(path)
        require(path.is_file() and not path.is_symlink() and path.drive.upper() in ('E:','D:'),'unsafe local read')
        if key not in self.checked:
            value=dict(bytes=path.stat().st_size,sha256=sha(path))
            if expected is not None:require(value==expected,'selected bytes differ: '+key)
            self.checked[key]=value
        return path
    def external_json(self,path,expected=None):return json.loads(self.check(path,expected).read_text(encoding='utf-8'))
    def path(self,name):
        require(name in self.inventory and name.endswith(('.json','.jsonl')),'JSON/JSONL selected members only')
        path=self.root/name;require(path.resolve().is_relative_to(self.root),'read escaped selected root')
        return self.check(path,self.inventory[name])
    def json(self,name):return json.loads(self.path(name).read_text(encoding='utf-8'))
    def unchanged(self):
        for name,record in self.checked.items():
            p=Path(name);require(p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],'readback input changed')


def choose(eligible_scores,epsilon):
    if not eligible_scores:return dict(development_path='C0',eligible_set=[],fallback_engineering_entry=True,weights_promoted=False)
    require(all(finite(s) for s in eligible_scores.values()),'undefined eligible score')
    if set(eligible_scores)==set(CANDIDATES) and abs(eligible_scores['C0']-eligible_scores['C1'])<=epsilon:chosen='C0'
    else:chosen=min(eligible_scores,key=lambda n:(eligible_scores[n],n))
    return dict(development_path=chosen,eligible_set=list(eligible_scores),fallback_engineering_entry=False,weights_promoted=False)


def component_cell(final_values,A_values,initial_values,T):
    report={};eligible=True;reasons=[];pending=[]
    for key,bound in T.items():
        f=average([v.get(key) for v in final_values]);a=average([v.get(key) for v in A_values]);i=average([v.get(key) for v in initial_values])
        calibrated=finite(bound) and bound>0
        if calibrated and (f is None or a is None):
            result='pending_missing_required_calibrated_metric';eligible=False;reasons.append(key+':missing_or_undefined')
        elif calibrated and f-a>bound:
            result='failed_noninferiority';eligible=False;reasons.append(key+':delta_above_T')
        elif calibrated:result='noninferior'
        else:result='pending_not_calibrated';pending.append(key)
        report[key]=dict(final=f,A=a,common_initial=i,T=bound,delta_final_minus_A=f-a if f is not None and a is not None else None,
            delta_final_minus_initial=f-i if f is not None and i is not None else None,
            noninferiority_status=result,absolute_distance_over_T=f/bound if calibrated and f is not None else None,
            absolute_within_T=(f<=bound) if calibrated and f is not None else None,
            noninferior_vs_common_initial=(f-i<=bound) if calibrated and f is not None and i is not None else None)
    return report,eligible,reasons,pending


def semantic_checks():
    T=dict(wave=.11,QPSD=.1,noise=.1,undefined_flag=None)
    A=[dict(wave=1.,QPSD=.2,noise=.3,undefined_flag=0.)]*2
    F=[dict(wave=.9,QPSD=.2,noise=.3,undefined_flag=0.),dict(wave=1.3,QPSD=.2,noise=.3,undefined_flag=0.)]
    cell,ok,_,pending=component_cell(F,A,A,T)
    require(ok and math.isclose(cell['wave']['delta_final_minus_A'],.1) and 'undefined_flag' in pending,'ALL6 means changed to ALL12 bank gate')
    F[0]=dict(F[0],wave=None)
    require(not component_cell(F,A,A,T)[1],'missing positive-T metric automatically passed')
    require(choose({'C0':1.,'C1':1.-.5e-12},1e-12)['development_path']=='C0','tie rule changed')
    require(choose({'C0':1.,'C1':1.-2e-12},1e-12)['development_path']=='C1','best eligible score rule changed')
    require(choose({},1e-12)['fallback_engineering_entry'] and not choose({},1e-12)['weights_promoted'],'empty set promoted a model')
    require(1-.9801122114104647 < .02,'near2percent rounding promoted a failed cell')
    return dict(passed_count=6,artificial_only=True,checks=['mean2bank_before_ALL6','missing_required_metric_ineligible','uncalibrated_pending','tie_prefers_C0','empty_set_no_weight_promotion','full_precision_usefulness_boundary'])


def trial_costs(reader,complete,p):
    log=reader.path('runs/primary/artifacts/operations.jsonl')
    attempted=defaultdict(lambda:defaultdict(int));completed=defaultdict(lambda:defaultdict(int));pending={}
    stage='shared_preparation';seed=-1;trial=-1;entered=finished=0
    with log.open(encoding='utf-8') as stream:
        for line in stream:
            require(line.endswith('\n'),'truncated operation journal')
            row=json.loads(line);event=row['event'];oid=row['operation_id']
            if event=='entered':
                require(oid==entered and oid not in pending,'operation identity not sequential')
                if row['name']=='initial_RMS_frozen_A_cache':seed+=1;stage=f'seed{seed}_CALIBRATION'
                if row['name']=='new_head_AdamW_created':trial+=1;require(trial<4,'extra trial');stage=TRIALS[trial]
                pending[oid]=(stage,row['charges']);entered+=1
                for k,v in row['charges'].items():attempted[stage][k]+=v
            elif event=='completed':
                require(oid in pending,'unentered/duplicate completed operation')
                label,charges=pending.pop(oid);require(charges==row['charges'],'operation charge changed')
                for k,v in charges.items():completed[label][k]+=v
                finished+=1
            else:raise ValueError('failed operation in successful saved journal')
    require(seed==1 and trial==3 and not pending and entered==finished,'incomplete serial cost journal')
    totals={k:sum(values.get(k,0) for values in completed.values()) for k in p['budget_caps']}
    require(totals==complete['counters'] and totals==complete['attempted_counters'],'cost segment total differs from COMPLETE')
    require({k:dict(v) for k,v in attempted.items()}=={k:dict(v) for k,v in completed.items()},'attempted/completed stage costs differ')
    return dict(schema='n01/wp2-saved-trial-costs/v1',operation_events_entered=entered,operation_events_completed=finished,
        actual_completed_by_stage={k:dict(v) for k,v in completed.items()},total_completed=totals,unknown_inflight_ids=[],failed_operation_ids=[],
        attribution='chronological entered-operation stages: shared preparation; each initial_RMS_frozen_A_cache seed calibration; each new_head_AdamW_created starts the next preregistered trial',
        shared_seed_calibration_not_silently_charged_to_one_candidate=True,no_new_scientific_cost_incurred=True)


def training_and_probe(reader,trial,p):
    prefix='runs/primary/artifacts/'+trial+'/'
    inputs=reader.json(prefix+'RUN_INPUTS.json');done=reader.json(prefix+'COMPLETE.json')
    with reader.path(prefix+'training.jsonl').open(encoding='utf-8') as stream:logs=[json.loads(line) for line in stream]
    expected=[1]+list(range(100,2001,100));require([r['G_updates'] for r in logs]==expected,'missing saved training log checkpoint')
    gradient=[]
    for step in (0,2000):
        saved=reader.json(prefix+f'probe_{step:04d}/SUMMARY.json');records=saved['records']
        require(len(records)==78 and saved['extra_scalar_AD_queries']==sum(r['AD_queries'] for r in records),'probe cell/query accounting differs')
        groups=[]
        for stratum in sorted({r['stratum'] for r in records}):
            rows=[r for r in records if r['stratum']==stratum];defined=[r for r in rows if r['status']=='computed']
            ratios=[r['gradient_norms']['weighted_distribution']/r['gradient_norms']['wave'] for r in defined if r['gradient_norms']['wave']>0]
            groups.append(dict(stratum=stratum,cells=len(rows),computed_cells=len(defined),AD_queries=sum(r['AD_queries'] for r in rows),
                fake_rows=stats([r['fake_rows'] for r in rows]),fake_parent_count=stats([r['fake_parent_count'] for r in rows]),
                weighted_distribution_to_wave_parameter_gradient_norm_ratio=stats(ratios),
                wave_distribution_cosine=stats([r['cosine']['wave__weighted_distribution'] for r in defined]),
                gradient_norms={k:stats([r['gradient_norms'][k] for r in defined]) for k in ('wave','weighted_distribution','weighted_regular','total')},
                native_QPSD=stats([r.get('native_QPSD') for r in rows]),peak_switch_rows_sum=sum(r['peak_switch_rows'] for r in rows),
                Q_nonpositive_rows_sum=sum(r['Q_nonpositive_rows'] for r in rows),PSD_undefined_rows_sum=sum(r['PSD_undefined_rows'] for r in rows),
                warning='conditional fake subset empirical loss against fixed global128 REAL reference; nonadditive subset gradients, never event contributions or percentages'))
        gradient.append(dict(G_updates=step,AD_queries=saved['extra_scalar_AD_queries'],source_parents=64,draws=2,banks=2,hypotheses=3,
            projection_context=saved['projection_context'],strata=groups,no_per_event_contribution_claim=saved['no_per_event_loss_contribution_claim']))
    configured_stream_hash=hashlib.sha256(json.dumps(inputs['seed'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(G_updates=done['G_updates'],candidate=done['candidate'],initial_seed=done['seed']['initialization'],coverage=done['coverage'],resources=done['resources'],
        frozen_A_005_unchanged=done['frozen_A_005_unchanged'],configured_stream_address_sha256=configured_stream_hash,
        configured_stream_fingerprint_is_actual_RNG_checkpoint_state=False,configured_streams=inputs['seed'],calibrated_initial_head_sha256=inputs['calibration']['calibrated_head_sha256'],
        initial_RMS_ADC_LSB=inputs['calibration']['actual_RMS_ADC_LSB'],logged_updates=expected,
        logged_preclip_gradient_norm=stats([r['preclip_total_gradient_norm'] for r in logs]),
        logged_gradient_above5_count=sum(r['preclip_total_gradient_norm']>5 for r in logs),logged_gradient_denominator=len(logs),
        logged_parameter_update_norm=stats([r['actual_parameter_update_norm'] for r in logs]),
        logged_loss_terms={k:stats([r[k] for r in logs]) for k in ('objective','wave','weighted_distribution','weighted_regular')},
        logs_are_21_individual_minibatches_not_2000_update_averages_or_population_evaluation=True,
        conditional_gradient_diagnostics=gradient),logs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True,type=Path);parser.add_argument('--collected',required=True,type=Path)
    parser.add_argument('--collection',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();repo=args.repo.resolve();out=args.out.resolve()
    require(out.drive.upper() in ('E:','D:'),'E/D output only');out.mkdir(parents=True,exist_ok=True)
    require(not (out/'DEVELOPMENT_SELECTION.json').exists(),'preserve prior selector output')
    temporary=out/'tmp';temporary.mkdir(exist_ok=True);os.environ.update(TEMP=str(temporary),TMP=str(temporary),PYTHONDONTWRITEBYTECODE='1')
    tests=semantic_checks();reader=Reader(args.collected,args.collection)
    p=reader.json('wp2_014_config.json');manifest=reader.json('WP2_SOURCE_MANIFEST.json');complete=reader.json('runs/primary/artifacts/COMPLETE.json')
    require(p['status']=='frozen' and p['development_selection']['noninferiority_cells']=='ALL6_same_candidate_2seeds_x3hypotheses','wrong frozen development quantifier')
    require(p['development_selection']['tie_absolute_epsilon']==1e-12 and p['development_selection']['tie_preference']=='C0','tie rule changed')
    T_path=Path(p['contracts']['tolerance_table']['local_path']);T_record={k:p['contracts']['tolerance_table'][k] for k in ('bytes','sha256')}
    tolerance=reader.external_json(T_path,T_record);T=tolerance['T_by_hypothesis']
    require(complete['all_four_trials_completed'] and complete['counters']['G_updates']==8000 and not complete['inflight'],'batch incomplete')
    cost=trial_costs(reader,complete,p)
    trial_summaries={};logs_by_trial={};sources={};cells=[];candidate_results={}
    for trial in TRIALS:
        trial_summaries[trial],logs_by_trial[trial]=training_and_probe(reader,trial,p)
        prefix='runs/primary/artifacts/'+trial+'/'
        initial=reader.json(prefix+'initial_evaluation/METRICS.json');final=reader.json(prefix+'final_evaluation/METRICS.json');comparison=reader.json(prefix+'HARD_COMPARISON.json')
        require(initial['source_parents']==final['source_parents']==6593 and initial['REAL_rows']==final['REAL_rows']==21047
            and initial['draws_per_bank']==final['draws_per_bank']==4 and initial['MC_banks']==final['MC_banks']==2,'evaluation population changed')
        sources[trial]=dict(initial=initial['metrics'],final=final['metrics'])
        comparisons={(r['bank'],r['hypothesis']):r for r in comparison['per_bank_hypothesis']}
        for bank,h in itertools.product(('bank0','bank1'),HYPOTHESES):
            for key,check in comparisons[bank,h]['checks'].items():
                require(check['final']==final['metrics'][bank+'_final_'+h]['values'][key] and check['A']==initial['metrics'][bank+'_A_'+h]['values'][key]
                    and check['common_initial']==initial['metrics'][bank+'_initial_'+h]['values'][key],'saved comparison disagrees with original metrics')
    for candidate in CANDIDATES:
        relevant=[];score_values=[]
        for seed,h in itertools.product((0,1),HYPOTHESES):
            trial=f'seed{seed}_{candidate}';source=sources[trial]
            F=[source['final'][f'bank{b}_final_{h}']['values'] for b in (0,1)]
            A=[source['initial'][f'bank{b}_A_{h}']['values'] for b in (0,1)]
            I=[source['initial'][f'bank{b}_initial_{h}']['values'] for b in (0,1)]
            components,eligible,reasons,pending=component_cell(F,A,I,T[h]);wave=components['wave']
            improvement_A=1-wave['final']/wave['A'] if wave['final'] is not None and finite(wave['A']) and wave['A']>0 else None
            improvement_initial=1-wave['final']/wave['common_initial'] if wave['final'] is not None and finite(wave['common_initial']) and wave['common_initial']>0 else None
            cell=dict(candidate=candidate,seed_index=seed,initial_seed=p['seed_sets'][seed]['initialization'],hypothesis=h,components=components,
                positive_T_components=[k for k,v in T[h].items() if finite(v) and v>0],uncalibrated_pending_components=pending,
                development_noninferiority_eligible_cell=eligible,eligibility_failure_reasons=reasons,
                relative_wave_improvement_vs_A=improvement_A,wave_usefulness_ge_2percent=improvement_A is not None and improvement_A>=.02,
                exact_usefulness_margin=improvement_A-.02 if improvement_A is not None else None,
                relative_wave_improvement_vs_common_initial=improvement_initial,
                absolute_positive_T_failed_components=[k for k,v in components.items() if v['absolute_within_T'] is False],
                absolute_uncalibrated_components_pending=True,scientific_accepted=False)
            cells.append(cell);relevant.append(cell);score_values.extend([v.get('wave') for v in F])
        eligible=all(r['development_noninferiority_eligible_cell'] for r in relevant) and all(finite(v) for v in score_values)
        candidate_results[candidate]=dict(eligible=eligible,final_wave_score=average(score_values),score_denominator=12,
            usefulness_ALL6_vs_A=all(r['wave_usefulness_ge_2percent'] for r in relevant),
            usefulness_failed_cells=[dict(seed_index=r['seed_index'],hypothesis=r['hypothesis'],improvement=r['relative_wave_improvement_vs_A'],margin=r['exact_usefulness_margin']) for r in relevant if not r['wave_usefulness_ge_2percent']],
            noninferiority_failed_cells=[dict(seed_index=r['seed_index'],hypothesis=r['hypothesis'],reasons=r['eligibility_failure_reasons']) for r in relevant if not r['development_noninferiority_eligible_cell']],
            absolute_positive_T_all6_pass=all(not r['absolute_positive_T_failed_components'] for r in relevant),
            uncalibrated_components_pending=True,scientific_accepted=False)
    chosen=choose({name:r['final_wave_score'] for name,r in candidate_results.items() if r['eligible']},p['development_selection']['tie_absolute_epsilon'])
    pairing=[]
    for seed in (0,1):
        x,y=f'seed{seed}_C0',f'seed{seed}_C1';sx,sy=trial_summaries[x],trial_summaries[y]
        require(sx['configured_streams']==sy['configured_streams'] and sx['calibrated_initial_head_sha256']==sy['calibrated_initial_head_sha256'],'paired configuration/initial bytes differ')
        same_source=all(a['source_parent_rows']==b['source_parent_rows'] for a,b in zip(logs_by_trial[x],logs_by_trial[y]))
        same_projection=all(a['projection_context']==b['projection_context'] for a,b in zip(logs_by_trial[x],logs_by_trial[y]))
        initial_differences=[]
        for key,metric in sources[x]['initial'].items():
            other=sources[y]['initial'][key]
            initial_differences.extend(abs(value-other['values'][name]) for name,value in metric['values'].items() if finite(value) and finite(other['values'][name]))
        pairing.append(dict(seed_index=seed,configured_stream_address_sha256=sx['configured_stream_address_sha256'],calibrated_initial_head_sha256=sx['calibrated_initial_head_sha256'],
            same_logged_SOURCE_parent_sequences_all21=same_source,same_logged_wave_projection_contexts_all21=same_projection,
            common_initial_and_A_distance_max_abs_difference=max(initial_differences),REAL_actual_row_ID_and_full_latent_state_identity='not independently read from selective JSON'))
    seed_differences=[]
    for candidate,h in itertools.product(CANDIDATES,HYPOTHESES):
        pair=sorted([r for r in cells if r['candidate']==candidate and r['hypothesis']==h],key=lambda r:r['seed_index'])
        seed_differences.append(dict(candidate=candidate,hypothesis=h,seed1_minus_seed0_final_distance={k:pair[1]['components'][k]['final']-pair[0]['components'][k]['final'] if pair[1]['components'][k]['final'] is not None and pair[0]['components'][k]['final'] is not None else None for k in pair[0]['components']},
            interpretation='two fitted-seed descriptive difference; no bootstrap/CI or cross-acquisition claim'))
    original_sources={}
    for name,record in manifest['files'].items():
        local=repo/name;require(local.is_file() and local.stat().st_size==record['bytes'] and sha(local)==record['sha256'],'frozen95 local source changed')
        original_sources[name]=record['sha256']
    require(len(original_sources)==95,'frozen source count changed')
    report=dict(schema='n01/wp2-json-development-selection/v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=STATUS,
        config_development_selection=p['development_selection'],**chosen,candidates=candidate_results,cells=cells,paired_initialization_and_logged_streams=pairing,
        training_and_conditional_gradient_summary=trial_summaries,seed_differences=seed_differences,
        aggregation='mean of two separate empirical distances within each seed/hypothesis, not pooled8-draw distance; eligibility ALL6; score mean12 final-wave distances',
        eligibility_scope='all positive finite frozen-T components have defined final and paired-A values in both banks; uncalibrated components stay pending and are not labelled PASS',
        no_near_threshold_retry=True,scientific_accepted=False,independent_raw_readback_complete=False,
        limitations=['historically exposed TRAIN development evidence from one acquisition','raw NPZ arrays retained on native host and not read here',
            'calibrated T maps registered synthetic perturbations, not independent hardware accuracy','conditional subset gradients are nonadditive and are not event contributions',
            '21 logged single minibatches do not describe every training update or the fixed whole-population metric curve'])
    reader.unchanged()
    write_new(out/'TRIAL_COST_READBACK.json',cost);write_new(out/'DEVELOPMENT_SELECTION.json',report);write_new(out/'SEMANTIC_CHECKS.json',tests)
    lines=['# WP2/014 保存 JSON 开发选择','',f"状态：`{STATUS}`。开发路径选择 **{chosen['development_path']}**；不晋级权重，不宣告科学验收。",'',
        '| 路径 | ALL6 非劣 | 12 个 final wave 均值 | ALL6 wave 相对 A ≥2% |','|---|---|---:|---|']
    for name,result in candidate_results.items():lines.append(f"| {name} | {result['eligible']} | {result['final_wave_score']:.17g} | {result['usefulness_ALL6_vs_A']} |")
    lines+=['','| 路径 | seed | 假设 | A wave | final wave | 相对 A 改善 | 相对共同 initial 改善 | 绝对超 T 分量 |','|---|---:|---|---:|---:|---:|---:|---|']
    for r in cells:
        w=r['components']['wave'];lines.append(f"| {r['candidate']} | {r['seed_index']} | {r['hypothesis']} | {w['A']:.9f} | {w['final']:.9f} | {100*r['relative_wave_improvement_vs_A']:.12f}% | {100*r['relative_wave_improvement_vs_common_initial']:.12f}% | {', '.join(r['absolute_positive_T_failed_components'])} |")
    lines+=['','C1 的 seed1-fast 改善精确为 0.019887788589535327，低于 0.02。不得因四舍五入接近 2% 判通过或追加补跑。',
        '','相对非劣使用同 seed/假设内两个 bank 的距离均值，再对全部 6 格取 ALL；不要求每个 bank 单独通过，也不把距离均值称为 8 draw 合并经验距离。零/未定义 T 分量继续 pending。',
        '','四 trial 均完成 2000 G；每 trial 实际 training backward=2000、probe AD=468、总 AD=2468。共享初始化/校准成本单列。总 G=8000、AD=9872、无在途；完整成本见 TRIAL_COST_READBACK.json。',
        '','梯度表为固定 64 父事件、2draw、2bank 条件子批经验损失的参数梯度，不能相加为全局事件贡献比例。日志只有 1、100…2000 共 21 个单批次点；与终点全人口评估分开。',
        '','选择性回收 142 文件、74,375,000 字节，包含薄头 checkpoint；0 个 NPZ。176 个原始文件约 3.077 GB 留在服务器。当前脚本未反序列化 checkpoint，未执行 NN/AD/优化、DEV 或 SSH。','']
    with (out/'DEVELOPMENT_SUMMARY.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines))
    require(not {'numpy','torch','n01'}&set(sys.modules),'unexpected numerical/model package imported')
    receipt=dict(schema='n01/wp2-json-postrun-receipt/v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=STATUS,
        reader_source=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__))),inputs_sha256=reader.checked,
        frozen95_sources_unchanged=original_sources,semantic_checks=tests,all_read_files_unchanged=True,
        operations_this_readback=dict(array_decodes=0,checkpoint_deserializations=0,NN_forward_rows=0,AD_queries=0,optimizer_steps=0,training_updates=0,DEV_rows=0,SSH_calls=0),
        raw_saved_arrays_locally_complete=False,scientific_accepted=False,ROOT_metadata_modified=False,
        artifacts={name:dict(bytes=(out/name).stat().st_size,sha256=sha(out/name)) for name in ('TRIAL_COST_READBACK.json','DEVELOPMENT_SELECTION.json','DEVELOPMENT_SUMMARY.md','SEMANTIC_CHECKS.json')})
    write_new(out/'READBACK_RECEIPT.json',receipt)
    print(json.dumps(dict(status=STATUS,development_path=chosen['development_path'],candidates=candidate_results,receipt_sha256=sha(out/'READBACK_RECEIPT.json'),artifact_sha256=receipt['artifacts']),ensure_ascii=True))


if __name__=='__main__':main()
