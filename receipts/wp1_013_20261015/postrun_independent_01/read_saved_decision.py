"""Independent saved-metric algebra and complete local inventory, no loss rerun."""
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,math
from pathlib import Path

ROOT=Path('E:/GAN/research/N01');OUT=ROOT/'receipts/wp1_013_20261015/postrun_independent_01'
BASE=ROOT/'runs/wp1_013_20261015/runtime_fix01_collected_03';JOB=BASE/'runs/primary';ART=JOB/'artifacts'
REC=ROOT/'receipts/wp1_013_20261015/native_repeat_fix01/collection_03'
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def finite(v):return v is not None and math.isfinite(v)
def near(a,b):
    if a is None or b is None:return a is b
    return math.isclose(a,b,rel_tol=2e-13,abs_tol=2e-14)
checks=[]
def ok(v,name):assert v,name;checks.append(dict(name=name,passed=True))

cfg=read(BASE/'wp1_013_config.json');collection=read(REC/'COLLECTION_RECEIPT.json');arc=read(REC/'REMOTE_ARCHIVE_RECEIPT.json')
ok(collection['status']=='native_output_inventory_verified' and not collection['partial_evidence_only'],'successful_collection_metadata')
inv=arc['inventory'];ok(len(inv)==collection['file_count']==265,'all265_regular_files_inventory')
actual={p.relative_to(BASE).as_posix() for p in BASE.rglob('*') if p.is_file()}
ok(actual==set(inv),'collected_inventory_has_no_missing_or_extra_files')
for n,r in inv.items():
    p=BASE/n;assert p.resolve().is_relative_to(BASE) and not p.is_symlink()
    assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],n
ok(True,'all265_collected_size_and_SHA_independently_verified')
print(json.dumps(dict(phase='inventory265_PASS',bytes=sum(v['bytes'] for v in inv.values()))),flush=True)
c=read(ART/'COMPLETE.json')
ok(all(inv['runs/primary/'+n]==r for n,r in c['artifacts'].items()),'producer_COMPLETE_artifacts_match_archive_inventory')
ok(sha(BASE/'wp1_013_config.json')==sha(ROOT/'wp1_013_config.json')
    and sha(BASE/'WP1_SOURCE_MANIFEST.json')==sha(ROOT/'WP1_SOURCE_MANIFEST.json'),'collected_frozen_controls_equal_authorized_current_bytes')

entered={};completed=set();names=Counter();charge={k:0 for k in cfg['budget_caps']}
with (ART/'operations.jsonl').open(encoding='utf8') as f:
    for line in f:
        assert line.endswith('\n');r=json.loads(line);i=r['operation_id'];event=r['event']
        if event=='entered':
            assert i==len(entered) and i not in entered;entered[i]=(r['name'],r['charges'])
        elif event=='completed':
            assert i in entered and i not in completed and entered[i]==(r['name'],r['charges'])
            completed.add(i);names[r['name']]+=1
            for k,v in r['charges'].items():assert type(v)is int and v>=0;charge[k]+=v
        else:raise AssertionError(('unexpected failed/unknown journal event',event))
ok(len(entered)==len(completed)==29201 and charge==c['counters'],'29201_named_operation_identity_charges_and_completion_order_verified')

cal=read(ART/'CALIBRATION.json');t_saved=read(ART/'CALIBRATED_TOLERANCES.json');params=read(ART/'PARAMETER_VALIDATION.json');selection=read(ART/'METHOD_SELECTION.json')
hyp=('fast','slow','common_level');records=cal['records']
ok(len(records)==126 and len({(r['case_index'],r['bank'],r['hypothesis']) for r in records})==126,'126_unique_calibration_case_bank_hyp_records')
def chosen(case):
    return (case['kind'] in ('gain','tail') and case.get('percent')==3) or (case['kind']=='noise' and case.get('percent')==10) or (case['kind']=='width' and case.get('sigma')==1)
ok(all(r['tolerance_case']==chosen(r['case']) for r in records),'seven_case_selection_independent_of_native_boolean')
T={};winner={};coverage_incomplete=0
for h in hyp:
    chosen_rows=[r for r in records if r['hypothesis']==h and chosen(r['case'])]
    indices=sorted({r['case_index'] for r in chosen_rows});assert len(indices)==7 and len(chosen_rows)==14
    T[h]={};winner[h]={}
    for key in chosen_rows[0]['self_distance']['values']:
        pairmeans=[];quality=True
        for idx in indices:
            pair=sorted([r for r in chosen_rows if r['case_index']==idx],key=lambda r:r['bank'])
            assert [r['bank'] for r in pair]==[0,1];v=[r['self_distance']['values'][key] for r in pair]
            if not all(finite(x) for x in v):quality=False;continue
            if key=='QPSD' and not all(r['self_distance']['coverage']['conditional_coverage_complete'] for r in pair):quality=False
            pairmeans.append(((v[0]+v[1])/2,idx,pair[0]['case']))
        best=max(pairmeans,key=lambda x:x[0]) if quality and len(pairmeans)==7 else None
        T[h][key]=best[0] if best is not None and best[0]>0 else None
        winner[h][key]=None if best is None else dict(case_index=best[1],case=best[2],mean_two_bank_distance=best[0])
        assert near(T[h][key],t_saved['T_by_hypothesis'][h][key]),(h,key,T[h][key],t_saved['T_by_hypothesis'][h][key])
ok(True,'all_T_values_reconstructed_max7_of_samecase_mean2banks_including_unavailable_zero_status')
def check_metric(m):
    global coverage_incomplete
    assert near(m['legacy_objective'],m['values']['wave']+2*m['values']['QPSD'])
    w=m['wave_components'];assert near(m['values']['wave'],.5*w['swd']+.25*w['coordinate']+.25*w['short_lag'])
    cv=m['coverage'];assert cv['global_term_includes_all_rows'] and cv['minimum_conditional_support']==16
    assert cv['projection_context']['fingerprint']==cfg['reference']['projection']['fingerprint']
    coverage_incomplete+=not cv['conditional_coverage_complete']
    assert sum(x['fake_count'] for x in cv['conditional_bins'])==cv['fake_global_rows']
    assert sum(x['real_count'] for x in cv['conditional_bins'])==cv['real_global_rows']
    assert all(x['supported']==(x['fake_count']>=16 and x['real_count']>=16) for x in cv['conditional_bins'])
for r in records:
    for key in ('distance_to_REAL','self_distance'):check_metric(r[key])
rows=params['records'];ok(len(rows)==360 and len({(r['direction'],r['nominal_scale_LSB'],r['sign'],r['bank'],r['batch'],r['hypothesis']) for r in rows})==360,'all30_states_2banks_2batches_3hyp_no_duplicate_or_dropped_cells')
for r in rows:
    for m in list(r['paths'].values())+[r['reference']]:check_metric(m)
    ref=r['reference'];S=r['paths']['S'];H=r['paths']['H_moving']
    assert near(r['S_delta_objective'],S['legacy_objective']-ref['legacy_objective'])
    assert near(r['H_delta_objective'],H['legacy_objective']-ref['legacy_objective'])
    for k in ('wave','QPSD','noise'):assert near(r['hard_component_delta'][k],H['values'][k]-ref['values'][k])
primary=[r for r in rows if r['direction'] in cfg['parameter_validation']['primary_directions'] and r['sign']==1 and r['nominal_scale_LSB']==1]
ok(len(primary)==24,'fixed_primary24_positive1LSB_cells_only')
decisions={};eps=1e-12
for d in cfg['parameter_validation']['primary_directions']:
    group=[r for r in primary if r['direction']==d]
    assert {(r['bank'],r['batch'],r['hypothesis']) for r in group}=={(b,k,h) for b in (0,1) for k in (0,1) for h in hyp}
    cells=[]
    for r in group:
        th=T[r['hypothesis']];calibrated=all(finite(th[k]) and th[k]>0 for k in ('wave','QPSD','noise'))
        ds=r['paths']['S']['legacy_objective']-r['reference']['legacy_objective']
        dh=r['paths']['H_moving']['legacy_objective']-r['reference']['legacy_objective']
        delta={k:r['paths']['H_moving']['values'][k]-r['reference']['values'][k] for k in ('wave','QPSD','noise')}
        ratios={k:r['S_H_self_distance'][k]/th[k] if finite(th[k]) and th[k]>0 and finite(r['S_H_self_distance'][k]) else None for k in ('wave','QPSD')}
        adequate=calibrated and ds<-eps and dh<-eps and all(delta[k]<=th[k] for k in delta)
        repair=calibrated and ds<-eps and dh>eps and any(finite(v) and v>1 for v in ratios.values())
        cells.append(dict(bank=r['bank'],batch=r['batch'],hypothesis=r['hypothesis'],S_delta=ds,H_delta=dh,hard_component_delta=delta,
            materiality_ratios=ratios,adequate=adequate,repair=repair,actual_RMS_V=r['actual_carrier_delta_RMS_V'],actual_absmax_V=r['actual_carrier_delta_absmax_V']))
    decisions[d]=dict(adequate_all12=all(r['adequate'] for r in cells),repair_all12=all(r['repair'] for r in cells),cells=cells)
    native=selection['direction_decisions'][d]
    assert decisions[d]['adequate_all12']==native['adequate_all12'] and decisions[d]['repair_all12']==native['repair_all12']
good=[d for d,x in decisions.items() if x['adequate_all12']];unsafe=[d for d,x in decisions.items() if x['repair_all12']]
choice='adequate' if good else 'repair_needed' if unsafe else 'inconclusive'
ok(choice==selection['choice']==c['method_choice'] and good==selection['adequate_directions'] and unsafe==selection['unsafe_directions'],'independent_ANY_same_direction_ALL12_and_priority_choice')
receipt=dict(schema='n01/wp1-independent-saved-decision/v1',status='PASS',passed=True,UTC=datetime.now(timezone.utc).isoformat(),
    collection=pin(REC/'COLLECTION_RECEIPT.json'),archive_receipt=pin(REC/'REMOTE_ARCHIVE_RECEIPT.json'),checks=checks,
    file_count=265,total_file_bytes=sum(v['bytes'] for v in inv.values()),operation_count=29201,operations_by_name=dict(names),
    T=T,winning_cases=winner,T_primary={h:{k:T[h][k] for k in ('wave','QPSD','noise')} for h in hyp},
    conditional_coverage_incomplete_metric_records=coverage_incomplete,
    decision=choice,adequate_directions=good,unsafe_directions=unsafe,direction_decisions=decisions,
    no_recomputed_raw_calibration_distances=True,no_new_statistic_fit_or_threshold_selection=True,actual_new_NN_rows=0,actual_AD_queries=0,
    actual_training_updates=0,original_CP_reads=0,original_DEV_wave_reads=0,SSH_calls=0,
    interpretation='Adequate is confined to the fixed theta2000/saved-gradient finite1LSB direction and all12 registered TRAINbank cells. It neither proves all training updates nor response matching. T is an engineering-assumption map, not hardware calibration.')
target=OUT/'DECISION_READBACK.json'
with target.open('x',encoding='utf8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(status='PASS',receipt=str(target),sha256=sha(target),decision=choice,adequate_directions=good,T_primary=receipt['T_primary'],coverage_incomplete=coverage_incomplete)))
