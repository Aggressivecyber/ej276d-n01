"""Independent stdlib-only artificial cells for the registered method decision."""
from pathlib import Path
import copy
import hashlib
import json
import sys

ROOT=Path('E:/GAN/research/N01')
OUT=ROOT/'receipts/wp1_013_20261015/independent_01/method_rule_01'
sys.path.insert(0,str(ROOT/'scripts'))
import wp1_parameter_probe as helper


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if OUT.exists():raise RuntimeError('Preserve previous independent fixture')
    names=('negative_saved012_total','prospective_AdamW2001')
    tolerance={h:dict(wave=1.,QPSD=1.,noise=1.) for h in ('fast','slow','common_level')}
    def cells(name,H=-.2,S=-.1,material=.5):
        return [dict(direction=name,bank=b,batch=k,hypothesis=h,sign=1,nominal_scale_LSB=1.,
                     S_delta_objective=S,H_delta_objective=H,hard_component_delta=dict(wave=.1,QPSD=.1,noise=0.),
                     S_H_self_distance=dict(wave=material,QPSD=material))
                for b in (0,1) for k in (0,1) for h in ('fast','slow','common_level')]
    checks=[]
    result=helper.select_method_once(cells(names[0])+cells(names[1],H=.2,material=2.),tolerance)
    checks.append(dict(name='ANYadequate_has_priority_but_reports_unsafe_other_direction',
                       passed=result['choice']=='adequate' and result['unsafe_directions']==[names[1]]))
    result=helper.select_method_once(cells(names[0],H=0.)+cells(names[1],H=.2,material=2.),tolerance)
    checks.append(dict(name='ALL12_material_S_H_opposite_sign_without_adequate_is_repair',passed=result['choice']=='repair_needed'))
    result=helper.select_method_once(cells(names[0])[:4]+cells(names[1]),tolerance)
    checks.append(dict(name='missing_cells_keep12_inconclusive_slots_other_direction_can_adequate',
                       passed=result['choice']=='adequate' and len(result['direction_decisions'][names[0]]['cells'])==12
                           and sum(x['reason']=='missing_primary_cell' for x in result['direction_decisions'][names[0]]['cells'])==8))
    result=helper.select_method_once([],tolerance,undefined_directions={names[1]:'zero_parameter_direction'})
    checks.append(dict(name='missing_and_zero_directions_close_inconclusive_without_native_failure',
                       passed=result['choice']=='inconclusive' and all(len(d['cells'])==12 for d in result['direction_decisions'].values())))
    all_bad=cells(names[0],H=.2,material=2.)+cells(names[1],H=.2,material=2.)
    for value in (None,0.,float('nan')):
        t=copy.deepcopy(tolerance)
        for h in t:t[h]['noise']=value
        result=helper.select_method_once(all_bad,t)
        checks.append(dict(name='all_required_tolerances_calibrated_for_repair_'+str(value),passed=result['choice']=='inconclusive'))
    malformed=[cells(names[0])+[cells(names[0])[0]],copy.deepcopy(cells(names[0]))]
    malformed[1][0]['bank']=2
    for index,x in enumerate(malformed):
        rejected=False
        try:helper.select_method_once(x,tolerance)
        except ValueError:rejected=True
        checks.append(dict(name='duplicate_or_outside_registered_axes_rejected_'+str(index),passed=rejected))
    checks.append(dict(name='no_numeric_or_neural_library_imported',passed=not any(n in sys.modules for n in ('torch','numpy','n01'))))
    receipt=dict(schema='n01/wp1-013-independent-method-artificial/v1',status='PASS' if all(x['passed'] for x in checks) else 'FAIL',
                 passed=all(x['passed'] for x in checks),checks=checks,artificial_only=True,
                 helper_sha256=sha(ROOT/'scripts/wp1_parameter_probe.py'),script_sha256=sha(Path(__file__)),
                 original_array_decodes=0,checkpoint_reads=0,NN_instances=0,NN_rows=0,AD_queries=0,optimizer_created=0,
                 scientific_updates=0,DEV_reads=0,SSH_calls=0,old_pipeline_replays=0)
    OUT.mkdir(parents=True,exist_ok=False)
    target=OUT/'ENGINEERING_RECEIPT.json'
    with target.open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    assert json.loads(target.read_bytes())['passed'] is True
    print(json.dumps(dict(path=str(target),sha256=sha(target),status=receipt['status'],checks=len(checks))))


if __name__=='__main__':main()
