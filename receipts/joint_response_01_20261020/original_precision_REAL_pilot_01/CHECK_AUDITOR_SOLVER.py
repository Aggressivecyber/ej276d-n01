from pathlib import Path
import ast,json,numpy as np,torch
from scipy.optimize import minimize

torch.set_num_threads(4)
root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
tree=ast.parse((here/'AUDIT_COMPLETE.py').read_text('utf-8'))
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='independent_solve')
scope=dict(np=np,minimize=minimize,fit_labels=('FIT0','FIT1','FIT2'))
exec(compile(ast.Module(body=[node],type_ignores=[]),'<saved-array-independent-solver>','exec'),scope)
records=[]
for directory,config,stems in [
    ('receipts/joint_response_01_20261020/original_draw_precision_fix01/terminal_01/collected/run_01',
     'configs/joint_response_01_original_draw_precision_fix01_20261020.json',['seed0_MULTIBANK_M0','seed1_MULTIBANK_M0']),
    ('receipts/joint_response_01_20261020/wake_20261020_0218_01/collected/run_01',
     'configs/joint_response_01_multibank_REAL_pilot_20261020.json',['seed0_physical_ITER00'])]:
    p=json.loads((root/config).read_bytes())
    for stem in stems:
        with np.load(root/directory/(stem+'.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        report=json.loads((root/directory/(stem+'.json')).read_bytes())
        for i,trial in enumerate(report['trials']):
            prefix=f'trial{i}_';step=a[prefix+'proposal']-a['theta']
            difference=scope['independent_solve'](a['stacked_baseline_residual'],a[prefix+'jacobian'],
                a[prefix+'native_jacobian'],a,p,trial['radius'],step)
            records.append(dict(stem=stem,trial=i,dimension=len(step),max_delta_error=difference))
receipt=dict(status='PASS_INDEPENDENT_SOLVER_ON_SAVED_54_AND_46_DIMENSION_DIRECTIONS',
    cases=records,max_delta_error=max(v['max_delta_error'] for v in records),new_model_calls=0,terminal_audit_executed=False)
(here/'AUDITOR_SOLVER_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
