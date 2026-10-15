"""Inspect local saved JSON structures only; never read scientific binary files."""
from pathlib import Path
import json
import sys

ROOT = Path('E:/GAN/research/N01')
ART = ROOT / 'runs/wp3_015_20261015/selected_collected_01/runs/primary/artifacts'
paths = {
    'selection': ROOT / 'receipts/wp3_015_20261015/postrun_independent_01/DEVELOPMENT_SELECTION.json',
    'metrics': ART / 'seed0_M0/final_evaluation/METRICS.json',
    'report': ART / 'seed0_M0/final_evaluation/bank0_final_fast_REPORT.json',
    'gate': ART / 'MEAN_RESIDUAL_GATE.json',
    'complete': ART / 'COMPLETE.json',
    'array': ROOT / 'runs/wp3_015_20261015/independent_readback_collected_01/artifacts/ARRAY_READBACK.json',
    'native': ART.parent / 'supervisor/exit_receipt.json',
    'info': ROOT / 'INPUT_INFORMATION_AUDIT.json',
    'tolerance': ROOT / 'CALIBRATED_TOLERANCES_WP1_013.json',
    'inventory': ROOT / 'receipts/wp3_015_20261015/collection_01/REMOTE_FULL_INVENTORY.json',
    'local': ROOT / 'receipts/wp3_015_20261015/postrun_independent_01/LOCAL_METADATA_READBACK.json',
}

def shape(obj, depth=0):
    if depth > 2:
        return str(type(obj).__name__) + (f'[{len(obj)}]' if isinstance(obj, (dict,list)) else '')
    if isinstance(obj, dict):
        return {k: shape(v, depth+1) for k,v in obj.items()}
    if isinstance(obj, list):
        return [f'len={len(obj)}', shape(obj[0], depth+1)] if obj else []
    return obj

for name, path in paths.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    obj = json.loads(path.read_bytes())
    if name == 'selection':
        val = {k:v for k,v in obj.items() if k != 'cells'}
    elif name == 'array':
        val = {'keys': list(obj), 'pair0': obj['pairs'][0]}
    elif name == 'report':
        val = {'bin0':obj['output_Q_conditions']['bins'][0], 'repeat':obj['repeated_parent'],
               'feature_stats':obj['population']['integrals']['Q_Vns'],
               'source_input_trends_keys':list(obj['source_input_trends']),
               'trend0':obj['source_input_trends']['correlations']['canonical_Q_Vns'],
               'fake_flag0':obj['population']['fake_flags']['Q_nonpositive']}
    elif name == 'metrics':
        val = shape(obj)
    elif name == 'gate':
        val = obj
    elif name == 'inventory':
        val = {'keys':list(obj),'job':obj.get('job')}
    elif name == 'info':
        val = {k:v for k,v in obj.items() if k not in ['PCA','measurements','associations','templates']}
        val['PCA'] = shape(obj['PCA'])
    elif name == 'local':
        val = {k:v for k,v in obj.items() if k not in ['checks','inputs','evidence']}
    elif name == 'tolerance':
        val = {'keys':list(obj),'T_by_hypothesis':obj.get('T_by_hypothesis')}
    else:
        val = shape(obj)
        val.pop('immutable_files_sha256', None)
    print(name, json.dumps(val, ensure_ascii=False, indent=2))
