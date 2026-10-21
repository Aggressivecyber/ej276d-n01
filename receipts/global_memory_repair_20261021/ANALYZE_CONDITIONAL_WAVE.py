from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
from scipy.optimize import linprog

out = Path(__file__).parent / 'conditional_wave_01'
step = json.loads((out / 'TRAINING_STEPS.jsonl').read_text())
memory = json.loads((out / 'CONDITIONAL_WAVE_MEMORY.json').read_bytes())
names = step['constraint_names']
a = np.asarray(step['constraint_changes'])
b = np.asarray(step['constraint_allowances'])
c = np.asarray([sum(p['losses'].values()) - sum(step['previous_losses'].values()) for p in step['probes']])
support = [k for k, keep in zip(memory['names'], memory['supported']) if keep]
scales = np.maximum(np.maximum(np.abs(a).max(1), np.abs(b)), 1e-12)
full = linprog(c / max(abs(c).max(), 1e-12),
    A_ub=np.vstack((a / scales[:, None], np.ones(len(c)))),
    b_ub=np.r_[b / scales, 1.], bounds=(0., 1.), method='highs')
assert full.success
keep = np.array([not k.startswith('Q_wave') for k in names])
old = linprog(c / max(abs(c).max(), 1e-12),
    A_ub=np.vstack((a[keep] / scales[keep, None], np.ones(len(c)))),
    b_ub=np.r_[b[keep] / scales[keep], 1.], bounds=(0., 1.), method='highs')
assert old.success
probes = []
for j, probe in enumerate(step['probes']):
    previous = step['previous_metrics']
    now = probe['positive_metrics']
    probes.append(dict(direction=probe['direction'],
        global_objective_change=float(c[j]),
        Q_wave_change_percent=100 * (now['Q_wave'] / previous['Q_wave'] - 1),
        global_wave_change_percent=100 * (now['block_wave'] / previous['block_wave'] - 1),
        PSD_change_percent=100 * (now['PSD_CDF'] / previous['PSD_CDF'] - 1),
        supported_wave_cells_worsened=[k for k in support if now[k] > previous[k]],
        old_continuous_constraints_violated=[k for i, k in enumerate(names) if keep[i] and a[i, j] > b[i]],
        native_flag_increases=[dict(bank=i, flag=k, before=step['previous_flags'][i][k], after=probe['flags'][i][k])
            for i in range(2) for k in probe['flags'][i] if probe['flags'][i][k] > step['previous_flags'][i][k]]))
record = dict(UTC=datetime.now(timezone.utc).isoformat(),
    supported_cells=len(support), total_cells=len(memory['names']), probes=probes,
    full_constraints=dict(coefficients=full.x.tolist(), predicted_change=float(c @ full.x)),
    old_constraints_only=dict(coefficients=old.x.tolist(), predicted_change=float(c @ old.x),
        conditional_cells_predicted_worse=[k for i, k in enumerate(names) if k in support and float(a[i] @ old.x) > 0]),
    binding_dual_constraints=[dict(name=k, scaled_dual=float(dual)) for k, dual in
        zip(names, full.ineqlin.marginals[:-1]) if abs(dual) > 1e-10],
    scientific_acceptance=False, new_model_forwards=0, optimizer_updates=0,
    scope='Saved native finite-direction linear analysis. Removing constraints is explanatory only; no relaxed-guard candidate is executed or accepted.')
(out / 'DIRECTION_ANALYSIS.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(dict(supported_cells=len(support), total_cells=len(memory['names']),
    probes=[dict(direction=p['direction'], Q_wave_change_percent=p['Q_wave_change_percent'],
        global_wave_change_percent=p['global_wave_change_percent'], PSD_change_percent=p['PSD_change_percent'],
        worsened_cells=len(p['supported_wave_cells_worsened']), old_failures=p['old_continuous_constraints_violated']) for p in probes],
    full_prediction=record['full_constraints'], old_constraint_prediction=record['old_constraints_only'],
    binding_dual_constraints=record['binding_dual_constraints'])))
