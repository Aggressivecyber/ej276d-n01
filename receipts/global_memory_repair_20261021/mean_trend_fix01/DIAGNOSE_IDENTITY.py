import json
from pathlib import Path
from collections import Counter
import torch
import numpy as np
import n01.inference028 as inference028
inference028.verified_path028 = lambda record: Path(record['path'])
import run_response038
run_response038.verified_path028 = inference028.verified_path028
from run_joint_response01 import load_joint_setup
from n01.global_memory01 import MeanHeadRepair01, GlobalRepairResponse01
from n01.mean_trend01 import ConditionedMeanRepair01

root = Path('/home/rylan/projects/ej276d-n01/mean_trend_repair_fix01_20261021')
p = json.loads((root / 'configs/mean_trend_repair_fix01_20261021.json').read_bytes())
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
device = torch.device('cuda:0')
counts = Counter()
with torch.no_grad():
    setup = load_joint_setup(p, device, counts)
    enc, base = setup['encoding'], setup['baseline']
    with np.load(p['roles_file'], allow_pickle=False) as z:
        parents = torch.as_tensor(z['source_train_rows'].copy(), device=device)
    source = enc['source']
    centered = source - source[:, :100].mean(1, keepdim=True)
    unit = centered / centered.amax(1, keepdim=True)
    filtered = torch.cat([base.response_filter(unit[s:s + 128, None]) for s in range(0, len(unit), 128)])
    with np.load(p['candidate_arrays'][0], allow_pickle=False) as z:
        theta = torch.as_tensor(z['candidate'].copy(), device=device)
    inherited = GlobalRepairResponse01(setup['joint_models'][0], theta, MeanHeadRepair01(base, enc, filtered))
    saved = torch.load(p['replay_checkpoint'], map_location=device, weights_only=False)
    counts['repair_checkpoint_loads'] += 1
    inherited.load_state_dict(saved['model'], strict=True)
    mean = ConditionedMeanRepair01(base, enc, filtered, inherited.mean)
    target = torch.cat([inherited.mean(r) for r in parents.split(128)])
    result = {}
    for chunk in [64, 128]:
        prediction = torch.cat([mean.uncached(r) for r in parents.split(chunk)])
        result[str(chunk)] = dict(max_abs_V=float((prediction - target).abs().max()),
            RMS_V=float((prediction - target).square().mean().sqrt()))
    captured = []
    hook = base.unet.register_forward_pre_hook(lambda m, a: captured.append(a))
    rows = parents[:128]
    base.signal_forward(source[rows], {k: torch.zeros((len(rows),) + s, device=device) for k, s in base.noise_shapes.items()})
    hook.remove()
    features, scalars = captured[0]
    decoded, _ = mean.unet(features, scalars)
    result['inputs'] = dict(features_max=float((features - mean.features[rows]).abs().max()),
        scalars_max=float((scalars - mean.scalars[rows]).abs().max()),
        cached_decoded_max=float((decoded - enc['decoded'][rows]).abs().max()))
    scalars_original = mean.scalars.clone()
    mean.scalars[rows] = scalars
    prediction = mean.uncached(rows)
    result['replayed_original_scalars'] = dict(max_abs_V=float((prediction - inherited.mean(rows)).abs().max()))
    mean.scalars.copy_(scalars_original)
    result.update(execution_counts=dict(counts), conditioned_mean_rows=mean.forward_rows,
        extra_baseline_rows=128, optimizer_updates=0, NAS_file_IO_checks=False, NAS_file_hash_verification=False)
(root / 'IDENTITY_DIAGNOSTIC.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
