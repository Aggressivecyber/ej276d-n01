"""Run a pinned research API on explicitly prepared clean SOURCE carriers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]

import numpy as np
import torch
from n01.finalization.api import load_response
from n01.finalization.output import save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--input',type=Path,required=True,
        help='NPZ containing float64 source_waveforms[P,384] in V and parent_ids[P]')
    parser.add_argument('--input-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--model-seed',type=int,choices=(0,1),required=True)
    parser.add_argument('--noise-hypothesis',choices=('fast','slow','common_level'),required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--gpu-gib',type=float,default=6.)
    parser.add_argument('--contributions',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path,digest in ((args.config,args.config_sha256),(args.input,args.input_sha256)):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Pinned input bytes changed: '+str(path))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_gib*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    with np.load(args.input,allow_pickle=False) as archive:
        source=archive['source_waveforms'].copy();ids=archive['parent_ids'].copy()
    if ids.ndim!=1 or ids.dtype.kind not in ('U','S','i','u'):
        raise ValueError('parent_ids must be a one-dimensional string or integer array')
    response=load_response(args.config,args.device)
    result=response.sample(torch.from_numpy(source),ids,args.draws,args.seed,
        args.model_seed,args.noise_hypothesis,True)
    # Complete native ADC/preADC are retained; optional component arrays are larger.
    if not args.contributions:
        del result['contributions']
    record=save_result(args.output,result,dict(config_sha256=args.config_sha256,
        input_sha256=args.input_sha256,operation_accounting=response.accounting,
        scientific_acceptance=False))
    print(json.dumps(dict(status=record['status'],rows=record['rows'],parents=record['parents'],
        output_sha256=record['output_sha256'],scientific_acceptance=False)))


if __name__=='__main__':
    main()
