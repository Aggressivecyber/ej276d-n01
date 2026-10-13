"""Fit256 response/loss scales to all prepared legal TRAIN and bind their bytes."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from n01.data import sha256,write_json
from n01.losses008 import fit_loss_statistics


def clean(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x


def main():
    torch.set_num_threads(8)
    folder=ROOT/'runs/window008_20261013/prepared'
    path=folder/'FIT_STATISTICS008.json'
    if path.exists():raise FileExistsError(path)
    data_path=folder/'DATASET008.npz';source_stats=folder/'FIT_DATA_STATISTICS.json'
    receipt=json.loads((folder/'PREPARATION_RECEIPT.json').read_text(encoding='utf8'))
    assert sha256(data_path)==receipt['output_data_sha256']
    assert sha256(source_stats)==receipt['output_statistics_sha256']
    began=time.monotonic()
    with np.load(data_path,allow_pickle=False) as z:
        loss=fit_loss_statistics(torch.from_numpy(z['real']),torch.from_numpy(z['source']),torch.from_numpy(z['real_baseline100']))
        counts=dict(SOURCE=len(z['source']),REAL=len(z['real']))
    result=dict(schema='n01/window008/fit_statistics/v1',model=json.loads(source_stats.read_text(encoding='utf8'))['model'],
        loss=loss,prepared_dataset_sha256=sha256(data_path),data_statistics_sha256=sha256(source_stats),
        statistics_split='TRAIN',all_train_counts=counts,DEV_waveforms_decoded=False)
    write_json(path,clean(result))
    reopened=json.loads(path.read_text(encoding='utf8'))
    assert reopened==clean(result)
    report=dict(status='all_TRAIN_model_and_loss_statistics_reopened_verified',elapsed_seconds=time.monotonic()-began,
        statistics_sha256=sha256(path),loss_source_sha256=sha256(ROOT/'src/n01/losses008.py'),
        output=str(path),prepared_dataset_sha256=sha256(data_path),counts=counts,DEV_waveforms_decoded=False,
        new_statistics=True,prior480_statistics_reused=False,scientific_model_forwards=0,scientific_optimizer_updates=0,
        q_floor_V_ns=reopened['loss']['q_floor'],view_dimensions={k:len(v['center']) for k,v in reopened['loss']['views'].items()})
    write_json(folder/'STATISTICS_RECEIPT.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
