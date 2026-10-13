"""Stage read-only copies of prepared TRAIN artifacts and register byte bindings."""
from pathlib import Path
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from n01.data import sha256,write_json


def main():
    source=ROOT/'runs/window008_20261013/prepared'
    target=ROOT/'data/window008'
    if target.exists():raise FileExistsError('Preserve a previous runtime staging attempt')
    target.mkdir(parents=True)
    for name in ('DATASET008.npz','FIT_STATISTICS008.json','FIT_DATA_STATISTICS.json','PREPARATION_RECEIPT.json','STATISTICS_RECEIPT.json'):
        shutil.copy2(source/name,target/name)
        if sha256(source/name)!=sha256(target/name):raise ValueError('Staged copy differs: '+name)
    path=ROOT/'window008_config.json'
    cfg=json.loads(path.read_text(encoding='utf8'))
    cfg['dataset_sha256']=sha256(target/'DATASET008.npz')
    cfg['statistics_sha256']=sha256(target/'FIT_STATISTICS008.json')
    cfg['evaluation']['matched_DEV_per_class']=256
    # Training source and evaluator have separate freezes. Evaluation cannot
    # choose a checkpoint or change the already frozen training objective.
    cfg['evaluation']['execution']='separate_frozen_job_after_all_three_final_checkpoints_and_native_training_exit'
    write_json(path,cfg)
    record=dict(status='read_only_runtime_copies_verified',dataset_sha256=cfg['dataset_sha256'],
        statistics_sha256=cfg['statistics_sha256'],config_sha256=sha256(path),
        data_role='all_legal_TRAIN_waveforms_plus_DEV_identity_only',DEV_waveforms_decoded=False)
    write_json(ROOT/'receipts/window008_20261013/RUNTIME_INPUT_BINDING.json',record)
    print(json.dumps(record))


if __name__=='__main__':main()
