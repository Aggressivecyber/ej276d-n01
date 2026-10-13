"""Bind immutable legacy states and register one structural recovery recipe."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):digest.update(part)
    return digest.hexdigest()


def main():
    cfg_path=ROOT/'window008_fix01_config.json'
    reference=ROOT/'provenance/window008_fix01_reference'
    if cfg_path.exists() or reference.exists():raise FileExistsError('Preserve an earlier recovery registration')
    base=ROOT/'window008_config.json';base_cfg=json.loads(base.read_text(encoding='utf8'))
    collection=json.loads((ROOT/'receipts/window008_20261013/FAILURE_COLLECTION.json').read_text(encoding='utf8'))
    if collection['failure']['G_updates']!=22269 or collection['failure']['D_updates']!=48807 or collection['native']['exit_code']!=1:
        raise ValueError('Recovery reference is not the bound failed execution')
    mapping=dict(common_initial_state='runs/science/common_initial_state.pt',
        full_stage_03000='runs/science/train/full/stage_03000.pt',
        full_initial='runs/science/train/full/initial.pt',
        deterministic_final='runs/science/train/deterministic/final.pt',
        deterministic_stage_03000='runs/science/train/deterministic/stage_03000.pt',
        deterministic_training_complete='runs/science/train/deterministic/TRAINING_COMPLETE.json',
        legacy_failure='runs/science/FAILURE.json',legacy_native_exit='runs/supervisor/exit_receipt.json',
        legacy_binding='runs/science/BINDING.json',legacy_training_inputs='runs/science/TRAINING_INPUTS.json',
        legacy_launch='runs/supervisor/launch.json',
        legacy_failure_checkpoint='runs/science/FAILURE_CHECKPOINT.pt',legacy_failure_context='runs/science/FAILURE_CONTEXT.pt')
    reference.mkdir(parents=True)
    inputs={}
    for name,original in mapping.items():
        path=ROOT/'runs/window008_20261013'/original;record=collection['files'][original]
        if sha(path)!=record['sha256']:raise ValueError('Historical checkpoint changed: '+original)
        destination=reference/(name+path.suffix)
        shutil.copy2(path,destination)
        if sha(destination)!=record['sha256']:raise ValueError('Recovery copy hash mismatch: '+name)
        inputs[name]=dict(path=destination.relative_to(ROOT).as_posix(),sha256=record['sha256'],
                          bytes=record['bytes'],original=original)
    cfg=dict(schema='n01/window008-fix01-config/v1',run_id='window008_fix01_20261013',
        registered_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        remote_root='/home/rylan/projects/ej276d-n01/window008_fix01_20261013',
        base_configuration='window008_config.json',base_configuration_sha256=sha(base),
        legacy_source_manifest='WINDOW008_SOURCE_MANIFEST.json',legacy_source_manifest_sha256=sha(ROOT/'WINDOW008_SOURCE_MANIFEST.json'),
        source_manifest='WINDOW008_FIX01_SOURCE_MANIFEST.json',protocol='docs/WINDOW008_STABILITY_FIX01_PROTOCOL.md',
        model_implementation='n01.model008_fix01.N01Response008Fix01',
        model_change='full joint/low only: gated raw stochastic bases receive the existing canonical-charge projection, then RMS normalization; base guards remain',
        effective_observation_samples=256,internal_carrier_samples=384,
        reference_inputs=inputs,execution_arms=['full','simple'],comparison_arms=base_cfg['arms'],
        resume=dict(arm='full',reference='full_stage_03000',G_updates=3000,D_updates_per_hypothesis=0,
                    stage='mean_response',next_G_update=3001,all_optimizer_sampler_stream_states_imported=True,
                    first_update_audit=True,first_update_same_version_replay=True),
        fresh_simple=dict(common_reference='common_initial_state',G_updates=0,D_updates_per_hypothesis=0),
        retained_deterministic=dict(final_reference='deterministic_final',G_updates=19000,D_updates_per_hypothesis=16000,
                                    no_new_updates=True,source_execution_native_exit_code=1,arm_training_complete=True),
        updates_per_valid_arm=19000,new_execution_G_updates=35000,new_execution_D_updates=96000,
        effective_comparison_G_updates=57000,effective_comparison_D_updates=144000,
        historical_actual_G_updates_after_success=57269,historical_actual_D_updates_after_success=144807,
        discarded_legacy_joint_G_updates=269,discarded_legacy_joint_D_updates=807,
        noise_updates=0,population_updates=0,C2ST_updates=0,DEV_waveforms_used_for_training=False,
        loss_and_seeds_changed=False,independent_acquisition_available=False,
        cpu_threads=base_cfg['cpu_threads'],gpu_memory_limit_GiB=base_cfg['gpu_memory_limit_GiB'],
        job_tree_RSS_limit_GiB=base_cfg['job_tree_RSS_limit_GiB'],wall_limit_s=21600,
        output_limit_MiB=base_cfg['output_limit_MiB'],checkpoint_interval=base_cfg['checkpoint_interval'],
        status_interval=base_cfg['status_interval'],one_active_scientific_job=True,
        supervision='owned resource guard only; primary verifies correct startup then schedules one estimated-completion follow-up',
        scientific_status='response_match_not_established')
    with cfg_path.open('x',encoding='utf8') as stream:stream.write(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='structural_recovery_inputs_bound',config=str(cfg_path),
                          config_sha256=sha(cfg_path),references=len(inputs),new_G_budget=35000,new_D_budget=96000)))


if __name__=='__main__':main()
