"""Bind the user-authorized continuation to a complete saved optimizer boundary."""
from pathlib import Path
import copy
import datetime as dt
import json
import shutil
from window008_fix01_runtime import ROOT,read,sha,write_new,require,validate_evidence

def main():
    fix,base,_=validate_evidence()
    pause=read(ROOT/'receipts/window008_fix01_20261013/USER_PAUSE_RECEIPT.json')
    require(pause['status']=='paused_by_user_before_server_shutdown' and pause['periodic_checkpoint']['G_updates']==6000,'Pause anchor differs')
    root=ROOT/'provenance/window008_resume01_reference';root.mkdir(exist_ok=False)
    old=ROOT/'runs/window008_fix01_20261013/runs'
    sources=dict(fix01_full_periodic_06000=old/'science/train/full/latest.pt',
        fix01_native_exit=old/'supervisor/exit_receipt.json',fix01_launch=old/'supervisor/launch.json',
        fix01_binding=old/'science/BINDING.json',fix01_training_inputs=old/'science/TRAINING_INPUTS.json',
        fix01_resume_validation=old/'science/RESUME_VALIDATION.json',fix01_failure=old/'science/FAILURE.json',
        fix01_user_stop_request=old/'USER_STOP_REQUEST.json',fix01_user_stop_verification=old/'USER_STOP_VERIFICATION.json',
        fix01_user_stop_verification_local=ROOT/'receipts/window008_fix01_20261013/USER_STOP_VERIFICATION.json',
        fix01_user_stop_collection=ROOT/'receipts/window008_fix01_20261013/USER_STOP_COLLECTION.json',
        fix01_user_pause=ROOT/'receipts/window008_fix01_20261013/USER_PAUSE_RECEIPT.json',
        fix01_legacy_validation=old/'science/LEGACY_VALIDATION.json')
    refs=copy.deepcopy(fix['reference_inputs'])
    for name,source in sources.items():
        path=root/(name+source.suffix);shutil.copy2(source,path)
        require(sha(path)==sha(source),'Reference copy differs: '+name)
        refs[name]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size,
                       original=source.relative_to(ROOT).as_posix())
    require(refs['fix01_full_periodic_06000']['sha256']=='921ec4c58387a5ad53fdf55ade10169930536106141e9070f5e4a8df60ab6c58','Boundary checkpoint differs')
    cfg=dict(schema='n01/window008-resume01-config/v1',run_id='window008_resume01_20261013',
        registered_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),remote_root='/home/rylan/projects/ej276d-n01/window008_resume01_20261013',
        base_configuration='window008_config.json',base_configuration_sha256=sha(ROOT/'window008_config.json'),
        legacy_source_manifest='WINDOW008_SOURCE_MANIFEST.json',legacy_source_manifest_sha256=sha(ROOT/'WINDOW008_SOURCE_MANIFEST.json'),
        fix01_configuration='window008_fix01_config.json',fix01_configuration_sha256=sha(ROOT/'window008_fix01_config.json'),
        fix01_source_manifest='WINDOW008_FIX01_SOURCE_MANIFEST.json',fix01_source_manifest_sha256=sha(ROOT/'WINDOW008_FIX01_SOURCE_MANIFEST.json'),
        source_manifest='WINDOW008_RESUME01_SOURCE_MANIFEST.json',protocol='docs/WINDOW008_RESUME01_EXECUTION.md',
        model_implementation='n01.model008_fix01.N01Response008Fix01',model_loss_data_seed_readout_changes=False,
        effective_observation_samples=256,internal_carrier_samples=384,reference_inputs=refs,
        execution_arms=['full','simple'],comparison_arms=['deterministic','full','simple'],updates_per_valid_arm=19000,
        resume=dict(arm='full',reference='fix01_full_periodic_06000',G_updates=6000,D_updates_per_hypothesis=3000,
                    stage='joint_stochastic',next_G_update=6001,all_optimizer_sampler_stream_states_imported=True,
                    first_update_audit=True,first_update_same_version_replay=True),
        fresh_simple=copy.deepcopy(fix['fresh_simple']),retained_deterministic=copy.deepcopy(fix['retained_deterministic']),
        new_execution_G_updates=32000,new_execution_D_updates=87000,
        inherited_effective_G_updates=25000,inherited_effective_D_updates=57000,
        effective_comparison_G_updates=57000,effective_comparison_D_updates=144000,
        historical_actual_G_updates_before_resume=25667,historical_actual_D_updates_before_resume=59001,
        historical_actual_G_updates_after_success=57667,historical_actual_D_updates_after_success=146001,
        global_N01_G_updates_before_resume=125667,
        discarded_legacy_joint_G_updates=269,discarded_legacy_joint_D_updates=807,
        discarded_user_stop_G_updates=398,discarded_user_stop_D_updates=1194,
        noise_updates=0,population_updates=0,C2ST_updates=0,DEV_waveforms_used_for_training=False,
        independent_acquisition_available=False,cpu_threads=12,gpu_memory_limit_GiB=10,job_tree_RSS_limit_GiB=8,
        wall_limit_s=21600,output_limit_MiB=3072,checkpoint_interval=1000,status_interval=25,
        one_active_scientific_job=True,scientific_status='response_match_not_established',
        user_resume_authorization='Direct user message on 2026-10-13: server is back; proceed. Child agents requested GPT-6.1-Sol Max.',
        supervision='Verify correct scientific startup once, then schedule one estimated-completion wakeup and rest.')
    write_new(ROOT/'window008_resume01_config.json',cfg)
    print(json.dumps(dict(status='resume_configuration_prepared',references=len(refs),config_sha256=sha(ROOT/'window008_resume01_config.json'))))

if __name__=='__main__':main()
