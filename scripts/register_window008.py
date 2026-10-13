"""Register the user's256-window replacement before new model training."""
import json
from pathlib import Path
import sys
import datetime as dt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from n01.data008 import DEFAULT_SPEC
from n01.data import write_json


def main():
    path=ROOT/'window008_config.json'
    if path.exists():raise FileExistsError(path)
    old=json.loads((ROOT/'response007_config.json').read_text(encoding='utf8'))
    cfg={k:old[k] for k in ('local_training_container','remote_training_container','train_container_sha256','identities_sha256','hypotheses')}
    cfg.update(run_id='window008_20261013',registered_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        remote_root='/home/rylan/projects/ej276d-n01/window008_20261013',
        user_authorization='User requested shorter256 effective window and parallel Astra Max Fast project work on2026-10-13.',
        window=DEFAULT_SPEC,internal_model_length=384,effective_observation_length=256,
        clean_source_ADC=False,REAL_existing_ADC=True,generated_ADC_once=True,
        source_canonical_charge_slice=[110,366],source_peak_anchor=126,
        scope='Relative effective response in the boundN01 software voltage domain; Gamma effective time extent only, not its complete normalization/resampling chain.',
        dataset='data/window008/DATASET008.npz',statistics='data/window008/FIT_STATISTICS008.json',
        noise_reference=old['references']['noise005'],source_manifest='WINDOW008_SOURCE_MANIFEST.json',
        arms=['deterministic','full','simple'],hypothesis_weights=[1/3]*3,
        response=dict(mean_response_updates=3000,joint_stochastic_updates=12000,low_learning_rate_updates=4000,
            updates_per_arm=19000,total_G_updates=57000,total_D_updates=144000,
            warm_start=False,population_learning=False,common_initial_state=True,independent_D_per_hypothesis=True),
        loss=dict(version='window008',weights=dict(wave=1.,QPSD=2.,noise=1.,adv=.1,regular=.01,trend=.1),
            hypothesis_reduction='mean_of_separate_losses',G_lr=.0002,D_lr=.0001,G_low_lr=.00005,D_low_lr=.000025,
            G_weight_decay=.0001,clip_norm=5.,baseline_noise_gradient_to_response='zero_by_zero_signal_prefix'),
        distribution_microbatches=2,source_parents_per_microbatch=256,responses_per_parent=2,real_rows_per_microbatch=512,
        seeds={name:2026101380+i for i,name in enumerate(old['seeds'])},
        evaluation=dict(draws_per_parent=4,chunk_rows=128,matched_TRAIN_per_class=2048,bootstrap_draws=1000,
            sample_waveform_rows=128,minimum_parents_per_bin=32,
            validation_policy='No DEV numeric decode during preparation or training. A once-frozen post-training DEV evaluation is reported separately and never selects settings.',
            training_diagnostics=True,DEV_evaluation_after_all_arms=True,C2ST_updates=0,
            scientific_pass_threshold=None,no_GMM_fits=True),
        engineering=dict(signal_prefix_max_V=1e-12,nonfinite_stops=True,nonpositive_signal_Q_stops=True,
            nonpositive_observed_Q_retained=True,guarded_PSD_retained=True,
            search_edge_and_global_outside_flags='keep_count_report_no_row_selection_no_search_adaptation',
            waveform_replay_first_steps=2,gradient_audit_interval=1000),
        cpu_threads=12,gpu_memory_limit_GiB=10,job_tree_RSS_limit_GiB=8,
        wall_limit_s=28800,output_limit_MiB=3072,checkpoint_interval=1000,status_interval=25,
        one_active_scientific_job=True,noise_updates=0,population_updates=0,
        independent_acquisition_available=False,validation_previously_exposed=True,
        scientific_status='response_match_not_established',review_mode='Astra_Max_parallel_engineering_and_primary_integration')
    write_json(path,cfg)
    print(json.dumps(dict(path=str(path),run_id=cfg['run_id'],G_budget=57000,D_budget=144000)))


if __name__=='__main__':main()
