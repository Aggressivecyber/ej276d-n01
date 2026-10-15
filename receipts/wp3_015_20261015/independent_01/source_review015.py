"""Source-only 015 independent review record; no numerical imports or input decode."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT / 'receipts/wp3_015_20261015/independent_01'
PINS = {
    'src/n01/model015.py': '149f7faefc9ca982700eea8c7ce7850e9c679d74985cb4fce90708743299dadc',
    'scripts/run_wp3_015.py': '5f688f828fdc563822f3fb28ddf1da2d558654280d4756798bd642f5a6445676',
    'scripts/wp3_controller.py': '6415e2c4c8a6339b0187ff564669adf6b15111015cfc41c0fc31fd71b02b2fcd',
}


def record(rel):
    p = ROOT / rel
    assert p.suffix not in {'.npz', '.npy', '.pt', '.pth', '.ckpt'}
    data = p.read_bytes()
    return dict(path=rel, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {rel: record(rel) for rel in [
        *PINS, 'AGENTS.md', 'USER_MASTER_PLAN_20261015.txt', 'wp3_015_design.json', 'WP3_DATA_ROLES.json',
        'docs/WP3_015_CONFIGURATION.md', 'MASTER_EXECUTION_PLAN.json', 'PROGRAM.json', 'jobs.json',
        'receipts/wp3_015_20261015/implementation_01/READY.json',
        'receipts/wp3_015_20261015/implementation_01/DESIGN.md',
        'receipts/wp3_015_20261015/engineering_01/artificial_cpu_02/ENGINEERING_RECEIPT.json',
        'receipts/wp3_015_20261015/engineering_01/controller_stdlib_03/ENGINEERING_RECEIPT.json',
        'receipts/wp2_014_20261015/collection_01/REMOTE_FULL_INVENTORY.json',
        'WP2_SOURCE_MANIFEST.json', 'wp2_014_config.json', 'scripts/wp2_controller.py']}
    for rel, expected in PINS.items():
        assert evidence[rel]['sha256'] == expected
        ast.parse((ROOT / rel).read_bytes(), filename=rel)
    ready = json.loads((ROOT / 'receipts/wp3_015_20261015/implementation_01/READY.json').read_bytes())
    for rel, expected in PINS.items():
        assert ready['sources'][rel]['sha256'] == expected
    cpu = json.loads((ROOT / 'receipts/wp3_015_20261015/engineering_01/artificial_cpu_02/ENGINEERING_RECEIPT.json').read_bytes())
    ops = json.loads((ROOT / 'receipts/wp3_015_20261015/engineering_01/controller_stdlib_03/ENGINEERING_RECEIPT.json').read_bytes())
    assert len(cpu['checks']) == 9 and len(ops['checks']) == 5 and cpu['status'] == ops['status'] == 'PASS'
    assert all(cpu['sources'][k] == v for k, v in PINS.items() if k in cpu['sources'])
    assert ops['source_SHA256'] == PINS
    design = json.loads((ROOT / 'wp3_015_design.json').read_bytes())
    factory_tree = ast.parse((ROOT / 'scripts/run_wp3_015.py').read_bytes())
    names = {'seed_set015', 'settings015', 'budget_caps015'}
    nodes = [n for n in factory_tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    scope = dict(RUN_ID='wp3_015_20261015', HYPOTHESES=('fast','slow','common_level'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<independent015-pure-factories>', 'exec'), scope)
    settings = scope['settings015']()
    caps = scope['budget_caps015'](settings)
    assert caps == ready['exact_budget_caps'] == cpu['checks'][-1]['budget_caps']
    assert all(caps[k] == value for k, value in design['exact_expected_maximum_counts'].items())
    checks = [
        dict(name='three_READY_source_bytes_and_reused9plus5_engineering_pins', passed=True),
        dict(name='stdlib_AST_factory_exact_maximum_NN_ADC_CP_AD_and_gate_only_budget_identity', passed=True),
        dict(name='M0_C1_continuation1288_and_M1_frozen_same_head_six_zero_parameters', passed=True,
             location='src/n01/model015.py:28-96; scripts/run_wp3_015.py:248-272,391-402'),
        dict(name='SOURCE_centered_canonical110_366_tail156_366_and_original_loss_qref_floor', passed=True,
             location='src/n01/model015.py:111-116; scripts/run_wp3_015.py:258-259'),
        dict(name='Gaussian_sigma2_len13_causal_tau16_len64_zeroQ_differences_and_prefix100', passed=True,
             location='src/n01/model015.py:41-45,117-136'),
        dict(name='gain_bound_RMS_trust_and_Qmu_qbar_qsig_synchronized', passed=True,
             proof='Qmu=(1+trust_scale*c_gain)*Q_A plus separately checked zeroQ residual; |c_gain|<=0.1,0<=trust_scale<=1 implies Qmu>=0.9Q_A for positive A charge.',
             location='src/n01/model015.py:115-136,145-160'),
        dict(name='zero_adapter_IEEE_identity_with_unit_delta_derivative_and_disabled_C1_identity', passed=True,
             location='src/n01/model015.py:17-25,138-153', evidence='CPU02 zero-init all-six nonzero gradients and finite differences'),
        dict(name='random_basis_still_original_A_amplitude_and_no_qsig_product', passed=True,
             location='src/n01/model010.py:189-202; src/n01/model015.py:142-160'),
        dict(name='fresh_arm_AdamW_same_seed_reset_streams_no_old_optimizer_resume', passed=True,
             location='scripts/run_wp3_015.py:262-272,393-402'),
        dict(name='mean_gate_only_REAL_TRAIN_and12_old_REPORT_mean_parent_members_before_CP_NN_random_optimizer', passed=True,
             location='scripts/run_wp3_015.py:167-195,365-389'),
        dict(name='working422_group_sandwich_SE_ALL6_strict_RMS_and_cosine_rule', passed=True,
             location='scripts/run_wp3_015.py:132-164', evidence='CPU02 independent small group-score algebra and negative gate'),
        dict(name='negative_gate_legal_COMPLETE_skip_zero_CP_NN_optimizer_AD_G_no_FIRST_required', passed=True,
             location='scripts/run_wp3_015.py:373-379; scripts/wp3_controller.py:152-158,194-198'),
        dict(name='single_backward_objective_weights_and_three_hypothesis_means_no_probe_or_refit', passed=True,
             location='scripts/run_wp3_015.py:284-343'),
        dict(name='original014_complete_hard_evaluation_flags_all_four_suites_retained', passed=True,
             location='scripts/run_wp3_015.py:345-355,409-419; inherited Engine014.evaluate'),
        dict(name='remote_only12_REPORT_metadata_binding_not_raw_copy_native_SHA_before_decode', passed=True,
             location='scripts/wp3_controller.py:71-86,99-125; inherited runtime_gate; scripts/run_wp3_015.py:108-119,180-187'),
        dict(name='original95plus97_protection_and_lazy_CDF_dependency_closure', passed=True,
             location='scripts/wp3_controller.py:94-145', evidence='controller_stdlib_03 pins35 reachable project files; absent CDF rejected'),
        dict(name='pinned014_stdlib_AST_source_before_import_same_snapshot_health_full_or_skip_schema', passed=True,
             location='scripts/wp3_controller.py:152-217,220-239'),
        dict(name='mature_owned_tree_lock_safe_tar_partial_preservation_no_retry_and512MiB_collection_reused', passed=True,
             location='exact BASE_SHA3480825e... unchanged; inherited wp2_controller runtime_gate/native_start/supervise/archive_inventory/controller_action'),
    ]
    result = dict(schema='n01/wp3-independent-source-review/v1', UTC=datetime.now(timezone.utc).isoformat(),
        status='PASS_SOURCE_AND_METHOD_AWAITING_ROOT_CONFIG_BINDING', passed=True,
        reviewer_role='continuing independent Sol Max reviewer; not implementation author',
        source_SHA256=PINS, evidence=evidence, checks=checks, budget_caps=caps,
        reused_engineering=dict(numeric_cpu_checks=9, controller_stdlib_checks=5,
            no_repeated_Torch_fixture=True, why='The existing actual two-arm artificial step, zero-grad finite differences and source-pinned Store/schema fixtures cover the reviewed new semantics.'),
        semantics=dict(contrast='M0 further C1 shape training versus M1 frozen C1 shape with six deterministic mean parameters',
            mean_gate='registered fixed-population residual rationale; working logical-group SE, not formal conditional-response identification',
            hard_baseline_DC='Signal prefix100 unchanged, so separate electronics baseline mismatch remains a recorded limitation.',
            mean_fixed_B='Mean updates can alter ADC/crop-observed covariance while the preADC stochastic B remains its original-A-based reference.',
            scientific_acceptance=False, weights_promoted=False),
        unresolved_source_blockers=[], pending_task_steps=['Root formal config literal/input/contract binding review',
            'Root source freeze followed by stdlib package SHA bridge', 'Root native first G1 plus same-snapshot health, or valid gate-skip native0'],
        native_resource_and_real_graph_execution_verified=False,
        current_review_operations=dict(NumPy_imports=0,Torch_imports=0,original_array_reads=0,trained_CP_reads=0,
            real_NN_rows=0,real_AD_queries=0,synthetic_NN_rows=0,synthetic_AD_queries=0,training_updates=0,SSH=0),
        source_changes_by_reviewer=0, scientific_launch_authorized_by_this_receipt=False)
    path = OUT / 'SOURCE_REVIEW.json'
    with path.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],checks=len(checks),path=str(path),sha256=record(str(path.relative_to(ROOT)))['sha256']),ensure_ascii=False))


if __name__ == '__main__':
    main()
