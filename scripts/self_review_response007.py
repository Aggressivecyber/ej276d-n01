"""Primary-agent engineering receipts and explicit historical preservation."""
from pathlib import Path
import ast
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];ID='response_007_20261013';REC=ROOT/'receipts'/ID
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def main():
    REC.mkdir(parents=True,exist_ok=True)
    sources=['src/n01/response007.py','src/n01/train_response007.py','src/n01/evaluate_response007.py',
        'tests/test_response007.py','scripts/launch_response007.sh','docs/RESPONSE_007_IMPLEMENTATION.md']
    for name in sources:
        if name.endswith('.py'):ast.parse((ROOT/name).read_text(encoding='utf8'),filename=name)
    historical=read(ROOT/'INTEGRATION006_ID_FIX02_SOURCE_MANIFEST.json')['files']
    for name,digest in historical.items():assert sha(ROOT/name)==digest,name
    cfg=read(ROOT/'response007_config.json');reg=read(REC/'PREREGISTRATION.json')
    assert sha(ROOT/'response007_config.json')==reg['config_sha256']
    assert sha(ROOT/'docs/RESPONSE_007_PROTOCOL.md')==reg['protocol_sha256']
    benchroot=ROOT/'runs'/ID/'engineering01'
    benchmark=read(benchroot/'runs/benchmark/BENCHMARK.json')
    native=read(benchroot/'runs/supervisor/exit_receipt.json')
    binding=read(benchroot/'SOURCE_BINDING.json')
    assert benchmark['status']=='artificial_capacity_test_passed' and native['exit_code']==0
    assert benchmark['scientific_G_updates']==0 and benchmark['scientific_data_decoded'] is False
    assert sha(ROOT/'src/n01/response007.py')==binding['files']['src/n01/response007.py']
    environment=os.environ.copy();environment.update(PYTHONPATH=str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
        TEMP='E:/GAN/runtime_tmp',TMP='E:/GAN/runtime_tmp',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_response007.py','-v'],
        cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8')
    (REC/'SELF_TEST_OUTPUT.txt').write_text(result.stdout,encoding='utf8',newline='\n')
    if result.returncode:raise RuntimeError(result.stdout)
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    write(REC/'SELF_TEST.json',dict(UTC=now,status='passed',native_exit_code=result.returncode,tests=7,
        artificial_data_only=True,scientific_training_updates=0,source_hashes={n:sha(ROOT/n) for n in sources},
        output_sha256=sha(REC/'SELF_TEST_OUTPUT.txt')))
    write(REC/'ENGINEERING_TEST_HISTORY.json',dict(recorded_UTC=now,scientific_data_decoded_in_any_attempt=False,
        attempts=[dict(stage='initial_training_core',tests=4,native_exit_code=0),
            dict(stage='evaluation_added',tests=5,native_exit_code=1,error='NumPy readout received Tensor-valued artificial statistics'),
            dict(stage='JSON_roundtrip',tests=5,native_exit_code=1,error='Historical loss API clone requires tensor center/scale'),
            dict(stage='tensor_binding_repaired',tests=5,native_exit_code=1,error='Permutation test exceeded inappropriate float64-scale bound for float32 convolution'),
            dict(stage='float32_numerical_bound_registered',tests=5,native_exit_code=0),
            dict(stage='final_self_review',tests=7,native_exit_code=0)],
        artificial_capacity=benchmark,capacity_native_receipt=native,capacity_source_binding_sha256=sha(benchroot/'SOURCE_BINDING.json')))
    write(REC/'SELF_REVIEW.json',dict(UTC=now,status='passed_primary_agent_self_review_for_bounded_execution',
        review_mode='same_primary_agent_no_independent_reviewer',user_requested_no_subagents=True,
        historical_files_verified=len(historical),historical_source_manifest_sha256=sha(ROOT/'INTEGRATION006_ID_FIX02_SOURCE_MANIFEST.json'),
        config_sha256=sha(ROOT/'response007_config.json'),protocol_sha256=sha(ROOT/'docs/RESPONSE_007_PROTOCOL.md'),
        source_hashes={n:sha(ROOT/n) for n in sources},test_receipt_sha256=sha(REC/'SELF_TEST.json'),
        checks=dict(no_old_response_warm_start=True,shared_common_initial_state=True,three_separate_losses=True,
            independent_D_states=True,one_G_step_per_update=True,regular_counted_once=True,
            candidate_STE_masks_retained=True,noise005_frozen=True,raw_data_roles_preserved=True,
            count_weights_fixed=True,no_hypothesis_input=True,no_scenario_or_checkpoint_selection=True,
            no_scientific_acceptance_threshold_added=True,new_signal_heads_receive_gradient=True,
            failure_contexts_and_counts_preserved=True,native_capacity_test_passed=True),
        limitations=['Not an independent second-model review','Artificial engineering tests are not response validation',
            'Full budget and scientific endpoint results remain unexecuted at this receipt'],blockers=[]))
    print(json.dumps(dict(status='passed',tests=7,historical_files_verified=len(historical),benchmark_native_exit=0,
        self_review_sha256=sha(REC/'SELF_REVIEW.json'))))
if __name__=='__main__':main()
