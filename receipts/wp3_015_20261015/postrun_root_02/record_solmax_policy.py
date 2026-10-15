"""Apply the user's latest model instruction to mutable continuation metadata."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
CURRENT = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    policy_path = HERE / 'AGENT_MODEL_POLICY_SOLMAX.json'
    assert not policy_path.exists()
    paths = [ROOT / p for p in ('MASTER_EXECUTION_PLAN.json', 'PROGRAM.json', 'jobs.json')] + [CURRENT]
    snapshot = HERE / 'metadata_before_solmax_policy'
    snapshot.mkdir(exist_ok=False)
    for p in paths:
        shutil.copy2(p, snapshot / p.name)
    policy = dict(schema='n01/agent-model-policy/v2', recorded_UTC=now,
        authority='latest direct user instruction: 6.1solmax',
        model='gpt-6.1-sol', reasoning_effort='max',
        roles=['planning', 'implementation', 'independent_review', 'future_dispatch'],
        supersedes_model='gpt-6-astra', supersedes_reasoning_effort='max',
        preserved_historical_authorship=True,
        old_Astra_agents_status='interrupted; do not resume',
        new_implementation_agent='/root/wp3_015_solmax_execution',
        new_independent_review_agent='/root/wp3_015_solmax_review',
        new_agents_model_and_effort_explicit_in_spawn=True,
        root_runtime_model_independently_verified=False,
        future_followup_prompt_must_use_this_policy=True)
    write(policy_path, policy)
    reference = dict(path=policy_path.relative_to(ROOT).as_posix(),
        bytes=policy_path.stat().st_size, sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest())
    data = [json.loads(p.read_bytes()) for p in paths]
    for d in data:
        if 'agent_model_policy' in d:
            d['previous_agent_model_policy'] = d['agent_model_policy']
        d.update(agent_model_policy=reference, latest_subagent_model_preference='GPT-6.1-Sol Max', updated_UTC=now)
    write(paths[0], data[0])
    mh = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    data[1]['master_plan_sha256'] = mh
    data[3]['master_plan_sha256'] = mh
    for p, d in zip(paths[1:], data[1:]):
        write(p, d)
    print(json.dumps(dict(status='latest_user_policy_applied', model=policy['model'], reasoning_effort='max', policy=reference)))


if __name__ == '__main__':
    main()
