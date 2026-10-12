"""Record the AST-verified ADC QA scope; no scientific arrays or replay."""
from datetime import datetime,timezone
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
FIX=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old=ROOT/'scripts/verify_integration006_numeric_fix01.py'
    new=ROOT/'scripts/verify_integration006_adc_fix02.py'
    tests=ROOT/'tests/test_verify_integration006_adc_fix02.py'
    assert sha(old)=='6de11280961ade7d15b222d5fcc4362938a2503616eb81b1f879bc5caa878ed9'
    spec=importlib.util.spec_from_file_location('ADC_scope_AST_checks',tests)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    case=module.ADCRepresentationFixTests('test_AST_only_ADC_helper_per_view_baseline_reference_and_receipt_description')
    case.test_AST_only_ADC_helper_per_view_baseline_reference_and_receipt_description()
    diff=FIX/'VERIFIER_ADC_FIX.diff'
    diff.write_text(''.join(difflib.unified_diff(old.read_text(encoding='utf8').splitlines(keepends=True),
        new.read_text(encoding='utf8').splitlines(keepends=True),fromfile=str(old),tofile=str(new))),encoding='utf8',newline='\n')
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='passed_ADC_QA_AST_scope_only',
        previous_verifier_sha256=sha(old),new_verifier_sha256=sha(new),tests_sha256=sha(tests),
        added_helper='check_independent_ADC_codes',all_previous_functions_except_replay_all_and_main_AST_identical=True,
        entire_previous_module_AST_restored_by_removing_helper_all_view_call_baseline_reference_and_receipt_field=True,
        ADC_helper_called_first_in_every_all_five_views_loop=True,
        baseline_reference='cpu(digitize(prefix,ste=False))[:,0] on the same device',
        only_receipt_field_added='independent_ADC_code_checks',generation_statistics_decision_gates_theory_counts_unchanged=True,
        diff_sha256=sha(diff),checker_sha256=sha(Path(__file__)))
    output=FIX/'SOURCE_DIFF_RECEIPT.json'
    assert not output.exists()
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(result))


if __name__=='__main__':main()
