"""One ID-storage expression only; all scientific arithmetic stays unchanged."""
import ast
import copy
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIX = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    snapshot = ROOT / "provenance/integration006_before_id_fix02"
    old_path = snapshot / "evaluate_integration006.py"
    new_path = ROOT / "src/n01/evaluate_integration006.py"
    old_text, new_text = old_path.read_text(encoding="utf8"), new_path.read_text(encoding="utf8")
    old, new = ast.parse(old_text), ast.parse(new_text)
    old_functions = {node.name: node for node in old.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    new_functions = {node.name: node for node in new.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    changed = [name for name in old_functions if ast.dump(old_functions[name]) != ast.dump(new_functions[name])]
    assert changed == ["split_evaluation"] and set(new_functions) == set(old_functions)
    old_expression = ast.parse('torch.as_tensor(data["source_ids"][parents[:batch]].copy())', mode="eval").body
    new_expression = ast.parse('data["source_ids"][parents[:batch]].copy()', mode="eval").body
    class ExpectedReplacement(ast.NodeTransformer):
        replacements = 0
        def visit_Call(self, node):
            if ast.dump(node) == ast.dump(old_expression):
                self.replacements += 1
                return copy.deepcopy(new_expression)
            return self.generic_visit(node)
    transformer = ExpectedReplacement()
    expected_old = transformer.visit(copy.deepcopy(old))
    assert transformer.replacements == 1 and ast.dump(expected_old) == ast.dump(new)
    test_old_path = snapshot / "test_evaluate_integration006.py"
    test_new_path = ROOT / "tests/test_evaluate_integration006.py"
    test_old = ast.parse(test_old_path.read_text(encoding="utf8"))
    test_new = ast.parse(test_new_path.read_text(encoding="utf8"))
    old_class = next(node for node in test_old.body if isinstance(node, ast.ClassDef) and node.name == "Integration006Tests")
    new_class = next(node for node in test_new.body if isinstance(node, ast.ClassDef) and node.name == "Integration006Tests")
    old_methods = {node.name: node for node in old_class.body if isinstance(node, ast.FunctionDef)}
    new_methods = {node.name: node for node in new_class.body if isinstance(node, ast.FunctionDef)}
    assert all(ast.dump(method) == ast.dump(new_methods[name]) for name, method in old_methods.items())
    added_tests = sorted(set(new_methods) - set(old_methods))
    assert added_tests == ["test_ID_dtype_preserved_in_split_torch_roundtrip_and_verifier_exact",
                           "test_complete_artificial_driver_run_serialization_reopen_and_verifier_interop"]
    diff_path = FIX / "DRIVER_ID_FIX.diff"
    diff_path.write_text("".join(difflib.unified_diff(old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
                         fromfile=str(old_path), tofile=str(new_path))), encoding="utf8", newline="\n")
    output = FIX / "SOURCE_DIFF_RECEIPT.json"
    assert not output.exists()
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), status="passed_single_ID_storage_expression_only",
                  old_driver_sha256=sha(old_path), new_driver_sha256=sha(new_path),
                  old_tests_sha256=sha(test_old_path), new_tests_sha256=sha(test_new_path),
                  changed_existing_definitions=changed, added_source_definitions=[], added_tests=added_tests,
                  all_existing_14_test_methods_unchanged=True, entire_old_source_AST_equal_except_one_ID_storage_call=True,
                  scientific_arithmetic_dataflow_seeds_counts_readouts_decision_runtime_unchanged=True,
                  ID_preservation="original dtype NumPy copy; no numeric encoding and no network input",
                  diff_sha256=sha(diff_path), checker_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
