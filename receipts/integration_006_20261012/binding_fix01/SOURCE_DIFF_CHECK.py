"""Prove the engineering fix changed only exact loss-metadata binding."""
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
    snapshot = ROOT / "provenance/integration006_failed_binding"
    old_path = snapshot / "evaluate_integration006.py"
    new_path = ROOT / "src/n01/evaluate_integration006.py"
    old_text, new_text = old_path.read_text(encoding="utf8"), new_path.read_text(encoding="utf8")
    old, new = ast.parse(old_text), ast.parse(new_text)
    old_functions = {node.name: node for node in old.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    new_functions = {node.name: node for node in new.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    changed = [name for name in old_functions if ast.dump(old_functions[name]) != ast.dump(new_functions[name])]
    added = sorted(set(new_functions) - set(old_functions))
    assert changed == ["frozen_models"] and added == ["metadata_json_bytes"]
    old_expression = ast.parse('state["loss_statistics"] == stats["loss"]', mode="eval").body
    new_expression = ast.parse('metadata_json_bytes(state["loss_statistics"]) == metadata_json_bytes(stats["loss"])', mode="eval").body
    class ExpectedReplacement(ast.NodeTransformer):
        replacements = 0
        def visit_Compare(self, node):
            if ast.dump(node) == ast.dump(old_expression):
                self.replacements += 1
                return copy.deepcopy(new_expression)
            return self.generic_visit(node)
    transformer = ExpectedReplacement()
    expected_old = transformer.visit(copy.deepcopy(old))
    assert transformer.replacements == 1
    new_without_helper = copy.deepcopy(new)
    new_without_helper.body = [node for node in new_without_helper.body
                               if not isinstance(node, ast.FunctionDef) or node.name != "metadata_json_bytes"]
    assert ast.dump(expected_old) == ast.dump(new_without_helper)
    test_old_path = snapshot / "test_evaluate_integration006.py"
    test_new_path = ROOT / "tests/test_evaluate_integration006.py"
    test_old = ast.parse(test_old_path.read_text(encoding="utf8"))
    test_new = ast.parse(test_new_path.read_text(encoding="utf8"))
    old_class = next(node for node in test_old.body if isinstance(node, ast.ClassDef) and node.name == "Integration006Tests")
    new_class = next(node for node in test_new.body if isinstance(node, ast.ClassDef) and node.name == "Integration006Tests")
    old_methods = {node.name: node for node in old_class.body if isinstance(node, ast.FunctionDef)}
    new_methods = {node.name: node for node in new_class.body if isinstance(node, ast.FunctionDef)}
    assert all(ast.dump(method) == ast.dump(new_methods[name]) for name, method in old_methods.items())
    new_test_names = sorted(set(new_methods) - set(old_methods))
    assert new_test_names == ["test_exact_metadata_rejects_one_ULP_shape_key_and_nonfinite_changes",
                             "test_exact_metadata_tensor_JSON_scalar_vector_float32_float64"]
    diff_path = FIX / "DRIVER_BINDING_FIX.diff"
    diff_path.write_text("".join(difflib.unified_diff(old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
                         fromfile=str(old_path), tofile=str(new_path))), encoding="utf8", newline="\n")
    output = FIX / "SOURCE_DIFF_RECEIPT.json"
    assert not output.exists()
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), status="passed_narrow_AST_change_only",
                  old_driver_sha256=sha(old_path), new_driver_sha256=sha(new_path),
                  old_tests_sha256=sha(test_old_path), new_tests_sha256=sha(test_new_path),
                  changed_existing_definitions=changed, added_definitions=added, added_tests=new_test_names,
                  all_existing_test_methods_unchanged=True, entire_old_source_AST_equal_except_one_metadata_comparison_and_new_helper=True,
                  scientific_dataflow_seeds_counts_readouts_decision_runtime_unchanged=True,
                  diff_sha256=sha(diff_path), checker_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
