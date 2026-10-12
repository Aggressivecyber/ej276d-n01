"""Record the derivative verifier's exact permitted AST changes, no science IO."""
import ast
import copy
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
FIX=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old_path=ROOT/'scripts/verify_integration006.py'
    new_path=ROOT/'scripts/verify_integration006_numeric_fix01.py'
    assert sha(old_path)=='5990b29c1f232329978fe5700b73e9826c346025e465ec76b340213c7e5cced0'
    old_text=old_path.read_text(encoding='utf8');new_text=new_path.read_text(encoding='utf8')
    old,new=ast.parse(old_text),ast.parse(new_text)
    old_defs={node.name:node for node in old.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    new_defs={node.name:node for node in new.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    added=sorted(set(new_defs)-set(old_defs))
    assert added==['check_theory_numeric','theory_A_backward_error']
    assert all(ast.dump(node)==ast.dump(new_defs[name]) for name,node in old_defs.items() if name!='main')
    normalized=copy.deepcopy(new)
    normalized.body=[node for node in normalized.body if not isinstance(node,ast.FunctionDef) or node.name not in added]
    new_main=next(node for node in normalized.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    old_try=next(node for node in old_defs['main'].body if isinstance(node,ast.Try))
    old_loop=next(node for node in old_try.body if isinstance(node,ast.For) and isinstance(node.iter,ast.Call) and
                  isinstance(node.iter.func,ast.Attribute) and isinstance(node.iter.func.value,ast.Name) and node.iter.func.value.id=='expected')
    class Restore(ast.NodeTransformer):
        calls=0
        fields=0
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='theory_numeric_checks':
                self.calls+=1;return copy.deepcopy(old_loop)
            return self.generic_visit(node)
        def visit_Call(self,node):
            for keyword in list(node.keywords):
                if keyword.arg=='theory_numeric_checks':self.fields+=1;node.keywords.remove(keyword)
            return self.generic_visit(node)
    restore=Restore();restore.visit(new_main)
    assert (restore.calls,restore.fields)==(1,1) and ast.dump(old)==ast.dump(normalized)
    diff_path=FIX/'VERIFIER_NUMERIC_FIX.diff'
    diff_path.write_text(''.join(difflib.unified_diff(old_text.splitlines(keepends=True),new_text.splitlines(keepends=True),
                         fromfile=str(old_path),tofile=str(new_path))),encoding='utf8',newline='\n')
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='passed_permitted_verifier_AST_changes_only',
                original_verifier_sha256=sha(old_path),new_verifier_sha256=sha(new_path),added_functions=added,
                all_original_functions_and_classes_except_main_AST_identical=True,
                entire_original_module_AST_restored_by_theory_call_and_one_diagnostic_field=True,
                original_sources_and_science_payload_unchanged=True,scientific_gate_seed_generation_replay_and_update_logic_unchanged=True,
                only_main_changes=['replace prior analytical loop with strict numeric helper','add theory_numeric_checks diagnostic receipt field'],
                diff_sha256=sha(diff_path),checker_sha256=sha(Path(__file__)))
    output=FIX/'SOURCE_DIFF_RECEIPT.json'
    assert not output.exists()
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(result))


if __name__=='__main__':main()
