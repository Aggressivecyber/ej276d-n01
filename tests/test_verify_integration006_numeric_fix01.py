"""Artificial float64 theory checks; no scientific arrays/weights/replay."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'scripts/verify_integration006.py'
NEW=ROOT/'scripts/verify_integration006_numeric_fix01.py'
spec=importlib.util.spec_from_file_location('verify006_numeric_artificial',NEW)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def artificial_theory():
    """Explicit SPD block kernel with exactly representable dimensionless A."""
    A=np.zeros((380,100),dtype=np.float64);A[:,1]=.25;A[:,99]=.125
    KBB=np.eye(100,dtype=np.float64)*1e-7
    KTB=A@KBB
    K=np.zeros((480,480),dtype=np.float64)
    K[:100,:100]=KBB;K[100:,:100]=KTB;K[:100,100:]=KTB.T
    K[100:,100:]=A@KBB@A.T+np.eye(380)*1e-7
    independent={h:dict(A=A.copy(),kernel_reference=K.copy(),mean=np.zeros(480),
                 covariance=np.eye(480)*1e-7,variance_Q=np.array(0.),prefix_increment_start=np.arange(90,99,dtype=np.int64))
                 for h in v.HYPOTHESES}
    saved={h+'_'+key:value.copy() for h,fields in independent.items() for key,value in fields.items()}
    return saved,independent


class NumericTheoryFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v.torch.set_num_threads(2)
        cls.old_source_hash=v.sha(OLD)

    @classmethod
    def tearDownClass(cls):
        if v.sha(OLD)!=cls.old_source_hash:
            raise AssertionError('Original frozen verifier source changed')

    def test_machine_precision_near_zero_A_difference_passes_with_backward_error(self):
        saved,independent=artificial_theory()
        difference=6*np.finfo(np.float64).eps
        for h in v.HYPOTHESES:
            saved[h+'_A'][0,0]=difference
        # Demonstrate the exact prior check would fail this dimensionless0.
        with self.assertRaises(AssertionError):
            v.close(saved['fast_A'],independent['fast']['A'],atol=1e-15,rtol=3e-10)
        with patch.object(v,'read_arrays',side_effect=AssertionError('No scientific read')), \
             patch.object(v,'load_models',side_effect=AssertionError('No checkpoint')), \
             patch.object(v,'replay_all',side_effect=AssertionError('No replay')):
            result=v.check_theory_numeric(saved,independent)
        self.assertEqual(result['A_absolute_floor'],64*np.finfo(np.float64).eps)
        self.assertEqual(result['relative_tolerance'],3e-10)
        self.assertEqual(result['other_theory_absolute_tolerance'],1e-15)
        for h in v.HYPOTHESES:
            self.assertEqual(result['hypotheses'][h]['A_max_absolute_difference'],difference)
            self.assertLessEqual(result['hypotheses'][h]['saved_A_backward_error'],64*np.finfo(np.float64).eps)
            self.assertLessEqual(result['hypotheses'][h]['independent_A_backward_error'],64*np.finfo(np.float64).eps)

    def test_significant_near_zero_A_perturbation_fails_forward_comparison(self):
        saved,independent=artificial_theory()
        saved['fast_A'][0,0]=2*(64*np.finfo(np.float64).eps)
        with self.assertRaises(AssertionError):
            v.check_theory_numeric(saved,independent)

    def test_backward_error_rejects_perturbation_hidden_by_relative_tolerance_or_saved_kernel(self):
        saved,independent=artificial_theory()
        saved['slow_A'][0,1]+=1e-11
        # This alteration is smaller than the unchanged rtol at A=.25.
        v.close(saved['slow_A'],independent['slow']['A'],atol=64*np.finfo(np.float64).eps,rtol=3e-10)
        # Supply a self-consistent saved K too. In voltage-squared units its
        # change is below the original fieldwise atol; it must not define the
        # reference equation for either A residual check.
        altered_KTB=saved['slow_A']@independent['slow']['kernel_reference'][:100,:100]
        saved['slow_kernel_reference'][100:,:100]=altered_KTB
        saved['slow_kernel_reference'][:100,100:]=altered_KTB.T
        v.close(saved['slow_kernel_reference'],independent['slow']['kernel_reference'],atol=1e-15,rtol=3e-10)
        with self.assertRaisesRegex(AssertionError,'saved A'):
            v.check_theory_numeric(saved,independent)

    def test_independent_A_is_separately_checked_against_registered_kernel(self):
        saved,independent=artificial_theory()
        independent['common_level']['A'][0,1]+=1e-11
        with self.assertRaisesRegex(AssertionError,'independent A'):
            v.check_theory_numeric(saved,independent)

    def test_nonfinite_shape_dtype_and_zero_denominator_are_rejected(self):
        for nonfinite in (np.nan,np.inf,-np.inf):
            saved,independent=artificial_theory()
            saved['fast_A'][0,0]=nonfinite
            with self.subTest(value=nonfinite),self.assertRaises(AssertionError):
                v.check_theory_numeric(saved,independent)
            saved,independent=artificial_theory()
            saved['fast_mean'][0]=nonfinite;independent['fast']['mean'][0]=nonfinite
            with self.subTest(nonfinite_other=nonfinite),self.assertRaises(AssertionError):
                v.check_theory_numeric(saved,independent)
        saved,independent=artificial_theory()
        saved['fast_A']=saved['fast_A'].T
        with self.assertRaises(AssertionError):v.check_theory_numeric(saved,independent)
        with self.assertRaises(AssertionError):
            v.theory_A_backward_error(np.zeros((380,100)),np.zeros((480,480)),label='zero denominator')
        with self.assertRaises(AssertionError):
            v.theory_A_backward_error(independent['fast']['A'].astype(np.float32),independent['fast']['kernel_reference'],label='precision')
        with self.assertRaises(AssertionError):
            v.theory_A_backward_error(independent['fast']['A'],np.zeros((480,479)),label='shape')
        with self.assertRaises(AssertionError):
            v.check_theory_numeric(saved,{'fast':independent['fast']})

    def test_other_analytical_fields_keep_original_strict_absolute_tolerance(self):
        for key in ('mean','variance_Q'):
            saved,independent=artificial_theory()
            if key=='mean':saved['fast_mean'][0]=2e-15
            else:saved['fast_variance_Q']=np.array(2e-15)
            with self.subTest(field=key),self.assertRaises(AssertionError):
                v.check_theory_numeric(saved,independent)
        saved,independent=artificial_theory()
        saved['fast_mean'][0]=.5e-15
        v.check_theory_numeric(saved,independent)

    def test_original_AST_unchanged_except_theory_call_and_diagnostic_receipt_field(self):
        old,new=ast.parse(OLD.read_text(encoding='utf8')),ast.parse(NEW.read_text(encoding='utf8'))
        old_defs={node.name:node for node in old.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        new_defs={node.name:node for node in new.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(set(new_defs)-set(old_defs),{'theory_A_backward_error','check_theory_numeric'})
        for name,node in old_defs.items():
            if name!='main':self.assertEqual(ast.dump(node),ast.dump(new_defs[name]),name)
        normalized=copy.deepcopy(new)
        normalized.body=[node for node in normalized.body if not isinstance(node,ast.FunctionDef) or
                         node.name not in ('theory_A_backward_error','check_theory_numeric')]
        new_main=next(node for node in normalized.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        old_main=old_defs['main']
        old_try=next(node for node in old_main.body if isinstance(node,ast.Try))
        old_theory_loop=next(node for node in old_try.body if isinstance(node,ast.For) and
                            isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Attribute) and
                            isinstance(node.iter.func.value,ast.Name) and node.iter.func.value.id=='expected')
        class RestoreTheoryCall(ast.NodeTransformer):
            calls=0
            fields=0
            def visit_Assign(self,node):
                if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='theory_numeric_checks':
                    self.calls+=1
                    return copy.deepcopy(old_theory_loop)
                return self.generic_visit(node)
            def visit_Call(self,node):
                for keyword in list(node.keywords):
                    if keyword.arg=='theory_numeric_checks':
                        self.fields+=1;node.keywords.remove(keyword)
                return self.generic_visit(node)
        restore=RestoreTheoryCall();restore.visit(new_main)
        self.assertEqual((restore.calls,restore.fields),(1,1))
        self.assertEqual(ast.dump(old),ast.dump(normalized))


if __name__=='__main__':unittest.main()
