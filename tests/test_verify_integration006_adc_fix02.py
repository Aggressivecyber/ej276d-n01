"""Artificial ADC code/representation checks; no real arrays, weights, replay."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'scripts/verify_integration006_numeric_fix01.py'
NEW=ROOT/'scripts/verify_integration006_adc_fix02.py'
spec=importlib.util.spec_from_file_location('verify006_adc_artificial',NEW)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def canonical(pre):
    return v.cpu(v.digitize(v.torch.from_numpy(pre),ste=False))


class ADCRepresentationFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v.torch.set_num_threads(2)
        cls.previous_hash=v.sha(OLD)

    @classmethod
    def tearDownClass(cls):
        if v.sha(OLD)!=cls.previous_hash:
            raise AssertionError('Previously signed numeric verifier changed')

    def test_same_integer_code_different_ULP_voltage_passes(self):
        pre=np.linspace(-.4,.4,960,dtype=np.float64).reshape(2,480)
        wave=canonical(pre)
        different=np.nextafter(wave,np.inf)
        self.assertFalse(np.array_equal(wave,different))
        with patch.object(v,'read_arrays',side_effect=AssertionError('No scientific read')), \
             patch.object(v,'load_models',side_effect=AssertionError('No weights')), \
             patch.object(v,'replay_all',side_effect=AssertionError('No replay')):
            result=v.check_independent_ADC_codes(different,pre,label='artificial ULP')
        self.assertEqual(result['integer_code_mismatches'],0)
        self.assertEqual(result['samples_checked'],960)
        self.assertLessEqual(result['maximum_canonical_voltage_error_V'],4*np.finfo(np.float64).eps)

    def test_one_ADC_step_code_error_is_rejected(self):
        pre=np.zeros((1,480),dtype=np.float64)
        wave=canonical(pre)
        for index in (0,99,100,399,400,479):
            codes=np.rint((wave+.5)*16383).astype(np.int64)
            codes[0,index]+=1
            changed=codes.astype(np.float64)/16383-.5
            with self.subTest(sample=index),self.assertRaisesRegex(AssertionError,'integer code mismatch'):
                v.check_independent_ADC_codes(changed,pre,label='one code error')

    def test_same_code_off_canonical_grid_is_rejected(self):
        pre=np.zeros((1,480),dtype=np.float64)
        for index in (0,479):
            for delta in (1e-8,.1/16383,8*np.finfo(np.float64).eps):
                wave=canonical(pre)
                wave[0,index]+=delta
                np.testing.assert_array_equal(np.rint((wave+.5)*16383),np.rint((canonical(pre)+.5)*16383))
                with self.subTest(sample=index,delta=delta),self.assertRaisesRegex(AssertionError,'canonical voltage reconstruction error'):
                    v.check_independent_ADC_codes(wave,pre,label='off grid')

    def test_nonfinite_shape_and_precision_are_rejected(self):
        for bad in (np.nan,np.inf,-np.inf):
            for which in ('wave','pre'):
                pre=np.zeros((1,480),dtype=np.float64);wave=canonical(pre)
                if which=='wave':wave[0,0]=bad
                else:pre[0,0]=bad
                with self.subTest(value=bad,field=which),self.assertRaises(AssertionError):
                    v.check_independent_ADC_codes(wave,pre,label='invalid')
        with self.assertRaises(AssertionError):
            v.check_independent_ADC_codes(np.zeros((1,479)),np.zeros((1,480)),label='shape')
        with self.assertRaises(AssertionError):
            v.check_independent_ADC_codes(np.zeros((0,480)),np.zeros((0,480)),label='empty')
        with self.assertRaises(AssertionError):
            v.check_independent_ADC_codes(np.zeros((1,480),np.float32),np.zeros((1,480)),label='precision')

    def test_halfway_nearest_even_and_saturation_codes(self):
        pre=np.zeros((1,480),dtype=np.float64)
        pre[0,:9]=[-10.,-.5,-2/16383,-1/16383,0.,1/16383,2/16383,.5,10.]
        scaled=(pre[0,2:7]+.5)*16383
        np.testing.assert_array_equal(scaled,8191.5+np.arange(-2,3))
        wave=canonical(pre)
        result=v.check_independent_ADC_codes(wave,pre,label='ties and saturation')
        self.assertEqual(result['integer_code_mismatches'],0)
        codes=np.rint((wave+.5)*16383).astype(np.int64)
        np.testing.assert_array_equal(codes[0,:9],[0,0,8190,8190,8192,8192,8194,16383,16383])

    def test_baseline_changed_code_zero_and_nonzero_fractions_use_same_device(self):
        prefix=v.torch.zeros((3,1,100),dtype=v.torch.float64)
        reference=v.cpu(v.digitize(prefix,ste=False))[:,0]
        pre=np.zeros((3,480),dtype=np.float64)
        pre[1,:25]=1.2/16383;pre[2,:100]=1.2/16383
        wave=canonical(pre)
        v.check_independent_ADC_codes(wave,pre,label='baseline examples')
        changed=np.mean(wave[:,:100]!=reference,axis=1)
        np.testing.assert_array_equal(changed,[0.,.25,1.])
        changed_codes=np.mean(np.rint((wave[:,:100]+.5)*16383)!=np.rint((reference+.5)*16383),axis=1)
        np.testing.assert_array_equal(changed,changed_codes)
        # The former cross-backend style of float restoration can change all
        # inequality bits for a zero-signal prefix while integer codes match.
        different_restoration=np.nextafter(reference,np.inf)
        self.assertEqual(np.mean(wave[0,:100]!=different_restoration[0]),1.)
        np.testing.assert_array_equal(np.rint((reference+.5)*16383),np.rint((different_restoration+.5)*16383))
        self.assertEqual(changed[0],0.)

    def test_AST_only_ADC_helper_per_view_baseline_reference_and_receipt_description(self):
        old,new=ast.parse(OLD.read_text(encoding='utf8')),ast.parse(NEW.read_text(encoding='utf8'))
        old_defs={node.name:node for node in old.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        new_defs={node.name:node for node in new.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(set(new_defs)-set(old_defs),{'check_independent_ADC_codes'})
        for name,node in old_defs.items():
            if name not in ('replay_all','main'):self.assertEqual(ast.dump(node),ast.dump(new_defs[name]),name)
        normalized=copy.deepcopy(new)
        normalized.body=[node for node in normalized.body if not isinstance(node,ast.FunctionDef) or node.name!='check_independent_ADC_codes']
        old_change=next(node for node in ast.walk(old_defs['replay_all']) if isinstance(node,ast.Assign) and
                        len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='change')
        class Restore(ast.NodeTransformer):
            helper_calls=0
            baseline_changes=0
            fields=0
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='check_independent_ADC_codes':
                    self.helper_calls+=1;return None
                return self.generic_visit(node)
            def visit_Assign(self,node):
                if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='change':
                    self.baseline_changes+=1;return copy.deepcopy(old_change)
                return self.generic_visit(node)
            def visit_Call(self,node):
                for keyword in list(node.keywords):
                    if keyword.arg=='independent_ADC_code_checks':self.fields+=1;node.keywords.remove(keyword)
                return self.generic_visit(node)
        restore=Restore();restore.visit(normalized)
        self.assertEqual((restore.helper_calls,restore.baseline_changes,restore.fields),(1,1,1))
        self.assertEqual(ast.dump(old),ast.dump(normalized))
        loops=[node for node in ast.walk(new_defs['replay_all']) if isinstance(node,ast.For) and
               isinstance(node.target,ast.Tuple) and [item.id for item in node.target.elts if isinstance(item,ast.Name)]==['iv','view']]
        self.assertEqual(len(loops),1)
        first=loops[0].body[0]
        self.assertIsInstance(first,ast.Expr)
        self.assertEqual(first.value.func.id,'check_independent_ADC_codes')


if __name__=='__main__':unittest.main()
