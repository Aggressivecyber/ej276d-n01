"""Semantic checks for diagnostic transforms and group isolation."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1] / 'scripts/diagnose_n01_windows.py'
spec = importlib.util.spec_from_file_location('window_audit', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WindowDiagnosticTests(unittest.TestCase):
    def test_baseline_transforms_remove_only_declared_components(self):
        rng = np.random.default_rng(14)
        x = rng.normal(0, .001, (8, 480))
        t = np.arange(480)
        shifted = x + .015 + .0002 * t
        self.assertTrue(np.allclose(module.view_array(x, 'baseline_detrend'), module.view_array(shifted, 'baseline_detrend'), atol=1e-8))
        self.assertTrue(np.allclose(module.view_array(x, 'baseline_demean'), module.view_array(x + .02, 'baseline_demean'), atol=1e-8))
        self.assertGreater(np.max(np.abs(module.view_array(x, 'baseline_demean') - module.view_array(shifted, 'baseline_demean'))), .005)

    def test_controls_keep_groups_disjoint_and_same_row_budget(self):
        parents = np.repeat(np.arange(512), 4)
        groups = np.repeat(np.arange(12), 50)
        cases = module.build_splits(parents, groups)
        for name in ('matched_SOURCE_REAL', 'matched_REAL_REAL'):
            self.assertEqual([len(cases[name][k]) for k in ('fit0', 'fit1', 'test0', 'test1')], [128, 128, 96, 96])
        selected = cases['matched_SOURCE_REAL']
        self.assertEqual(len(np.unique(parents[selected['fit0']])), 128)
        self.assertEqual(len(np.unique(parents[selected['test0']])), 96)

    def test_stratified_control_separates_groups_between_roles_and_events_between_domains(self):
        groups = np.repeat(np.arange(12), 50)
        s = module.stratified_real_control(groups)
        self.assertFalse(set(groups[s['fit0']]) & set(groups[s['test0']]))
        self.assertFalse(set(s['test0']) & set(s['test1']))
        for g in np.unique(groups[s['test0']]):
            self.assertEqual(np.sum(groups[s['test0']] == g), np.sum(groups[s['test1']] == g))


if __name__ == '__main__':
    unittest.main()
