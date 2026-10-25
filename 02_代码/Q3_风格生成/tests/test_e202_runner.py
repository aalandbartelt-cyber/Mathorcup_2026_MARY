from pathlib import Path
import json
import sys
import tempfile
import unittest

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
from run_e202_gaussian_baselines import compose_metric_row,validate_formal_config,verify_e201_gate


class E202RunnerTests(unittest.TestCase):
    def test_metric_scope_from_evaluator_cannot_overwrite_fold_scope(self):
        row=compose_metric_row(
            'outer_validation_evaluation',
            {'repeat':0,'fold':0},
            {'scope':'descriptive_distribution_diagnostics_no_hypothesis_test','mean_w1':0.2},
        )
        self.assertEqual(row['evaluation_scope'],'outer_validation_evaluation')
        self.assertEqual(row['diagnostic_scope'],'descriptive_distribution_diagnostics_no_hypothesis_test')
        self.assertNotIn('scope',row)

    def test_formal_config_requires_registered_cpu_10_by_3_protocol(self):
        config={
            'methods':['style_gaussian_clipped','style_copula_shrunk'],
            'seeds':[1,2,3,4,5],
            'n_repeats':10,
            'n_folds':3,
            'target_styles':['Tripel','Kriek','Lambic'],
            'gpu_used':False,
        }
        validate_formal_config(config)
        bad=dict(config,methods=['style_gaussian'])
        with self.assertRaises(ValueError):validate_formal_config(bad)
        bad=dict(config,gpu_used=True)
        with self.assertRaises(ValueError):validate_formal_config(bad)

    def test_gate_accepts_frozen_e201_without_reading_q2_runtime_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest={
                'experiment_id':'E201',
                'status':'complete',
                'n_outer_folds':30,
                'signature':{
                    'q1_final_oof_sha256':'abc',
                    'entry_gate':{'checkpoint':'R00F0','scope':'outer_train_fold','sha256':'def'},
                },
            }
            path=root/'q3_run_manifest_E201.json'
            path.write_text(json.dumps(manifest))
            gate=verify_e201_gate(root,expected_q1_oof_sha256='abc')
            self.assertEqual(gate['checkpoint'],'R00F0')
            self.assertEqual(gate['scope'],'outer_train_fold')
            self.assertIn('e201_manifest_sha256',gate)

    def test_gate_rejects_wrong_scope_or_q1_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/'q3_run_manifest_E201.json'
            base={
                'experiment_id':'E201','status':'complete','n_outer_folds':30,
                'signature':{'q1_final_oof_sha256':'abc','entry_gate':{'checkpoint':'R00F0','scope':'full_data_descriptive','sha256':'def'}},
            }
            path.write_text(json.dumps(base))
            with self.assertRaises(ValueError):verify_e201_gate(root,expected_q1_oof_sha256='abc')
            base['signature']['entry_gate']['scope']='outer_train_fold';path.write_text(json.dumps(base))
            with self.assertRaises(ValueError):verify_e201_gate(root,expected_q1_oof_sha256='other')


if __name__=='__main__':
    unittest.main()
