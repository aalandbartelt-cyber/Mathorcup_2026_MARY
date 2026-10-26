from pathlib import Path
import copy
import sys
import unittest

import numpy as np
import yaml

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from run_e205_sensory_audit import validate_config,transform_elastic
from selection_models import fit_fold_preprocessor


class E205RunnerTests(unittest.TestCase):
    def setUp(self):
        self.config=yaml.safe_load((SRC.parent/'configs/E205_sensory_audit.yaml').read_text())

    def test_freezes_read_only_q1_and_no_all_data_graph_weights(self):
        validate_config(self.config)
        for key in ['q1_retrained','nnls_refit','hyperparameter_search','q2_top3_in_optimizer','physical_safety_certified']:
            config=dict(self.config,**{key:True})
            with self.subTest(key=key):
                with self.assertRaises(ValueError):validate_config(config)

    def test_missing_mask_and_proxy_rules_cannot_change_implicitly(self):
        for key,value in [('source_missing_mask','impute_before_predict'),('proxy_positive_r2_repeats',0),
                          ('support_min_repeat_fraction',0.5),('proxy_numerical_tolerance',0.1)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):validate_config(dict(self.config,**{key:value}))

    def test_elastic_transform_reproduces_training_only_preprocessor(self):
        train=np.array([[0.,np.nan,1.],[1.,2.,4.],[2.,5.,3.],[3.,4.,2.]])
        valid=np.array([[100.,3.,3.],[np.nan,np.nan,2.]])
        processed=fit_fold_preprocessor(train,valid)
        np.testing.assert_allclose(transform_elastic(processed,train),processed.x_train,rtol=0,atol=1e-12)
        np.testing.assert_allclose(transform_elastic(processed,valid),processed.x_validation,rtol=0,atol=1e-12)
        # An extreme query cannot modify fitted training medians or scales.
        medians=processed.medians.copy();scales=processed.scales.copy()
        transform_elastic(processed,np.array([[1e9,np.nan,1e9]]))
        np.testing.assert_array_equal(processed.medians,medians)
        np.testing.assert_array_equal(processed.scales,scales)


if __name__=='__main__':unittest.main()
