from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from run_e209_lambic_search import validate_config,anchor_pool_ids


class LambicRunnerTests(unittest.TestCase):
    def setUp(self):self.config=yaml.safe_load((SRC.parent/'configs/E209_lambic_search.yaml').read_text())

    def test_fixed_search_cannot_relax_margins_or_access_auditor(self):
        validate_config(self.config)
        for key,value in [('target_style','Tripel'),('primary_proxy_guard_margin',.01),
            ('final_two_model_guard_margin',0.),('maximum_source_change_rms',2.),
            ('audit_used_during_search',True),('nnls_refit',True),('gpu_used',True)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):validate_config(dict(self.config,**{key:value}))

    def test_geometry_and_seed_choose_unique_anchors_without_proxy_scores(self):
        pool=np.arange(6000.)[:,None];source=np.array([0.]);scale=np.ones(1)
        a=anchor_pool_ids(pool,source,scale,self.config,230)
        b=anchor_pool_ids(pool,source,scale,self.config,230)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:512],np.arange(512))
        self.assertEqual(len(np.unique(a)),1024)
        self.assertTrue((a[512:]>=512).all())


if __name__=='__main__':unittest.main()
