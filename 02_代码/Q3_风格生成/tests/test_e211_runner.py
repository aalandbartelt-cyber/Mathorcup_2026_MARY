import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))


class BasisRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('run_e211_basis_mixture'),'basis runner not implemented')
        from run_e211_basis_mixture import validate_config,select_inside_training
        self.validate=validate_config;self.select=select_inside_training
        self.config=yaml.safe_load((SRC.parent/'configs/E211_basis_mixture.yaml').read_text())

    def test_protocol_cannot_change_folds_streams_geometry_or_fallback(self):
        self.validate(self.config)
        for key,value in [('n_repeats',3),('inner_folds',3),('gpu_used',True),('random_stream_policy','shared'),('fallback_order',['best_outer'])]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):self.validate(dict(self.config,**{key:value}))
        with self.assertRaises(ValueError):self.validate(dict(self.config,geometry=dict(self.config['geometry'],coordinate_dim=8)))

    def test_training_only_selection_is_deterministic_and_equal_budget_per_geometry(self):
        rng=np.random.default_rng(47);raw=rng.normal(size=(24,6));raw[0,0]=np.nan;labels=np.array(['a']*12+['b']*12)
        config=dict(self.config,target_styles=['a','b'])
        first,table=self.select(raw,labels,list(range(5)),config)
        second,again=self.select(raw,labels,list(range(5)),config)
        self.assertEqual(first,second);self.assertTrue(table.equals(again));self.assertEqual(len(table),36)
        self.assertTrue(table.n_inner_training_style.eq(6).all());self.assertTrue(table.n_inner_validation_style.eq(6).all())
        self.assertTrue(table.groupby(['inner_fold','style','method']).selected.sum().eq(1).all())


if __name__=='__main__':unittest.main()
