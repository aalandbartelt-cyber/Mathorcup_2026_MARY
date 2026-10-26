from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from run_e208_joint_generation import validate_config,select_inside_training


class JointRunnerTests(unittest.TestCase):
    def setUp(self):self.config=yaml.safe_load((SRC.parent/'configs/E208_joint_generation.yaml').read_text())

    def test_outer_protocol_and_candidate_set_cannot_silently_change(self):
        validate_config(self.config)
        for key,value in [('n_repeats',1),('n_folds',5),('inner_folds',3),('q1_retrained',True),
                          ('gpu_used',True),('inner_spread_log_weight',0.)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):validate_config(dict(self.config,**{key:value}))

    def test_selection_is_reproducible_training_only_and_records_each_inner_split(self):
        rng=np.random.default_rng(17);x=rng.normal(size=(20,5));x[0,0]=np.nan;styles=np.array(['a']*10+['b']*10)
        config=dict(self.config,target_styles=['a','b'])
        first,table=select_inside_training(x,styles,config)
        second,again=select_inside_training(x,styles,config)
        self.assertEqual(first,second)
        self.assertTrue(table.equals(again))
        self.assertEqual(len(table),28)
        self.assertEqual(set(table.inner_fold),{0,1})
        self.assertTrue(table.n_inner_training_style.eq(5).all())
        self.assertTrue(table.n_inner_validation_style.eq(5).all())
        self.assertTrue(table.groupby(['inner_fold','style']).selected_joint.sum().eq(1).all())


if __name__=='__main__':unittest.main()
