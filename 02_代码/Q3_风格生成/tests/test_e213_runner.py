import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))


class NeuralRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('run_e213_neural_interpolation'),'runner not implemented')
        from run_e213_neural_interpolation import validate_config,select_inside_training,choices_from_table
        self.validate=validate_config;self.select=select_inside_training;self.choose=choices_from_table
        self.config=yaml.safe_load((SRC.parent/'configs/E213_neural_interpolation.yaml').read_text())

    def test_fixed_training_and_nested_budget_cannot_change(self):
        self.validate(self.config)
        for key,value in [('n_repeats',3),('inner_folds',3),('device','cpu'),('training_seed',42),('seeds',[1,2,3,4,5]),('kaggle_used_for_training',True)]:
            with self.subTest(key=key):
                changed=copy.deepcopy(self.config);changed[key]=value
                with self.assertRaises(ValueError):self.validate(changed)
        changed=copy.deepcopy(self.config);changed['model']['train_steps']=400
        with self.assertRaises(ValueError):self.validate(changed)

    def test_inner_selection_replays_and_neural_choice_is_inside_fixed_grid(self):
        rng=np.random.default_rng(37);raw=rng.normal(size=(24,6));raw[0,0]=np.nan;styles=np.array(['a']*12+['b']*12)
        config=copy.deepcopy(self.config);config.update(device='cpu',target_styles=['a','b'])
        config['model'].update(hidden_dim=8,latent_dim=3,train_steps=3)
        chosen,table,history=self.select(raw,styles,config,train_seed=12)
        again,other,_=self.select(raw,styles,config,train_seed=12)
        self.assertEqual(chosen,again);self.assertTrue(table.equals(other));self.assertEqual(len(table),36)
        self.assertEqual(chosen,self.choose(table,config));self.assertEqual(len(history),6)
        self.assertTrue(table.n_inner_training_style.eq(6).all());self.assertTrue(table.n_inner_validation_style.eq(6).all())
        for s,methods in chosen.items():
            self.assertEqual(set(methods),set(config['methods']))
            self.assertEqual(methods['pca_noise']['beta'],0.);self.assertEqual(methods['neural_noise']['beta'],0.)
            self.assertEqual(methods['neural_curve']['beta'],.5)
            self.assertIn(methods['neural_selected']['beta'],[0.,.5])
            for v in methods.values():self.assertIn(v['alpha'],[.25,.5,.75])


if __name__=='__main__':unittest.main()
