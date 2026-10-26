from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from run_e210_latent_correlation import validate_config,select_inside_training
from audit_e210 import assess
import test_e208_audit as fixtures


class LatentRunnerTests(unittest.TestCase):
    def setUp(self):self.config=yaml.safe_load((SRC.parent/'configs/E210_latent_correlation.yaml').read_text())

    def test_selection_is_training_local_deterministic_and_preserves_inner_denominators(self):
        config=dict(self.config,target_styles=['a','b'])
        rng=np.random.default_rng(33);x=rng.normal(size=(32,4));x[0,0]=np.nan;styles=np.array(['a']*16+['b']*16)
        a,table=select_inside_training(x,styles,config);b,again=select_inside_training(x,styles,config)
        self.assertEqual(a,b);self.assertTrue(table.equals(again));self.assertEqual(len(table),28)
        self.assertTrue(table.n_inner_training_style.eq(8).all());self.assertTrue(table.n_inner_validation_style.eq(8).all())
        self.assertTrue(table.groupby(['inner_fold','style']).selected.sum().eq(1).all())

    def test_protocol_and_preexisting_replacement_limits_cannot_relax(self):
        validate_config(self.config)
        for key,value in [('n_repeats',3),('inner_folds',3),('correlation_shrinkages',[0.,1.]),('q1_retrained',True)]:
            with self.assertRaises(ValueError):validate_config(dict(self.config,**{key:value}))
        fixture=fixtures.JointReplacementTests();fixture.setUp()
        table=fixture.table.replace({'method':{'joint_selected':'latent_selected'}})
        raw=fixture.raw.replace({'method':{'joint_selected':'latent_selected'}})
        decision,_,_=assess(table,raw,self.config)
        self.assertTrue(decision['replace_copula']);self.assertEqual(decision['primary_distribution'],'latent_selected')
        changed=dict(self.config,replacement_rules=dict(self.config['replacement_rules'],robust_energy_better_repeats=6))
        with self.assertRaises(ValueError):assess(table,raw,changed)


if __name__=='__main__':unittest.main()
