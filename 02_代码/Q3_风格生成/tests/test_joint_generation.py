from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from joint_generation import JointStyleGenerator,diversity_diagnostics,balanced_inner_splits


class JointGenerationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(9)
        self.x=rng.normal(size=(30,6));self.x[:15,0]=2.5;self.x[:10,1]=0.
        self.styles=np.array(['a']*15+['b']*15)

    def test_beta_versions_are_reproducible_and_preserve_style_support_and_constants(self):
        model=JointStyleGenerator().fit(self.x,self.styles,scale=np.ones(6))
        for kind in ['beta_linear','beta_step']:
            a=model.generate('a',200,kind,seed=13);b=model.generate('a',200,kind,seed=13)
            np.testing.assert_array_equal(a,b)
            self.assertTrue((a>=self.x[:15].min(axis=0)-1e-12).all())
            self.assertTrue((a<=self.x[:15].max(axis=0)+1e-12).all())
            self.assertTrue((a[:,0]==2.5).all())

    def test_beta_step_preserves_floor_mass_and_uniform_probability_margins(self):
        model=JointStyleGenerator().fit(self.x,self.styles,scale=np.ones(6))
        values=model.generate('a',30000,'beta_step',seed=5)
        self.assertAlmostEqual(float((values[:,1]==0).mean()),10/15,delta=.02)
        for ranks in model.ranks_['a'].T:
            self.assertEqual(sorted(ranks.tolist()),list(range(1,16)))

    def test_mixture_selects_complete_rows_instead_of_coordinate_averaging(self):
        model=JointStyleGenerator().fit(self.x,self.styles,scale=np.ones(6))
        copula=model.generate('a',50,'copula',seed=29)
        local=model.generate('a',50,'local',seed=29)
        mixed=model.generate('a',50,'mix_50',seed=29)
        self.assertTrue(np.all(np.isclose(mixed,copula).all(axis=1)|np.isclose(mixed,local).all(axis=1)))

    def test_collapsed_mean_cannot_be_a_diversity_winner_despite_high_coverage(self):
        real=np.array([[-1.,0.],[0.,-1.],[1.,0.],[0.,1.]])
        mean=np.zeros_like(real)
        result=diversity_diagnostics(real,mean,real+10,np.ones(2))
        self.assertEqual(result['coverage'],1.)
        self.assertEqual(result['pairwise_spread_ratio'],0.)
        self.assertTrue(np.isinf(result['log_spread_error']))
        self.assertTrue(result['collapsed'])

    def test_balanced_inner_splits_exclude_each_validation_sample_and_keep_target_counts(self):
        styles=np.array(['a']*7+['b']*6+['rare'])
        splits=balanced_inner_splits(styles,seed=5)
        seen=[]
        for train,valid in splits:
            self.assertFalse(set(train)&set(valid))
            self.assertEqual(set(train)|set(valid),set(range(len(styles))))
            for target in ['a','b']:
                self.assertGreaterEqual(int((styles[train]==target).sum()),2)
                self.assertGreaterEqual(int((styles[valid]==target).sum()),2)
            seen.extend(valid)
        self.assertEqual(sorted(seen),list(range(len(styles))))


if __name__=='__main__':unittest.main()
