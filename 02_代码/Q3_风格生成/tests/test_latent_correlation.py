from pathlib import Path
import sys
import unittest

import numpy as np

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from latent_correlation import LatentCorrelationGenerator


class LatentCorrelationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(45);a=rng.normal(size=40)
        self.x=np.column_stack([a,2*a+rng.normal(size=40)*.1,np.ones(40),np.round(a)])
        self.model=LatentCorrelationGenerator().fit(self.x,np.repeat('s',40),target_styles=['s'])

    def test_correlation_is_psd_unit_diagonal_and_keeps_constant_edge_independent(self):
        for rho in [0.,.05,.5,1.]:
            matrix=self.model.correlation('s',rho)
            np.testing.assert_allclose(np.diag(matrix),1.,atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(),-1e-12)
            np.testing.assert_allclose(matrix[2],np.eye(4)[2],atol=1e-12)
        np.testing.assert_allclose(self.model.correlation('s',1.),np.eye(4),atol=1e-12)
        self.assertGreater(self.model.correlation('s',0.)[0,1],.9)

    def test_seed_empirical_support_constant_and_noncopying(self):
        a=self.model.generate('s',200,shrinkage=.25,seed=11)
        np.testing.assert_array_equal(a,self.model.generate('s',200,shrinkage=.25,seed=11))
        np.testing.assert_array_equal(a[:,2],np.ones(200))
        self.assertTrue((a>=self.x.min(0)-1e-12).all());self.assertTrue((a<=self.x.max(0)+1e-12).all())
        self.assertFalse(np.any(np.all(np.isclose(a[:,None],self.x[None],atol=1e-10,rtol=0.),axis=2)))

    def test_strong_dependence_and_independence_have_distinct_joint_structure(self):
        a=self.model.generate('s',3000,shrinkage=0.,seed=12)
        b=self.model.generate('s',3000,shrinkage=1.,seed=12)
        self.assertGreater(np.corrcoef(a[:,:2],rowvar=False)[0,1],.9)
        self.assertLess(abs(np.corrcoef(b[:,:2],rowvar=False)[0,1]),.08)
        with self.assertRaises(ValueError):self.model.generate('s',10,shrinkage=-.1,seed=1)


if __name__=='__main__':unittest.main()
