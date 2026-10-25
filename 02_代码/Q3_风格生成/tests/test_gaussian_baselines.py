from pathlib import Path
import sys,unittest
import numpy as np
SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from gaussian_baselines import StyleGaussianGenerator

class GaussianBaselineTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(3);self.x=rng.normal(size=(30,4));self.styles=np.array(['a']*15+['b']*15)
    def test_registered_methods_reproduce_shape_and_seed(self):
        m=StyleGaussianGenerator().fit(self.x,self.styles)
        for method in ['style_gaussian','style_copula']:
            a=m.generate('a',8,method=method,seed=4)['values'];b=m.generate('a',8,method=method,seed=4)['values'];self.assertEqual(a.shape,(8,4));np.testing.assert_array_equal(a,b)
    def test_style_isolation_and_no_unseen_fallback(self):
        m=StyleGaussianGenerator().fit(self.x,self.styles);a=m.generate('a',6)['values'];b=m.generate('b',6)['values'];self.assertFalse(np.allclose(a,b))
        with self.assertRaises(ValueError):m.generate('c',3)
    def test_invalid_data_and_method_rejected(self):
        bad=self.x.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):StyleGaussianGenerator().fit(bad,self.styles)
        m=StyleGaussianGenerator().fit(self.x,self.styles)
        with self.assertRaises(ValueError):m.generate('a',2,method='copula_bad')
