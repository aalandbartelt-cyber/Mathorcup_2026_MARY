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
        with self.assertRaises(ValueError):StyleGaussianGenerator().fit(self.x,self.styles,mean_shrinkage=1.1)
        m=StyleGaussianGenerator().fit(self.x,self.styles)
        with self.assertRaises(ValueError):m.generate('a',2,method='copula_bad')

    def test_formal_gaussian_is_reproducible_and_clipped_to_style_support(self):
        x=self.x.copy();x[:,0]*=1000.;x[:15,1]=2.5
        m=StyleGaussianGenerator().fit(x,self.styles)
        a=m.generate('a',200,method='style_gaussian_clipped',seed=17)
        b=m.generate('a',200,method='style_gaussian_clipped',seed=17)
        np.testing.assert_array_equal(a['values'],b['values'])
        lo=x[:15].min(0);hi=x[:15].max(0)
        self.assertTrue(np.all(a['values']>=lo-1e-12))
        self.assertTrue(np.all(a['values']<=hi+1e-12))
        self.assertGreaterEqual(a['pre_clip_cell_fraction'],0.)
        self.assertLessEqual(a['pre_clip_cell_fraction'],1.)
        self.assertGreaterEqual(a['pre_clip_row_fraction'],0.)
        self.assertLessEqual(a['pre_clip_row_fraction'],1.)

    def test_shrunk_copula_handles_ties_and_has_psd_latent_covariance(self):
        x=self.x.copy();x[:15,0]=0.;x[:15,1]=np.round(x[:15,1],1)
        m=StyleGaussianGenerator().fit(x,self.styles)
        sample=m.generate('a',100,method='style_copula_shrunk',seed=11)
        self.assertTrue(np.isfinite(sample['values']).all())
        self.assertTrue(np.all(sample['values']>=x[:15].min(0)-1e-12))
        self.assertTrue(np.all(sample['values']<=x[:15].max(0)+1e-12))
        diagnostic=m.diagnostics('a')
        self.assertGreaterEqual(diagnostic['gaussian_min_eigenvalue'],-1e-10)
        self.assertGreaterEqual(diagnostic['copula_min_eigenvalue'],-1e-10)
        self.assertEqual(sample['pre_clip_cell_fraction'],0.)
        self.assertEqual(sample['pre_clip_row_fraction'],0.)
