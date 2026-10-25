from pathlib import Path
import sys,unittest
import numpy as np
SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from empirical_generation import StyleEmpiricalGenerator

class EmpiricalGenerationTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0.,0.],[2.,4.],[4.,8.],[20.,40.],[22.,44.]])
        self.styles=np.array(['a','a','a','b','b'])
    def test_resampling_is_style_only_and_does_not_change_fit(self):
        model=StyleEmpiricalGenerator().fit(self.x,self.styles)
        self.x[:]=999
        sample=model.generate('a',20,method='empirical_resample',seed=1)
        self.assertTrue(set(map(tuple,sample['values'])).issubset({(0.,0.),(2.,4.),(4.,8.)}))
        self.assertTrue(np.all(sample['anchor_a']<3))
    def test_interpolation_uses_distinct_style_anchors(self):
        model=StyleEmpiricalGenerator().fit(self.x,self.styles)
        result=model.generate('b',12,method='local_interpolation',seed=2)
        self.assertTrue(np.all(result['anchor_a']!=result['anchor_b']))
        self.assertTrue(np.all((result['values'][:,0]>20)&(result['values'][:,0]<22)))
        np.testing.assert_allclose(result['values'][:,1],2*result['values'][:,0])
    def test_mean_is_an_explicit_degenerate_control(self):
        model=StyleEmpiricalGenerator().fit(self.x,self.styles)
        sample=model.generate('a',6,method='style_mean',seed=0)['values']
        np.testing.assert_array_equal(sample,np.tile([2.,4.],(6,1)))
    def test_unseen_style_cannot_use_global_fallback(self):
        model=StyleEmpiricalGenerator().fit(self.x,self.styles)
        with self.assertRaises(ValueError):model.generate('c',3,method='empirical_resample',seed=0)
    def test_generation_is_seed_reproducible(self):
        model=StyleEmpiricalGenerator().fit(self.x,self.styles)
        for method in ['empirical_resample','local_interpolation','style_mean']:
            a=model.generate('a',9,method=method,seed=99);b=model.generate('a',9,method=method,seed=99)
            np.testing.assert_array_equal(a['values'],b['values'])
    def test_missing_or_nonfinite_data_rejected_until_fold_imputation(self):
        self.x[0,0]=np.nan
        with self.assertRaises(ValueError):StyleEmpiricalGenerator().fit(self.x,self.styles)

if __name__=='__main__':unittest.main()
