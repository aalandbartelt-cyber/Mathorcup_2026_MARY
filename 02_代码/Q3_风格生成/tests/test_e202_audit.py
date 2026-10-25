from pathlib import Path
import sys
import unittest
import numpy as np

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
from audit_e202 import training_winsorize


class E202AuditTests(unittest.TestCase):
    def test_winsor_bounds_come_only_from_training_rows(self):
        training=np.arange(40,dtype=float).reshape(20,2)
        validation=np.array([[1e9,-1e9],[5.,6.]])
        clipped_training,(clipped_validation,),scale,bounds=training_winsorize(training,validation,lower=.1,upper=.9)
        low,high=bounds
        np.testing.assert_allclose(low,np.quantile(training,.1,axis=0))
        np.testing.assert_allclose(high,np.quantile(training,.9,axis=0))
        self.assertTrue(np.all(clipped_validation<=high))
        self.assertTrue(np.all(clipped_validation>=low))
        self.assertTrue(np.all(scale>0))
        self.assertEqual(clipped_training.shape,training.shape)

    def test_invalid_quantiles_or_nonfinite_training_are_rejected(self):
        training=np.arange(20,dtype=float).reshape(10,2)
        with self.assertRaises(ValueError):training_winsorize(training,training,lower=.9,upper=.1)
        training[0,0]=np.nan
        with self.assertRaises(ValueError):training_winsorize(training,training)


if __name__=='__main__':
    unittest.main()
