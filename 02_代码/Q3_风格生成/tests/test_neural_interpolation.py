import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))


class NeuralInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('neural_interpolation'),'neural interpolation not implemented')
        import neural_interpolation as module
        self.mod=module
        self.config=yaml.safe_load((SRC.parent/'configs/E213_neural_interpolation.yaml').read_text())['model']
        self.config=dict(self.config,train_steps=8,hidden_dim=8,latent_dim=3)
        rng=np.random.default_rng(21);self.x=rng.normal(size=(30,7));self.x[:,6]=2.
        self.labels=np.array(['a']*15+['b']*15)

    def fit(self):
        return self.mod.ResidualInterpolator().fit(self.x,self.labels,self.config,
            target_styles=['a','b'],train_seed=12,device='cpu')

    def test_curvature_is_zero_at_endpoints_and_for_affine_decoder(self):
        rng=np.random.default_rng(7);a=rng.normal(size=(12,3));b=rng.normal(size=(12,3));t=np.linspace(0,1,12)
        weights=rng.normal(size=(3,5));bias=rng.normal(size=5)
        value=self.mod.curvature_delta(a,b,t,lambda z:z@weights+bias)
        np.testing.assert_allclose(value,0.,atol=1e-12)
        value=self.mod.curvature_delta(a,b,t,lambda z:z*z)
        np.testing.assert_allclose(value,-t[:,None]*(1-t[:,None])*(a-b)**2,atol=1e-12)
        np.testing.assert_allclose(value[[0,-1]],0.,atol=1e-12)

    def test_training_is_seeded_and_out_of_sample_calls_do_not_change_training_statistics(self):
        first=self.fit();second=self.fit()
        for key,value in first.state_arrays().items():np.testing.assert_array_equal(value,second.state_arrays()[key])
        center=first.center_.copy();query=self.x[:2].copy();query[:,0]=1e6
        self.assertTrue(np.isfinite(first.reconstruct(query,'neural')).all())
        np.testing.assert_array_equal(center,first.center_)
        self.assertEqual(len(first.history_),8)

    def test_zero_correction_recovers_original_local_primitive(self):
        from empirical_generation import StyleEmpiricalGenerator
        model=self.fit();scale=self.x.std(axis=0);scale=np.where(scale>1e-12,scale,1.)
        old=StyleEmpiricalGenerator().fit(self.x,self.labels,scale=scale)
        expected=old.generate('a',40,method='local_interpolation',seed=14)['values']
        actual=model.local('a',40,'raw',seed=14,noise_seed=15,beta=0.,gamma=0.)
        np.testing.assert_array_equal(actual['values'],expected)

    def test_saved_cpu_weights_restore_the_exact_generator_without_refitting(self):
        model=self.fit();state=model.state_arrays()
        restored=self.mod.ResidualInterpolator.from_arrays(self.x,self.labels,self.config,
            target_styles=['a','b'],arrays=state)
        first=model.generate('a',80,'neural',alpha=.5,beta=.5,gamma=.5,seed=31)
        again=restored.generate('a',80,'neural',alpha=.5,beta=.5,gamma=.5,seed=31)
        np.testing.assert_array_equal(first['values'],again['values'])
        self.assertEqual(len(set(first['stream_seeds'])),4)

    def test_local_outputs_preserve_constants_support_and_report_clipping(self):
        model=self.fit()
        for family,beta in [('pca',0.),('neural',0.),('neural',.5)]:
            sample=model.local('a',100,family,seed=22,noise_seed=23,beta=beta,gamma=.5)
            values=sample['values'];np.testing.assert_array_equal(values[:,6],np.full(100,2.))
            self.assertTrue((values>=self.x[:15].min(0)-1e-12).all());self.assertTrue((values<=self.x[:15].max(0)+1e-12).all())
            self.assertTrue(np.all((sample['preclip_cell_fraction']>=0)&(sample['preclip_cell_fraction']<=1)))
            self.assertFalse(np.any(sample['anchor_a']==sample['anchor_b']))


if __name__=='__main__':unittest.main()
