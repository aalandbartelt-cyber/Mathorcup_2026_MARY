import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))


class BasisMixtureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('basis_mixture'),'chemical basis mixture is not implemented')
        from basis_mixture import ChemicalBasisGeometry,ChemicalMixtureGenerator,mixture_seeds
        self.Geometry=ChemicalBasisGeometry;self.Generator=ChemicalMixtureGenerator;self.seeds=mixture_seeds
        self.config=yaml.safe_load((SRC.parent/'configs/E211_basis_mixture.yaml').read_text())['geometry']
        rng=np.random.default_rng(8);self.x=rng.normal(size=(32,8));self.x[:,7]=2.
        self.labels=np.array(['a']*16+['b']*16);self.indices=list(range(6))

    def test_geometry_is_training_only_and_handles_constants_and_out_of_sample(self):
        geometry=self.Geometry().fit(self.x,self.indices,self.config)
        before=geometry.transform(self.x).copy();center=geometry.center_.copy()
        out=self.x[:2].copy();out[:,0]=1e8
        self.assertTrue(np.isfinite(geometry.transform(out)).all())
        np.testing.assert_array_equal(before,geometry.transform(self.x));np.testing.assert_array_equal(center,geometry.center_)
        again=self.Geometry().fit(self.x,self.indices,self.config)
        np.testing.assert_allclose(before,again.transform(self.x),atol=1e-12)
        self.assertEqual(geometry.basis(before).shape[1],2*len(geometry.prototypes_))
        flat=self.Geometry().fit(np.ones((12,8)),self.indices,self.config)
        self.assertTrue(np.isfinite(flat.transform(np.ones((2,8)))).all())

    def test_neighborhoods_are_same_style_nonself_with_stability_and_fallback_diagnostics(self):
        model=self.Generator().fit(self.x,self.labels,self.indices,self.config,target_styles=['a','b'])
        for style in ['a','b']:
            ids=np.flatnonzero(self.labels==style)
            for mode in ['raw','pca','basis']:
                for anchor,neighbors in zip(ids,model.neighbors_[style][mode]):
                    self.assertNotIn(anchor,neighbors);self.assertTrue(np.all(self.labels[neighbors]==style));self.assertEqual(len(neighbors),3)
        self.assertEqual(len(model.stability_table_),16)
        self.assertTrue(model.stability_table_.median_jaccard.between(0,1).all())

    def test_raw_local_primitive_reproduces_original_and_preserves_interpolation(self):
        from empirical_generation import StyleEmpiricalGenerator
        model=self.Generator().fit(self.x,self.labels,self.indices,self.config,target_styles=['a','b'])
        scale=np.std(self.x,axis=0);scale=np.where(scale>1e-12,scale,1.)
        old=StyleEmpiricalGenerator().fit(self.x,self.labels,scale=scale)
        for mode in ['raw','pca','basis']:
            sample=model.local('a',40,mode,seed=11)
            expected=(1-sample['mixture'][:,None])*self.x[sample['anchor_a']]+sample['mixture'][:,None]*self.x[sample['anchor_b']]
            np.testing.assert_array_equal(sample['values'],expected)
            if mode=='raw':np.testing.assert_array_equal(sample['values'],old.generate('a',40,method='local_interpolation',seed=11)['values'])
            self.assertTrue(sample['mixture'].min()>=.2);self.assertTrue(sample['mixture'].max()<=.8)
            np.testing.assert_allclose(sample['values'][:,7],np.full(40,2.),atol=1e-12,rtol=0.)

    def test_whole_row_mixture_uses_independent_streams_and_common_component_samples(self):
        model=self.Generator().fit(self.x,self.labels,self.indices,self.config,target_styles=['a','b'])
        chooser,copula_seed,local_seed=self.seeds(52)
        self.assertEqual(len(set([chooser,copula_seed,local_seed])),3)
        copula=model.copulas_['a'].generate('a',100,method='style_copula_shrunk',seed=copula_seed)['values']
        local=model.local('a',100,'raw',seed=local_seed)['values']
        mask=np.random.default_rng(chooser).random(100)<.5
        mixed=model.generate('a',100,'raw',alpha=.5,seed=52)
        np.testing.assert_array_equal(mixed['values'],np.where(mask[:,None],local,copula))
        np.testing.assert_array_equal(mixed['local_selected'],mask)
        np.testing.assert_array_equal(model.generate('a',100,'raw',alpha=0.,seed=52)['values'],copula)
        np.testing.assert_array_equal(model.generate('a',100,'raw',alpha=1.,seed=52)['values'],local)
        u=np.random.default_rng(chooser).random(20000);v=np.random.default_rng(local_seed).random(20000)
        self.assertLess(abs(np.corrcoef(u,v)[0,1]),.03)

    def test_basis_far_zero_and_degenerate_cases_revert_to_pca_neighbors(self):
        geometry=self.Geometry().fit(self.x,self.indices,self.config)
        pca,_=geometry.neighborhood(self.x[:8],mode='pca')
        geometry.prototypes_+=1e7
        basis,diagnostic=geometry.neighborhood(self.x[:8],mode='basis')
        np.testing.assert_array_equal(pca,basis)
        self.assertTrue(diagnostic.fallback.all())


if __name__=='__main__':unittest.main()
