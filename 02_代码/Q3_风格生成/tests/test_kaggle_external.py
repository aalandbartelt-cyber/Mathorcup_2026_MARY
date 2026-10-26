import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))


class KaggleExternalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('kaggle_external'),'external audit primitives not implemented')
        import kaggle_external as mod
        self.mod=mod

    def test_normalization_excludes_all_zero_rows_without_turning_counts_into_scores(self):
        d=pd.DataFrame({c:[0.,1.,2.] for c in self.mod.FLAVORS})
        x=self.mod.normalized_flavors(d)
        self.assertFalse(x.flavor_usable.iloc[0]);self.assertTrue(x.flavor_usable.iloc[1:].all())
        self.assertTrue(x.filter(like='prop_').iloc[0].isna().all())
        np.testing.assert_allclose(x.filter(like='prop_').iloc[1:].sum(axis=1),1.)
        d.loc[1,'Sour']=-1.
        with self.assertRaises(ValueError):self.mod.normalized_flavors(d)

    def test_semantic_mapping_preserves_kriek_subset_and_does_not_merge_other_fruit_lambics(self):
        rows=pd.DataFrame({'Style':['Tripel','Lambic - Fruit','Lambic - Fruit','Lambic - Gueuze'],
            'Name':['T','Cherry Kriek','Peach fruit','G'],'Description':['']*4,'key':[1,2,3,4]})
        mapping=pd.DataFrame({'official_style':['Tripel','Kriek','Lambic'],
            'kaggle_style':['Tripel','Lambic - Fruit (Kriek/Cherry/Cerise)','Lambic - Gueuze'],
            'subset':['','kriek',''],'primary_pool':[1,1,1]})
        selected=self.mod.select_mapped(rows,mapping,['Tripel','Kriek','Lambic'])
        self.assertEqual(selected['Tripel'].key.tolist(),[1]);self.assertEqual(selected['Kriek'].key.tolist(),[2])
        self.assertEqual(selected['Lambic'].key.tolist(),[4])

    def test_cluster_bootstrap_reuses_brewery_weights_across_styles(self):
        groups={'a':pd.DataFrame({'Brewery':['shared','other'],'v':[0.,1.]}),
            'b':pd.DataFrame({'Brewery':['shared','other'],'v':[2.,3.]})}
        estimates=self.mod.cluster_profiles(groups,['v'],100,seed=11)
        np.testing.assert_allclose(estimates['b']-estimates['a'],2.,atol=1e-12)
        again=self.mod.cluster_profiles(groups,['v'],100,seed=11)
        np.testing.assert_array_equal(estimates['a'],again['a'])

    def test_uncertain_external_direction_does_not_become_a_model_failure(self):
        self.assertEqual(self.mod.direction_status(-.1,.2,.5),'external_uncertain')
        self.assertEqual(self.mod.direction_status(.1,.2,.5),'direction_agrees')
        self.assertEqual(self.mod.direction_status(.1,.2,-.5),'direction_disagrees')
        self.assertEqual(self.mod.direction_status(.1,.2,0.),'model_tie')


if __name__=='__main__':unittest.main()
