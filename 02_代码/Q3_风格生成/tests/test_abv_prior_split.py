import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))


class ABVSplitTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('prepare_abv_prior_split'),'split not implemented')
        from prepare_abv_prior_split import build_split
        self.build=build_split

    def test_brewery_group_stays_together_across_styles_and_normalization(self):
        frame=pd.DataFrame({'key':[1,2,3,4],'Brewery':[' Beer A ','beer a','B','C'],'ABV':[9.,6.,5.,6.]})
        groups={'Tripel':frame.iloc[[0,2]],'Lambic':frame.iloc[[1,3]]}
        result,excluded=self.build(groups,seed=20261026)
        a=result[result.external_key.isin(['1','2'])]
        self.assertEqual(a.brewery_sha256.nunique(),1);self.assertEqual(a.split.nunique(),1);self.assertEqual(len(excluded),0)
        other,_=self.build({s:g.iloc[::-1] for s,g in reversed(list(groups.items()))},seed=20261026)
        self.assertTrue(result.equals(other));self.assertNotIn('ABV',result)

    def test_invalid_abv_missing_brewery_and_duplicate_keys_are_not_silently_used(self):
        frame=pd.DataFrame({'key':range(5),'Brewery':['a','b',None,'d','e'],'ABV':[0.,21.,7.,float('nan'),8.]})
        result,excluded=self.build({'Tripel':frame},seed=20261026)
        self.assertEqual(result.external_key.tolist(),['4']);self.assertEqual(len(excluded),4)
        with self.assertRaises(ValueError):self.build({'Tripel':pd.concat([frame,frame])},seed=20261026)


if __name__=='__main__':unittest.main()
