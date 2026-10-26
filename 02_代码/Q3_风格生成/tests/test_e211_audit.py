import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd
import yaml

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))


class BasisAuditTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('audit_e211'),'basis audit not implemented')
        from audit_e211 import assess
        self.assess=assess;self.config=yaml.safe_load((SRC.parent/'configs/E211_basis_mixture.yaml').read_text())
        rows=[]
        for method,(w1,energy,coverage,spread) in {'style_copula_shrunk':(.5,.6,.5,.2),
            'raw_mixture':(.49,.59,.57,.22),'pca_mixture':(.488,.588,.57,.22),'basis_mixture':(.48,.58,.60,.21)}.items():
            for style in ['Tripel','Kriek','Lambic']:
                for repeat in range(10):
                    for scale in ['raw','robust']:
                        rows.append({'method':method,'style':style,'repeat':repeat,'metric_scale':scale,'mean_w1':w1,
                            'energy_v':energy,'coverage':coverage,'log_spread_error':spread,'near_duplicate_fraction':0.,
                            'collapsed':False,'density_k1':1.,'pairwise_spread_ratio':.9,'spearman_pair_rmse':.5})
        self.table=pd.DataFrame(rows);self.raw=self.table.copy()
        self.diag=pd.DataFrame([{'repeat':r,'fold':f,'style':s,'basis_effective':True} for r in range(10)
            for f in range(3) for s in ['Tripel','Kriek','Lambic']])

    def test_basis_requires_both_overall_gain_and_increment_over_two_simpler_methods(self):
        decision,_,_=self.assess(self.table,self.raw,self.diag,self.config)
        self.assertEqual(decision['selected_distribution'],'basis_mixture');self.assertTrue(decision['basis_increment_supported'])
        weak=self.table.copy();weak.loc[weak.method.eq('basis_mixture'),'energy_v']=.589
        decision,_,_=self.assess(weak,self.raw,self.diag,self.config)
        self.assertFalse(decision['basis_increment_supported']);self.assertEqual(decision['selected_distribution'],'raw_mixture')

    def test_failed_simple_mixture_and_ineffective_basis_keep_original_copula(self):
        table=self.table.copy();table.loc[table.method.eq('raw_mixture'),'log_spread_error']=.3
        diag=self.diag.copy();diag.loc[diag['style'].eq('Lambic'),'basis_effective']=False
        decision,_,_=self.assess(table,self.raw,diag,self.config)
        self.assertEqual(decision['selected_distribution'],'style_copula_shrunk')
        self.assertFalse(decision['basis_increment_supported'])

    def test_outer_copying_rejects_basis_even_if_means_improve(self):
        raw=self.raw.copy();raw.loc[raw.method.eq('basis_mixture'),'near_duplicate_fraction']=.1
        decision,_,_=self.assess(self.table,raw,self.diag,self.config)
        self.assertEqual(decision['selected_distribution'],'raw_mixture')

    def test_frozen_increment_and_fallback_limits_cannot_change(self):
        config=dict(self.config,basis_increment_rules=dict(self.config['basis_increment_rules'],energy_better_repeats=6))
        with self.assertRaises(ValueError):self.assess(self.table,self.raw,self.diag,config)
        config=dict(self.config,replacement_rules=dict(self.config['replacement_rules'],robust_coverage_min_increase=0.))
        with self.assertRaises(ValueError):self.assess(self.table,self.raw,self.diag,config)


if __name__=='__main__':unittest.main()
