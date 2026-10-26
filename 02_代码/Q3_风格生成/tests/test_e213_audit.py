import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd
import yaml

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))


class NeuralAuditTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('audit_e213'),'audit not implemented')
        from audit_e213 import assess
        self.assess=assess;self.config=yaml.safe_load((SRC.parent/'configs/E213_neural_interpolation.yaml').read_text())
        from run_e213_neural_interpolation import ALL_METHODS
        from run_e208_joint_generation import METRICS
        rows=[];clips=[]
        gains={'style_copula_shrunk':0.,'raw_mixture':-.004,'pca_noise':-.006,'neural_noise':-.009,'neural_curve':-.01,'neural_selected':-.01}
        for method in ALL_METHODS:
            for style in self.config['target_styles']:
                for repeat in range(10):
                    for scale in ['raw','robust']:
                        d={k:0. for k in METRICS};d.update(mean_w1=1.,energy_v=1.+gains[method],coverage=.5+(method!='style_copula_shrunk')*.06,
                            log_spread_error=.1,pairwise_spread_ratio=1.,density_k1=1.,collapsed=False)
                        rows.append(dict(method=method,style=style,repeat=repeat,metric_scale=scale,**d))
                    if method not in ['style_copula_shrunk','raw_mixture']:
                        clips.append(dict(method=method,style=style,repeat=repeat,local_cell_clipping=.01))
        self.r=pd.DataFrame(rows);self.clips=pd.DataFrame(clips)

    def test_neural_requires_real_increment_over_both_controls(self):
        decision,*_=self.assess(self.r,self.r,self.clips,self.config)
        self.assertEqual(decision['selected_distribution'],'neural_selected')
        r=self.r.copy();r.loc[r.method.eq('neural_selected'),'energy_v']+=.004
        decision,*_=self.assess(r,r,self.clips,self.config)
        self.assertEqual(decision['selected_distribution'],'pca_noise')
        self.assertFalse(decision['neural_increment_supported'])

    def test_high_clipping_rejects_an_apparent_gain(self):
        clips=self.clips.copy();clips['local_cell_clipping']=.3
        decision,*_=self.assess(self.r,self.r,clips,self.config)
        self.assertEqual(decision['selected_distribution'],'style_copula_shrunk')

    def test_unchanged_rules_are_enforced(self):
        config=copy.deepcopy(self.config);config['replacement_rules']['robust_energy_min_decrease']=.001
        with self.assertRaises(ValueError):self.assess(self.r,self.r,self.clips,config)

    def test_formal_coverage_checks_exact_fold_style_seed_keys(self):
        from audit_e213 import validate_coverage
        from itertools import product
        from run_e213_neural_interpolation import ALL_METHODS
        keys=['repeat','fold','style','method','seed_index','metric_scale']
        full=pd.DataFrame(product(range(10),range(3),self.config['target_styles'],ALL_METHODS,range(5),['raw','robust']),columns=keys)
        validate_coverage(full,self.config)
        with self.assertRaises(ValueError):validate_coverage(full.iloc[:-1],self.config)
        bad=full.copy();bad.loc[bad['fold'].eq(2),'fold']=3
        with self.assertRaises(ValueError):validate_coverage(bad,self.config)
        config=copy.deepcopy(self.config);config['neural_increment']['references']=['raw_mixture']
        with self.assertRaises(ValueError):self.assess(self.r,self.r,self.clips,config)


if __name__=='__main__':unittest.main()
