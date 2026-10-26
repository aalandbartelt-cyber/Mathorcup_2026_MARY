from pathlib import Path
import sys
import unittest

import pandas as pd
import yaml

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from audit_e208 import replacement_decision


class JointReplacementTests(unittest.TestCase):
    def setUp(self):
        self.config=yaml.safe_load((SRC.parent/'configs/E208_joint_generation.yaml').read_text())
        rows=[]
        for method in ['joint_selected','style_copula_shrunk']:
            for style in ['Tripel','Kriek','Lambic']:
                for repeat in range(10):
                    for scale in ['raw','robust']:
                        new=method=='joint_selected'
                        rows.append({'method':method,'style':style,'repeat':repeat,'metric_scale':scale,
                            'mean_w1':.49 if new else .5,'energy_v':.48 if new else .5,
                            'coverage':.56 if new else .5,'log_spread_error':.15 if new else .2,
                            'pairwise_spread_ratio':.9,'density_k1':1.,'spearman_pair_rmse':.5,
                            'near_duplicate_fraction':0.,'collapsed':0.})
        self.table=pd.DataFrame(rows)
        self.raw=pd.DataFrame({'method':['joint_selected'],'near_duplicate_fraction':[0.],'collapsed':[False]})

    def test_joint_improvement_passes_only_with_all_registered_safeguards(self):
        decision,_,_=replacement_decision(self.table,self.raw,self.config)
        self.assertTrue(decision['replace_copula'])
        self.assertEqual(decision['n_repeats_robust_energy_better'],10)
        bad=self.raw.copy();bad['near_duplicate_fraction']=.01
        self.assertFalse(replacement_decision(self.table,bad,self.config)[0]['replace_copula'])

    def test_average_gain_cannot_hide_one_style_regression(self):
        table=self.table.copy();mask=table.method.eq('joint_selected')&table['style'].eq('Lambic')&table.metric_scale.eq('robust')
        table.loc[mask,'mean_w1']=.53
        decision,_,_=replacement_decision(table,self.raw,self.config)
        self.assertTrue(decision['checks']['robust_w1_noninferior'])
        self.assertFalse(decision['checks']['no_style_w1_regression'])
        self.assertFalse(decision['replace_copula'])

    def test_positive_mean_gain_cannot_replace_repeat_count_rule(self):
        table=self.table.copy();mask=table.method.eq('joint_selected')&table.metric_scale.eq('robust')
        table.loc[mask,'energy_v']=[.45 if v<6 else .505 for v in table.loc[mask,'repeat']]
        decision,_,_=replacement_decision(table,self.raw,self.config)
        self.assertTrue(decision['checks']['robust_energy_improves'])
        self.assertEqual(decision['n_repeats_robust_energy_better'],6)
        self.assertFalse(decision['replace_copula'])

    def test_registered_limits_cannot_be_changed_after_result(self):
        changed=dict(self.config,replacement_rules=dict(self.config['replacement_rules'],robust_energy_better_repeats=6))
        with self.assertRaises(ValueError):replacement_decision(self.table,self.raw,changed)


if __name__=='__main__':unittest.main()
