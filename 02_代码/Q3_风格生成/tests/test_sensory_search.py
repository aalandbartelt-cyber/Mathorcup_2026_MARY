from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sensory_search import constrained_paths,select_candidates


class SensorySearchTests(unittest.TestCase):
    def test_paths_never_raise_a_known_constraint_and_do_not_insert_zero_lambda(self):
        source=np.array([2.,.2,-1.]);anchors=np.array([[4.,.8,-.5],[1.,.1,-2.]])
        values,anchor_ids,lambdas=constrained_paths(source,anchors,[.25,.5,1.],[0,1],
            np.array([100.,.2]),np.array([True,True]),np.array([19.,.09]),['log10','raw'])
        self.assertEqual(values.shape,(6,3));self.assertTrue((lambdas>0).all())
        self.assertTrue((values[:,0]<=2.+1e-12).all());self.assertTrue((values[:,1]<=.2+1e-12).all())
        self.assertEqual(set(anchor_ids),{0,1})
        with self.assertRaises(ValueError):constrained_paths(source,anchors,[0.,1.],[0,1],
            np.array([100.,.2]),np.array([True,True]),np.array([19.,.09]),['log10','raw'])

    def test_selection_preserves_all_failures_as_rejections(self):
        gains=np.array([.005,.02,.1]);change=np.array([.01,.3,.7]);mean=np.array([-.2,.1,-.1])
        q90=np.array([-.1,.2,.05]);support=np.ones(3)
        selected,feasible=select_candidates(gains,change,mean,q90,support)
        self.assertFalse(feasible.any());self.assertEqual(set(selected.values()),{None})

    def test_fixed_rules_use_primary_model_only_and_deterministic_ties(self):
        gain=np.array([.1,.2,.15,.2]);change=np.array([.3,.6,.5,.6]);mean=np.array([-.1,-.1,-.3,-.1])
        q90=np.array([-.02,-.03,-.15,-.03]);support=np.ones(4)
        selected,mask=select_candidates(gain,change,mean,q90,support)
        self.assertTrue(mask.all())
        self.assertEqual(selected,{'minimum_change':0,'maximum_target_gain':1,'maximum_proxy_margin':2})


if __name__=='__main__':unittest.main()
