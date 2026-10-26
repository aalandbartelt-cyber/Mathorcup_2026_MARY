from pathlib import Path
import sys
import unittest

import pandas as pd

SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from audit_e209_lambic import stronger_status,validate_selection


class LambicAuditTests(unittest.TestCase):
    def setUp(self):
        self.screen={'proxy_status':'predicted_nonworsening','primary_mean_delta':-.05,'audit_mean_delta':-.02,
            'primary_repeat_q90_delta':-.03,'audit_repeat_q90_delta':-.015}
        self.kw=dict(n_failed=0,n_unknown=0,gain=.2,change=.5,support=1.)

    def test_stronger_margin_keeps_both_means_and_quantiles(self):
        self.assertEqual(stronger_status(self.screen,**self.kw),'eligible_statistical_candidate')
        weak=dict(self.screen,audit_repeat_q90_delta=-.005)
        self.assertEqual(stronger_status(weak,**self.kw),'proxy_margin_insufficient')
        self.assertEqual(stronger_status(weak,**self.kw,margin=0.),'eligible_statistical_candidate')
        bad=dict(self.screen,proxy_status='proxy_unreliable')
        self.assertEqual(stronger_status(bad,**self.kw),'proxy_unreliable')

    def test_unknown_direct_measurement_never_converts_to_success(self):
        self.assertEqual(stronger_status(self.screen,**dict(self.kw,n_unknown=1)),'needs_direct_verification')
        self.assertEqual(stronger_status(self.screen,**dict(self.kw,n_failed=1)),'rejected_direct_constraint')
        self.assertNotEqual(stronger_status(self.screen,**dict(self.kw,change=1.1)),'eligible_statistical_candidate')

    def test_complete_denominator_includes_sources_with_no_available_candidate(self):
        selected=pd.DataFrame([{'selection_id':f'{b}_Lambic_{rule}','target_style':'Lambic','scenario':'C1','candidate_available':False}
            for b in range(11) for rule in ['minimum_change','maximum_target_gain','maximum_proxy_margin']])
        spectra=pd.DataFrame(columns=['selection_id','x'])
        validate_selection(selected,spectra,['x'],range(11))
        with self.assertRaises(ValueError):validate_selection(selected.iloc[:-3],spectra,['x'],range(11))
        with self.assertRaises(ValueError):validate_selection(selected,pd.DataFrame([{'selection_id':'phantom','x':1}]),['x'],range(11))


if __name__=='__main__':unittest.main()
