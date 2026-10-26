from pathlib import Path
import hashlib
import json
import unittest

import pandas as pd

RESULT=Path(__file__).resolve().parents[3]/'03_结果/Q3_风格生成'


class FrozenCandidateArtifactTests(unittest.TestCase):
    def test_locked_search_audit_and_sensitivity_outputs_have_consistent_hashes(self):
        for filename in ['q3_candidate_lock_E205B.json','q3_run_manifest_E205B.json','q3_run_manifest_E207.json']:
            manifest=json.loads((RESULT/filename).read_text())
            for name,digest in manifest['output_sha256'].items():
                with self.subTest(manifest=filename,output=name):
                    self.assertEqual(hashlib.sha256((RESULT/name).read_bytes()).hexdigest(),digest)
        search=json.loads((RESULT/'q3_candidate_lock_E205B.json').read_text())
        audit=json.loads((RESULT/'q3_run_manifest_E205B.json').read_text())
        self.assertFalse(search['auditor_accessed'])
        self.assertTrue(audit['no_auditor_based_reselection'])
        self.assertFalse(audit['physical_safety_certified'])

    def test_primary_denominator_unknowns_and_margin_sensitivity_remain_visible(self):
        primary=pd.read_csv(RESULT/'q3_primary_candidates_E207.csv')
        self.assertEqual(len(primary),33)
        self.assertEqual(primary.groupby('target_style').size().to_dict(),{'Kriek':11,'Lambic':11,'Tripel':11})
        self.assertFalse(primary.physical_safety_certified.any())
        accepted=primary[primary.eligible]
        self.assertTrue(accepted.n_unknown.eq(0).all() and accepted.n_failed.eq(0).all())
        self.assertTrue(accepted.relative_target_gain.ge(.01).all())
        margin=pd.read_csv(RESULT/'q3_proxy_margin_sensitivity_E207.csv')
        self.assertEqual(len(margin),99)
        self.assertEqual(set(margin.margin_z),{0.,.01,.02})
        self.assertEqual(set(margin.selection_id),set(primary.selection_id))
        # A stronger diagnostic must not overwrite the original selected rows.
        zero=margin[margin.margin_z.eq(0)].set_index('selection_id')
        self.assertTrue(zero.loc[accepted.selection_id,'both_proxy_nonworsening'].all())


if __name__=='__main__':unittest.main()
