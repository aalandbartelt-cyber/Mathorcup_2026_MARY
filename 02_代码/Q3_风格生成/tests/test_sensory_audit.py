from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sensory_audit import (verify_cache_files, migration_model_input,
                          proxy_screen, candidate_status, support_audit,verify_archived_prediction)


class SensoryAuditTests(unittest.TestCase):
    def test_float32_archive_requires_exact_identity_at_its_storage_precision(self):
        actual=np.array([[.3,.7]],dtype=np.float64);saved=actual.astype(np.float32)
        self.assertGreater(verify_archived_prediction(actual,saved),1e-9)
        with self.assertRaises(ValueError):verify_archived_prediction(actual+.0001,saved)
        with self.assertRaises(ValueError):verify_archived_prediction(actual+.0001,actual.copy())

    def test_missing_cache_is_not_retrained(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                verify_cache_files(d, 0, 0, 'frozen')
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_cache_requires_matching_signature_hash_and_fold_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/'R00F0.joblib';model.write_bytes(b'fixture')
            meta={'signature':'frozen','model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
                  'repeat':0,'fold':0,'nnls_refit':False,'hyperparameter_search':False,'q1_outputs_written':False}
            path=root/'R00F0.json';path.write_text(json.dumps(meta))
            self.assertEqual(verify_cache_files(root,0,0,'frozen')[0],model)
            with self.assertRaises(ValueError):verify_cache_files(root,0,0,'other')
            meta['fold']=1;path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):verify_cache_files(root,0,0,'frozen')
            meta['fold']=0;path.write_text(json.dumps(meta));model.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_cache_files(root,0,0,'frozen')

    def test_source_missing_mask_is_preserved_for_each_candidate(self):
        raw=np.array([[1.,np.nan,3.],[np.nan,2.,5.]])
        values=np.array([[9.,8.,7.],[6.,5.,4.],[3.,2.,1.]])
        output=migration_model_input(values,[0,1,0],raw)
        self.assertTrue(np.isnan(output[0,1]) and np.isnan(output[1,0]) and np.isnan(output[2,1]))
        self.assertEqual(output[0,0],9.)
        self.assertTrue(np.isfinite(values).all())

    def test_zero_migration_restores_raw_source_instead_of_imputed_design(self):
        raw=np.array([[1.,np.nan,3.]])
        design=np.array([[1.,100.,3.]])
        np.testing.assert_allclose(migration_model_input(design,[0],raw),raw,equal_nan=True)

    def test_unreliable_or_disagreeing_proxy_cannot_certify(self):
        self.assertEqual(proxy_screen([-1.]*10,[-1.]*10,False,True)['proxy_status'],'proxy_unreliable')
        self.assertEqual(proxy_screen([-1.]*10,[1.]*10,True,True)['proxy_status'],'proxy_disagreement')
        result=proxy_screen([-1.]*10,[-.5]*10,True,True)
        self.assertEqual(result['proxy_status'],'predicted_nonworsening')
        self.assertFalse(result['physical_safety_certified'])

    def test_repeat_upper_quantile_does_not_hide_worsening(self):
        deltas=np.array([-1.]*8+[.1,.2])
        result=proxy_screen(deltas,deltas,True,True)
        self.assertEqual(result['proxy_status'],'proxy_variation_uncertain')
        self.assertNotIn('confidence_interval',result)

    def test_unknown_chemistry_and_zero_target_gain_remain_unusable(self):
        args=dict(available=True,n_failed=0,n_unknown=0,target_gain=True,
                  proxy_status='predicted_nonworsening',support_fraction=1.)
        self.assertEqual(candidate_status(**args),'eligible_statistical_candidate')
        self.assertEqual(candidate_status(**dict(args,n_unknown=1)),'needs_direct_verification')
        self.assertEqual(candidate_status(**dict(args,target_gain=False)),'no_target_gain')
        self.assertEqual(candidate_status(**dict(args,proxy_status='proxy_unreliable')),'proxy_unreliable')
        self.assertEqual(candidate_status(**dict(args,support_fraction=.8)),'outside_predictor_support')

    def test_support_uses_training_only_and_does_not_clip_extreme_queries(self):
        train=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.],[4.,4.]])
        query=np.array([[2.1,2.1],[10000.,10000.]])
        result=support_audit(train,query)
        self.assertTrue(result['supported'][0])
        self.assertFalse(result['supported'][1])
        self.assertGreater(result['nearest_distance'][1],1000.)
        self.assertEqual(result['outside_range_fraction'][1],1.)


if __name__=='__main__':unittest.main()
