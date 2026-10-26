from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit_e205b_sensory_search import verify_committed_lock


class OnceOnlyAuditTests(unittest.TestCase):
    def test_locked_candidates_must_be_committed_and_hash_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'results';folder.mkdir();candidate=folder/'candidates.csv'
            candidate.write_text('id,value\n1,0.5\n')
            lock=folder/'lock.json';record={'status':'locked_before_model_family_audit','auditor_accessed':False,
                'output_sha256':{'candidates.csv':hashlib.sha256(candidate.read_bytes()).hexdigest()}}
            blob=json.dumps(record).encode();lock.write_bytes(blob)
            with patch('audit_e205b_sensory_search.subprocess.check_output',return_value=blob):
                self.assertEqual(verify_committed_lock(root,lock),record)
                lock.write_text(json.dumps(dict(record,auditor_accessed=True)))
                with self.assertRaises(ValueError):verify_committed_lock(root,lock)
                lock.write_bytes(blob);candidate.write_text('id,value\n1,0.8\n')
                with self.assertRaises(ValueError):verify_committed_lock(root,lock)

    def test_auditor_scored_search_cannot_masquerade_as_a_locked_blind_search(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lock=root/'lock.json'
            blob=json.dumps({'status':'locked_before_model_family_audit','auditor_accessed':True,'output_sha256':{}}).encode()
            lock.write_bytes(blob)
            with patch('audit_e205b_sensory_search.subprocess.check_output',return_value=blob):
                with self.assertRaises(ValueError):verify_committed_lock(root,lock)


if __name__=='__main__':unittest.main()
