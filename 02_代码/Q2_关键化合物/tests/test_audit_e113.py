from pathlib import Path
import sys,tempfile,json
import pandas as pd
import unittest
SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from audit_e113 import audit

class E113AuditTests(unittest.TestCase):
    def test_partial_hash_and_scope_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);slot=root/'q2_graph_checkpoints_E113/R00F0';slot.mkdir(parents=True)
            (root/'q2_progress_E113.json').write_text(json.dumps({'status':'running','completed_folds':1}))
            tables={'summary':pd.DataFrame([{'split_id':'R00F0','scope':'outer_train_fold'}]),
                    'edges':pd.DataFrame([{'scope':'outer_train_fold','source':'a','target':'b','partial_correlation':.2}]),
                    'membership':pd.DataFrame({'scope':['outer_train_fold']*218,'feature_code':[f'f{i}' for i in range(218)],'module_id':['M001']*218})}
            import hashlib
            hashes={}
            for name,frame in tables.items():
                path=slot/(name+'.csv.gz');frame.to_csv(path,index=False,compression='gzip');hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            (slot/'manifest.json').write_text(json.dumps({'files':hashes}))
            report=audit(root);self.assertEqual(report['errors'],[]);self.assertEqual(report['completed_folds'],1)
    def test_descriptive_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);slot=root/'q2_graph_checkpoints_E113/R00F0';slot.mkdir(parents=True)
            (root/'q2_progress_E113.json').write_text(json.dumps({'status':'running','completed_folds':1}))
            import hashlib
            frames={'summary':pd.DataFrame([{'split_id':'R00F0','scope':'outer_train_fold'}]),'edges':pd.DataFrame([{'scope':'full_data_descriptive','source':'a','target':'b','partial_correlation':.2}]),'membership':pd.DataFrame({'scope':['outer_train_fold']*218,'feature_code':[f'f{i}' for i in range(218)],'module_id':['M001']*218})}
            h={}
            for n,f in frames.items():p=slot/(n+'.csv.gz');f.to_csv(p,index=False,compression='gzip');h[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
            (slot/'manifest.json').write_text(json.dumps({'files':h}));self.assertTrue(audit(root)['errors'])
if __name__=='__main__':unittest.main()
