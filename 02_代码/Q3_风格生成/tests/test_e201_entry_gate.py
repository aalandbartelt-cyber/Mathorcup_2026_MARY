from pathlib import Path
import sys,tempfile,unittest,json,hashlib
from unittest.mock import patch
import pandas as pd
SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
import run_e201_empirical_baselines as runner

class EntryGateTests(unittest.TestCase):
    def fixture(self,root,*,completed=1,scope='outer_train_fold'):
        graph=root/'.worktrees/q2-evidence-opt/03_结果/Q2_关键化合物';slot=graph/'q2_graph_checkpoints_E113/R00F0';slot.mkdir(parents=True)
        (graph/'q2_progress_E113.json').write_text(json.dumps({'completed_folds':completed,'n_features':218,'n_outer_folds':30}))
        hashes={}
        for name in ['summary','edges','membership']:
            p=slot/(name+'.csv.gz');pd.DataFrame({'scope':[scope]}).to_csv(p,index=False,compression='gzip');hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        (slot/'manifest.json').write_text(json.dumps({'files':hashes}));return slot
    def test_verified_first_checkpoint_unlocks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            with patch.object(runner,'CANONICAL',root):self.assertEqual(runner.verify_entry_gate()['checkpoint'],'R00F0')
    def test_no_completed_fold_does_not_unlock(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,completed=0)
            with patch.object(runner,'CANONICAL',root),self.assertRaises(ValueError):runner.verify_entry_gate()
    def test_full_descriptive_graph_does_not_unlock(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,scope='full_data_descriptive')
            with patch.object(runner,'CANONICAL',root),self.assertRaises(ValueError):runner.verify_entry_gate()

if __name__=='__main__':unittest.main()
