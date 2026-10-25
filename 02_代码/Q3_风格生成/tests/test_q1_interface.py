from pathlib import Path
import sys,tempfile,json,hashlib
import numpy as np
import unittest
SRC=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(SRC))
from q1_interface import FrozenQ1V5OOF

class Q1InterfaceTests(unittest.TestCase):
    def test_requires_final_nnls_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=root/'03_结果/Q1_感官评分';r.mkdir(parents=True);p=r/'oof.npz';a=np.zeros((10,250,50));np.savez(p,prediction=a,mean_prediction=np.zeros((250,50)),target_code=np.arange(50))
            reg={'final_model':'感官组收缩NNLS','oof_path':'03_结果/Q1_感官评分/oof.npz','oof_sha256':hashlib.sha256(p.read_bytes()).hexdigest()};(r/'final_model_registry.json').write_text(json.dumps(reg))
            q=FrozenQ1V5OOF(root);self.assertEqual(q.predictions.shape,(10,250,50))
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):FrozenQ1V5OOF(root)
    def test_old_name_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=root/'03_结果/Q1_感官评分';r.mkdir(parents=True);(r/'final_model_registry.json').write_text(json.dumps({'final_model':'ExtraTrees'}))
            with self.assertRaises(ValueError):FrozenQ1V5OOF(root)
