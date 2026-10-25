"""Read-only Q1 v5 interface for Q3; no old baseline fallback."""
from pathlib import Path
import hashlib,json
import numpy as np

class FrozenQ1V5OOF:
    def __init__(self,root=Path('/home/mary/mathorcup')):
        self.root=Path(root).resolve();self.result=self.root/'03_结果/Q1_感官评分'
        self.registry=json.loads((self.result/'final_model_registry.json').read_text())
        if self.registry.get('final_model')!='感官组收缩NNLS':raise ValueError('Q3 requires frozen final NNLS Q1 interface')
        path=(self.root/self.registry['oof_path']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.registry['oof_sha256']:raise ValueError('Q1 v5 OOF hash mismatch')
        with np.load(path,allow_pickle=False) as a:self.arrays={k:a[k].copy() for k in a.files}
        if self.arrays['prediction'].shape!=(10,250,50):raise ValueError('Q1 final OOF dimensions changed')
        self.arrays['prediction'].setflags(write=False);self.arrays['mean_prediction'].setflags(write=False)
        self.signature=self.registry['oof_sha256']

    @property
    def predictions(self):return self.arrays['prediction']
    @property
    def mean_prediction(self):return self.arrays['mean_prediction']
    @property
    def target_codes(self):return self.arrays['target_code'].astype(str)
