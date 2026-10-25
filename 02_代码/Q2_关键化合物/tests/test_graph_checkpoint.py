from pathlib import Path
import sys,tempfile,unittest
import pandas as pd
SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
from graph_checkpoint import save_fold_checkpoint,load_fold_checkpoint

class GraphCheckpointTests(unittest.TestCase):
    def tables(self):
        return {'summary':pd.DataFrame([{'scope':'outer_train_fold','repeat':0,'fold':0}]),
                'alpha_scores':pd.DataFrame([{'alpha':.1}]),
                'edges':pd.DataFrame([{'scope':'outer_train_fold','source':'a','target':'b','partial_correlation':.2}]),
                'membership':pd.DataFrame([{'scope':'outer_train_fold','feature_code':'a','module_id':'M001'},
                                           {'scope':'outer_train_fold','feature_code':'b','module_id':'M001'}])}
    def test_roundtrip_and_changed_identity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.assertIsNone(load_fold_checkpoint(p,'R00F0',{'signature':'one'}))
            save_fold_checkpoint(p,'R00F0',{'signature':'one'},self.tables())
            got=load_fold_checkpoint(p,'R00F0',{'signature':'one'})
            self.assertEqual(got['membership'].module_id.tolist(),['M001','M001'])
            with self.assertRaises(ValueError):load_fold_checkpoint(p,'R00F0',{'signature':'two'})
    def test_corrupt_file_is_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);save_fold_checkpoint(p,'R00F0',{},self.tables())
            (p/'R00F0/summary.csv.gz').write_bytes(b'broken')
            with self.assertRaises(ValueError):load_fold_checkpoint(p,'R00F0',{})
    def test_descriptive_scope_never_enters_fold_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            tables=self.tables();tables['edges']['scope']='full_data_descriptive'
            with self.assertRaises(ValueError):save_fold_checkpoint(Path(d),'R00F0',{},tables)

if __name__=='__main__':unittest.main()
