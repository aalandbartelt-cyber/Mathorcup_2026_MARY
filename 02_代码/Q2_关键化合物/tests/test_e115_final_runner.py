from pathlib import Path
import sys
import unittest
import pandas as pd

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
from run_e115_final_nnls import matching_fold_top3


class E115FinalRunnerTests(unittest.TestCase):
    def test_top3_comes_only_from_matching_outer_training_fold(self):
        rows=[]
        for repeat,fold,prefix in [(0,0,'a'),(0,1,'b')]:
            for target in ['t1','t2']:
                for rank in range(1,5):
                    rows.append({'repeat':repeat,'fold':fold,'target_code':target,
                                 'feature_code':f'{prefix}{target}{rank}','fold_rank':rank})
        effects=pd.DataFrame(rows)
        chosen=matching_fold_top3(effects,repeat=0,fold=1,target_codes=['t1','t2'])
        self.assertEqual(chosen['t1'],['bt11','bt12','bt13'])
        self.assertEqual(chosen['t2'],['bt21','bt22','bt23'])
        self.assertTrue(all(value.startswith('b') for values in chosen.values() for value in values))

    def test_incomplete_matching_fold_is_rejected(self):
        effects=pd.DataFrame({
            'repeat':[0,0],'fold':[0,0],'target_code':['t','t'],
            'feature_code':['a','b'],'fold_rank':[1,2],
        })
        with self.assertRaisesRegex(ValueError,'Top-3'):
            matching_fold_top3(effects,repeat=0,fold=0,target_codes=['t'])


if __name__=='__main__':
    unittest.main()
