from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
from q1_final_adapter import FrozenQ1FinalModel, FrozenFoldNNLS, fit_frozen_base_models, raw_module_permutation


class Fixture:
    def __init__(self,root):
        self.root=Path(root);self.result=self.root/'03_结果/Q1_感官评分';self.result.mkdir(parents=True)
        self.program=self.root/'05_论文/Q1_感官评分/程序';self.program.mkdir(parents=True)
        self.attachments=self.root/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件';self.attachments.mkdir(parents=True)
        rng=np.random.default_rng(19)
        self.x=pd.DataFrame(rng.normal(size=(250,231)),columns=[f'f{i}' for i in range(231)])
        self.y=pd.DataFrame(rng.normal(size=(250,50)),columns=[f't{i}' for i in range(50)])
        self.beer=np.arange(250);self.style=np.repeat('S',250)
        for name,data in [('1.csv',self.x),('4.csv',self.y)]:
            frame=data.copy();frame.insert(0,'tasting_category_fine',self.style);frame.insert(0,'beer_id',self.beer);frame.insert(0,'beer',self.beer);frame.to_csv(self.attachments/name,index=False)
        self.assignment=np.tile(np.arange(250)%3,(10,1))
        public=self.result/'publication_final';public.mkdir()
        self.oof=public/'final_model_oof.npz'
        np.savez_compressed(self.oof,y_true=self.y.to_numpy(),prediction=np.tile(self.y.to_numpy()[None],(10,1,1)),mean_prediction=self.y.to_numpy(),beer=self.beer,style=self.style,target_code=self.y.columns.to_numpy(str),fold_assignment=self.assignment)
        self.registry={'final_model':'感官组收缩NNLS','oof_path':str(self.oof.relative_to(self.root)),'oof_sha256':hashlib.sha256(self.oof.read_bytes()).hexdigest(),'n_repeats':10,'outer_folds':3,'n_targets':50}
        self.registry_path=self.result/'final_model_registry.json';self.registry_path.write_text(json.dumps(self.registry))
        config={'base_models':['extra_trees','hgb','ridge_quantile','pls2_quantile'],'extra_trees':{'n_estimators':4,'max_depth':None}}
        (self.program/'model_config.json').write_text(json.dumps(config))
        bases={'extra_trees':{'parameter':{'max_features':1.,'min_samples_leaf':1}},'hgb':{'parameter':{'name':'tiny','max_leaf_nodes':3,'min_samples_leaf':5,'learning_rate':.05,'l2_regularization':1.,'max_iter':3}},'ridge_quantile':{'parameter':1.},'pls2_quantile':{'parameter':2}}
        weights=np.zeros((50,4));weights[:,0]=1.
        params=[{'repeat':r,'fold':f,'base_selection_json':json.dumps(bases),'nnls_parameters_json':json.dumps({'group':{'weights':weights.tolist(),'intercept':[0.]*50,'global_weights':[1.,0.,0.,0.]}})} for r in range(10) for f in range(3)]
        self.parameters=self.result/'q1_selected_parameters_E009.csv';pd.DataFrame(params).to_csv(self.parameters,index=False)
        audit=self.result/'audit_E009';audit.mkdir()
        (audit/'formal_audit.json').write_text(json.dumps({'integrity':{'source_artifact_sha256':{str(self.parameters.relative_to(self.root)):hashlib.sha256(self.parameters.read_bytes()).hexdigest()}}}))


class FrozenInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.fixture=Fixture(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_registry_hash_is_required(self):
        self.fixture.oof.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash|SHA'):
            FrozenQ1FinalModel(self.fixture.root)
    def test_old_baseline_is_rejected(self):
        f=self.fixture;f.registry['final_model']='ExtraTrees';f.registry_path.write_text(json.dumps(f.registry))
        with self.assertRaisesRegex(ValueError,'final|NNLS'):
            FrozenQ1FinalModel(f.root)
    def test_parameter_tampering_is_rejected(self):
        self.fixture.parameters.write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash|SHA'):
            FrozenQ1FinalModel(self.fixture.root)
    def test_identity_and_fold_membership_are_validated(self):
        f=self.fixture;source=FrozenQ1FinalModel(f.root)
        source.validate_data(f.x,f.y,f.beer,f.style)
        valid=np.flatnonzero(f.assignment[0]==0);train=np.flatnonzero(f.assignment[0]!=0)
        source.validate_fold(0,0,train,valid)
        with self.assertRaises(ValueError):source.validate_fold(0,0,train[:-1],valid)
        with self.assertRaises(ValueError):source.validate_data(f.x.iloc[:,::-1],f.y,f.beer,f.style)
        with self.assertRaises(ValueError):source.validate_data(f.x,f.y,f.beer[::-1],f.style)
    def test_read_only_predictions_cannot_change_frozen_source(self):
        s=FrozenQ1FinalModel(self.fixture.root);pred=s.fold_prediction(0,0)
        self.assertFalse(pred.flags.writeable)
        with self.assertRaises(ValueError):pred[0,0]=99.
    def test_weights_and_intercepts_have_no_simplex_or_positive_intercept_constraint(self):
        class Model:
            def predict(self,x):return np.column_stack([x[:,0],x[:,1]])
        w=np.array([[1.2,.4,.3,.1],[.1,.2,.1,.9]])
        m=FrozenFoldNNLS([Model() for _ in range(4)],w,np.array([-3.,2.]),['a','b'])
        x=np.array([[1.,np.nan],[2.,3.]])
        # NaNs are deliberately handled by each learner, not by a shared ET transform.
        out=m.predict(np.nan_to_num(x));self.assertEqual(out.shape,(2,2))
        np.testing.assert_allclose(out[0],[-1.,2.])
        with self.assertRaises(ValueError):m.predict(pd.DataFrame([[1.,2.]],columns=['b','a']))
    def test_q1_cache_location_is_rejected_before_any_fit(self):
        f=self.fixture;s=FrozenQ1FinalModel(f.root)
        outer=SimpleNamespace(repeat=0,fold=0,train_index=np.flatnonzero(f.assignment[0]!=0),validation_index=np.flatnonzero(f.assignment[0]==0))
        with patch('q1_final_adapter.Q2_RESULTS',f.root/'03_结果/Q2_关键化合物'),patch('q1_final_adapter.fit_frozen_base_models',side_effect=AssertionError('must not fit')):
            with self.assertRaisesRegex(ValueError,'Q2'):
                s.reconstruct_fold(f.x,f.y,f.beer,f.style,outer,cache_dir=f.result/'bad_cache')
    def test_old_cache_signature_is_rejected_before_model_deserialization(self):
        f=self.fixture;s=FrozenQ1FinalModel(f.root)
        allowed=f.root/'03_结果/Q2_关键化合物';cache=allowed/'models';cache.mkdir(parents=True)
        (cache/'R00F0.joblib').write_bytes(b'not a model')
        (cache/'R00F0.json').write_text(json.dumps({'signature':'old'}))
        outer=SimpleNamespace(repeat=0,fold=0,train_index=np.flatnonzero(f.assignment[0]!=0),validation_index=np.flatnonzero(f.assignment[0]==0))
        with patch('q1_final_adapter.fit_frozen_base_models',side_effect=AssertionError('must not fit')):
            with self.assertRaisesRegex(ValueError,'signature'):
                s.reconstruct_fold(f.x,f.y,f.beer,f.style,outer,cache_dir=cache,q2_result_dir=allowed)

    def test_q1_paper_tree_cannot_be_used_as_cache_root_override(self):
        f=self.fixture;s=FrozenQ1FinalModel(f.root)
        outer=SimpleNamespace(repeat=0,fold=0,train_index=np.flatnonzero(f.assignment[0]!=0),validation_index=np.flatnonzero(f.assignment[0]==0))
        paper=f.root/'05_论文/Q1_感官评分'
        with self.assertRaisesRegex(ValueError,'canonical Q2'):
            s.reconstruct_fold(f.x,f.y,f.beer,f.style,outer,cache_dir=paper/'cache',q2_result_dir=paper)


class FrozenBaseFitTests(unittest.TestCase):
    def test_raw_module_permutation_moves_missing_masks_and_leaves_context_fixed(self):
        class MissingSensitiveModel:
            def predict(self,x):
                return (np.nan_to_num(x[:,0],nan=-10.)+100*np.isnan(x[:,0])+x[:,2])[:,None]
        model=FrozenFoldNNLS([MissingSensitiveModel() for _ in range(4)],[[1.,0.,0.,0.]],[0.],['a','b','aggregate'])
        x=np.array([[1.,0.,10.],[np.nan,1.,20.],[3.,2.,30.],[4.,3.,40.]])
        prediction,order=raw_module_permutation(model,x,['a'],strict_feature_codes=['a','b'],rng=np.random.default_rng(8))
        expected=np.nan_to_num(x[order,0],nan=-10.)+100*np.isnan(x[order,0])+x[:,2]
        np.testing.assert_array_equal(prediction[:,0],expected)
        with self.assertRaisesRegex(ValueError,'strict'):
            raw_module_permutation(model,x,['aggregate'],strict_feature_codes=['a','b'],rng=np.random.default_rng(8))
    def test_frozen_factories_match_q1_reference_predictions_and_preserve_missing_masks(self):
        from sklearn.model_selection import train_test_split
        import importlib.util
        root=Path('/home/mary/mathorcup/05_论文/Q1_感官评分/程序')
        before=sys.dont_write_bytecode;sys.dont_write_bytecode=True
        try:
            spec=importlib.util.spec_from_file_location('frozen_baseline_reference',root/'baseline_models.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
            spec=importlib.util.spec_from_file_location('frozen_hgb_reference',root/'ensemble_models.py');hgb=importlib.util.module_from_spec(spec);spec.loader.exec_module(hgb)
        finally:sys.dont_write_bytecode=before
        rng=np.random.default_rng(4);x=rng.normal(size=(35,5));x[::5,0]=np.nan;y=rng.normal(size=(35,2))
        train=np.arange(26);val=np.arange(26,35);seed=123
        parameters={'extra_trees':{'max_features':.6,'min_samples_leaf':2},'hgb':{'name':'tiny','max_leaf_nodes':3,'min_samples_leaf':5,'learning_rate':.05,'l2_regularization':1.,'max_iter':3},'ridge_quantile':1.,'pls2_quantile':2}
        config={'base_models':['extra_trees','hgb','ridge_quantile','pls2_quantile'],'extra_trees':{'n_estimators':4,'max_depth':None}}
        models=fit_frozen_base_models(x[train],y[train],parameters,config,seed=seed)
        expected=[ref.fit_predict_extra_trees(x[train],y[train],x[val],n_estimators=4,max_features=.6,min_samples_leaf=2,max_depth=None,random_state=seed,n_jobs=1),hgb.fit_predict_hgb_multioutput(x[train],y[train],x[val],**{k:v for k,v in parameters['hgb'].items() if k!='name'},random_state=seed+100000),ref.fit_predict_ridge_quantile(x[train],y[train],x[val],alpha=1.,random_state=seed+200000),ref.fit_predict_pls_quantile(x[train],y[train],x[val],n_components=2,random_state=seed+300000)]
        for model,truth in zip(models,expected):np.testing.assert_allclose(model.predict(x[val]),truth,atol=1e-12,rtol=0)

if __name__=='__main__':unittest.main()
