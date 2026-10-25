"""Read the frozen final Q1 interface; reconstruct selected learners only in Q2.

No parameter search, weight fitting, Q1 output writes or baseline fallback occurs.
Every predictor accepts the original 231 columns. Missing values are processed
inside each learner's own frozen pipeline, including its missing indicators.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

BASE_MODELS=('extra_trees','hgb','ridge_quantile','pls2_quantile')
FINAL_NAME='感官组收缩NNLS'
VALIDATION_ATOL=1e-10
Q2_RESULTS=Path(__file__).resolve().parents[3]/'03_结果/Q2_关键化合物'


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _quantile_preprocessor(n_train,seed):
    return FeatureUnion([
        ('continuous',Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('quantile',QuantileTransformer(n_quantiles=min(100,n_train),output_distribution='normal',subsample=None,random_state=seed)),
        ])),
        ('missing_indicator',MissingIndicator(features='missing-only',error_on_new=False)),
    ])


@dataclass
class TargetwiseHGB:
    imputer: SimpleImputer
    models: list

    def predict(self,values):
        transformed=self.imputer.transform(values)
        return np.column_stack([model.predict(transformed) for model in self.models])


def fit_frozen_base_models(x_train,y_train,parameters,config,*,seed):
    """Fit exactly four preselected learners, without any parameter selection."""
    if tuple(config['base_models'])!=BASE_MODELS or set(parameters)!=set(BASE_MODELS):
        raise ValueError('final NNLS must use the four frozen base learners')
    x,y=np.asarray(x_train,float),np.asarray(y_train,float)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not np.isfinite(y).all() or np.isinf(x).any():
        raise ValueError('finite targets and raw possibly-missing predictors must align')
    fitted=[]
    for position,name in enumerate(BASE_MODELS):
        random_state=int(seed)+position*100000
        parameter=parameters[name]
        if name=='extra_trees':
            model=Pipeline([
                ('imputer',SimpleImputer(strategy='median',add_indicator=True)),
                ('model',ExtraTreesRegressor(n_estimators=int(config['extra_trees']['n_estimators']),
                    max_features=float(parameter['max_features']),min_samples_leaf=int(parameter['min_samples_leaf']),
                    max_depth=config['extra_trees']['max_depth'],random_state=random_state,n_jobs=1)),
            ])
            model.fit(x,y)
        elif name=='hgb':
            imputer=SimpleImputer(strategy='median',add_indicator=True)
            transformed=imputer.fit_transform(x);models=[]
            for target in range(y.shape[1]):
                estimator=HistGradientBoostingRegressor(**{k:v for k,v in parameter.items() if k!='name'},random_state=random_state+target)
                estimator.fit(transformed,y[:,target]);models.append(estimator)
            model=TargetwiseHGB(imputer,models)
        elif name=='ridge_quantile':
            model=Pipeline([
                ('preprocessor',_quantile_preprocessor(len(x),random_state)),
                ('scaler',StandardScaler()),('model',Ridge(alpha=float(parameter))),
            ])
            model.fit(x,y)
        else:
            model=Pipeline([
                ('preprocessor',_quantile_preprocessor(len(x),random_state)),
                ('model',PLSRegression(n_components=int(parameter),scale=True,max_iter=1000)),
            ])
            model.fit(x,y)
        fitted.append(model)
    return fitted


class FrozenFoldNNLS:
    """A four-learner raw-input predictor with already-shrunk frozen weights."""

    def __init__(self,models,weights,intercept,feature_names):
        self.models=list(models);self.weights=np.asarray(weights,float).copy()
        self.intercept=np.asarray(intercept,float).copy();self.feature_names=tuple(feature_names)
        if (len(self.models)!=4 or self.weights.ndim!=2 or self.weights.shape[1]!=4
                or self.intercept.shape!=(self.weights.shape[0],) or (self.weights<0).any()
                or not np.isfinite(self.weights).all() or not np.isfinite(self.intercept).all()):
            raise ValueError('invalid frozen weights or free target intercepts')
        self.weights.setflags(write=False);self.intercept.setflags(write=False)

    def predict(self,values):
        if isinstance(values,pd.DataFrame) and tuple(values.columns.astype(str))!=self.feature_names:
            raise ValueError('raw feature order differs from frozen Q1')
        x=np.asarray(values,float)
        if x.ndim!=2 or x.shape[1]!=len(self.feature_names) or np.isinf(x).any():
            raise ValueError('predict expects the complete ordered raw feature matrix')
        stack=np.stack([np.asarray(model.predict(x),float) for model in self.models],axis=-1)
        if stack.shape[1:]!=self.weights.shape or not np.isfinite(stack).all():
            raise ValueError('base predictions disagree with frozen target/model dimensions')
        return np.einsum('ntm,tm->nt',stack,self.weights)+self.intercept


def raw_module_permutation(model,values,module_codes,*,strict_feature_codes,rng):
    """Move raw module values and masks together, holding other fields fixed.

    The other 13 fields remain conditioning context. This is a predictive
    permutation diagnostic, not a chemically coherent causal intervention.
    """
    names=list(model.feature_names);members=list(module_codes)
    if not members or len(set(members))!=len(members) or not set(members)<=set(strict_feature_codes):
        raise ValueError('module members must be distinct strict compound fields')
    if isinstance(values,pd.DataFrame) and tuple(values.columns.astype(str))!=tuple(names):
        raise ValueError('raw feature order differs from frozen Q1')
    x=np.asarray(values,float)
    if x.ndim!=2 or x.shape[1]!=len(names) or not len(x):raise ValueError('complete raw input required')
    positions=[names.index(name) for name in members]
    order=rng.permutation(len(x));changed=x.copy();changed[:,positions]=x[order][:,positions]
    return model.predict(changed),order


class FrozenQ1FinalModel:
    """Hash-bound, identity-checked reader of the user-frozen final Q1 outputs."""

    def __init__(self,root):
        self.root=Path(root).resolve();self.result=self.root/'03_结果/Q1_感官评分'
        self.registry_path=self.result/'final_model_registry.json'
        self.registry=json.loads(self.registry_path.read_text())
        if (self.registry.get('final_model')!=FINAL_NAME or self.registry.get('n_repeats')!=10
                or self.registry.get('outer_folds')!=3 or self.registry.get('n_targets')!=50):
            raise ValueError('registry is not the frozen final group NNLS 10x3 interface')
        self.oof_path=(self.root/self.registry['oof_path']).resolve()
        if not self.oof_path.is_relative_to(self.result) or _sha(self.oof_path)!=self.registry['oof_sha256']:
            raise ValueError('final OOF SHA256 hash or location differs from registry')
        self.parameter_path=self.result/'q1_selected_parameters_E009.csv'
        audit=json.loads((self.result/'audit_E009/formal_audit.json').read_text())
        recorded=audit['integrity']['source_artifact_sha256'][str(self.parameter_path.relative_to(self.root))]
        if _sha(self.parameter_path)!=recorded:raise ValueError('frozen parameter SHA256 hash mismatch')
        with np.load(self.oof_path,allow_pickle=False) as archive:
            self.arrays={key:archive[key].copy() for key in ['y_true','prediction','mean_prediction','beer','style','target_code','fold_assignment']}
        if (self.arrays['prediction'].shape!=(10,250,50) or self.arrays['y_true'].shape!=(250,50)
                or self.arrays['mean_prediction'].shape!=(250,50) or self.arrays['fold_assignment'].shape!=(10,250)):
            raise ValueError('frozen Q1 sample/target/repeat dimensions differ')
        if not np.isfinite(self.arrays['prediction']).all() or not np.isfinite(self.arrays['y_true']).all():
            raise ValueError('frozen Q1 predictions or targets are non-finite')
        if not np.allclose(self.arrays['mean_prediction'],self.arrays['prediction'].mean(axis=0),rtol=0,atol=1e-14):
            raise ValueError('mean OOF differs from its ten repeats')
        if not all(set(row.tolist())=={0,1,2} for row in self.arrays['fold_assignment']):
            raise ValueError('every Q1 repeat must contain exactly three validation folds')
        for value in self.arrays.values():value.setflags(write=False)
        self.target_codes=tuple(self.arrays['target_code'].astype(str))
        self.parameter_table=pd.read_csv(self.parameter_path)
        pairs=set(map(tuple,self.parameter_table[['repeat','fold']].to_numpy()))
        if len(self.parameter_table)!=30 or pairs!={(r,f) for r in range(10) for f in range(3)}:
            raise ValueError('frozen NNLS parameter table must contain all 30 unique folds')
        program=self.root/'05_论文/Q1_感官评分/程序'
        self.config_path=program/'model_config.json';self.config=json.loads(self.config_path.read_text())
        if tuple(self.config['base_models'])!=BASE_MODELS:raise ValueError('frozen base-model order differs')
        public_hashes=self.result/'publication_audit/artifact_sha256.json'
        if public_hashes.exists():
            hashes=json.loads(public_hashes.read_text());key=str(self.config_path.relative_to(self.root))
            if key not in hashes or hashes[key]!=_sha(self.config_path):raise ValueError('final configuration SHA256 hash mismatch')
        attachments=self.root/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件'
        chemistry=pd.read_csv(attachments/'1.csv')
        self.feature_names=tuple(c for c in chemistry.columns if c not in ['beer','beer_id','tasting_category_fine'])
        if len(self.feature_names)!=231 or len(set(self.feature_names))!=231:
            raise ValueError('frozen predictor requires exactly 231 distinct raw input fields')
        self.reference_x=chemistry.loc[:,self.feature_names].to_numpy(float)
        self.signature=hashlib.sha256(json.dumps({'registry':_sha(self.registry_path),'oof':_sha(self.oof_path),
            'parameters':_sha(self.parameter_path),'config':_sha(self.config_path),
            'raw_chemistry':_sha(attachments/'1.csv'),'adapter':_sha(__file__)},sort_keys=True).encode()).hexdigest()

    def validate_data(self,x,y,beer,style):
        if tuple(x.columns.astype(str))!=self.feature_names or tuple(y.columns.astype(str))!=self.target_codes:
            raise ValueError('Q2 feature/target order differs from frozen Q1')
        if not np.array_equal(np.asarray(beer),self.arrays['beer']) or not np.array_equal(np.asarray(style).astype(str),self.arrays['style'].astype(str)):
            raise ValueError('Q2 beer/style identity or order differs from frozen Q1')
        if not np.allclose(np.asarray(x,float),self.reference_x,rtol=0,atol=1e-12,equal_nan=True):
            raise ValueError('Q2 chemistry values or missing masks differ from frozen Q1')
        if not np.allclose(np.asarray(y,float),self.arrays['y_true'],rtol=0,atol=1e-12):
            raise ValueError('Q2 sensory labels differ from frozen Q1')

    def validate_fold(self,repeat,fold,train_index,validation_index):
        if repeat not in range(10) or fold not in range(3):raise ValueError('invalid frozen repeat/fold')
        expected=np.flatnonzero(self.arrays['fold_assignment'][repeat]==fold)
        train=np.flatnonzero(self.arrays['fold_assignment'][repeat]!=fold)
        if not np.array_equal(np.asarray(validation_index),expected) or not np.array_equal(np.asarray(train_index),train):
            raise ValueError('Q2 training/validation rows differ from frozen Q1 fold')

    def fold_prediction(self,repeat,fold):
        if repeat not in range(10) or fold not in range(3):raise ValueError('invalid frozen repeat/fold')
        out=self.arrays['prediction'][repeat,self.arrays['fold_assignment'][repeat]==fold].copy()
        out.setflags(write=False);return out

    def fold_parameters(self,repeat,fold):
        row=self.parameter_table.loc[(self.parameter_table['repeat']==repeat)&(self.parameter_table['fold']==fold)]
        if len(row)!=1:raise ValueError('no unique frozen fold parameters')
        selections=json.loads(row.iloc[0]['base_selection_json'])
        parameters={name:selections[name]['parameter'] for name in BASE_MODELS}
        nnls=json.loads(row.iloc[0]['nnls_parameters_json'])['group']
        return parameters,np.asarray(nnls['weights'],float),np.asarray(nnls['intercept'],float)

    def reconstruct_fold(self,x,y,beer,style,outer,*,cache_dir,q2_result_dir=None):
        """Rebuild fixed outer-fold learners for Q2 permutation, never Q1 search."""
        self.validate_data(x,y,beer,style)
        self.validate_fold(outer.repeat,outer.fold,outer.train_index,outer.validation_index)
        allowed=Q2_RESULTS.resolve() if q2_result_dir is None else Path(q2_result_dir).resolve()
        if allowed.name!='Q2_关键化合物' or allowed.parent.name!='03_结果':
            raise ValueError('q2_result_dir must be the canonical Q2 result root')
        cache=Path(cache_dir).resolve()
        if not cache.is_relative_to(allowed) or cache.is_relative_to(self.result):
            raise ValueError('predictor caches must be written under the Q2 result directory')
        cache.mkdir(parents=True,exist_ok=True)
        prefix=cache/f'R{outer.repeat:02d}F{outer.fold}'
        model_path=prefix.with_suffix('.joblib');meta_path=prefix.with_suffix('.json')
        expected=self.fold_prediction(outer.repeat,outer.fold)
        reused=False
        if model_path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if meta.get('signature')!=self.signature or meta.get('model_sha256')!=_sha(model_path):
                raise ValueError('Q2 cache signature or model hash differs; do not silently reuse')
            model=joblib.load(model_path);reused=True
        else:
            params,weights,intercept=self.fold_parameters(outer.repeat,outer.fold)
            models=fit_frozen_base_models(x.iloc[outer.train_index],y.iloc[outer.train_index],params,self.config,
                                         seed=20261023+outer.repeat*10+outer.fold)
            model=FrozenFoldNNLS(models,weights,intercept,self.feature_names)
        prediction=model.predict(x.iloc[outer.validation_index])
        error=float(np.max(np.abs(prediction-expected)))
        if not np.isfinite(error) or error>VALIDATION_ATOL:
            raise ValueError(f'NNLS reconstruction differs from frozen OOF: max_abs_error={error}; limit={VALIDATION_ATOL}')
        if not reused:
            temporary=model_path.with_suffix('.joblib.tmp');joblib.dump(model,temporary);temporary.replace(model_path)
            meta={'signature':self.signature,'model_sha256':_sha(model_path),'repeat':outer.repeat,'fold':outer.fold,
                  'validation_max_abs_error':error,'validation_atol':VALIDATION_ATOL,
                  'hyperparameter_search':False,'nnls_refit':False,'q1_outputs_written':False,
                  'scope':'Q2_reconstruction_of_frozen_outer_training_predictor'}
            temporary=meta_path.with_suffix('.json.tmp');temporary.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');temporary.replace(meta_path)
        return model,{'cache_reused':reused,'max_abs_error':error,'signature':self.signature,'cache_path':str(model_path)}
