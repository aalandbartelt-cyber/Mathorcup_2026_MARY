"""E115 hierarchical attribution using the frozen final Q1 NNLS predictor.

This runner never searches hyperparameters, refits NNLS weights, or uses the
legacy ExtraTrees-only attribution. Module membership is fixed by the fold-local
E113 graph without validation-target gating; field conditional permutations are
performed on observed validation values while missing masks remain missing.
"""
from pathlib import Path
import hashlib,json,time,platform
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from q2_contract import RESULT_DIR,load_q2
from q2_pipeline import load_frozen_outer_folds
from hierarchical_importance import fit_conditional_feature_model,residual_permutation
from q1_final_adapter import FrozenQ1FinalModel,raw_module_permutation

CANONICAL=Path('/home/mary/mathorcup')
ADAPTER_READY=True

def _modules(result_dir,repeat,fold,feature_codes):
    membership=pd.read_csv(Path(result_dir)/'q2_module_membership_E113.csv.gz')
    selected=membership.loc[membership.scope.eq('outer_train_fold')&membership.repeat.eq(repeat)&membership.fold.eq(fold)]
    modules=[g.feature_code.astype(str).tolist() for _,g in selected.groupby('module_id',sort=True)]
    flat=[x for m in modules for x in m]
    if sorted(flat)!=sorted(feature_codes):raise ValueError('E113 fold modules do not cover strict candidates')
    return modules

def _fold_field_rows(model,x_train,x_valid,y_valid,style_train,style_valid,modules,feature_codes,target_codes,repeat,fold,seed,repeats=10):
    rows=[];feature_position={v:i for i,v in enumerate(model.feature_names)}
    baseline=model.predict(x_valid);base_mae=np.mean(np.abs(y_valid-baseline),axis=0)
    strict_position={v:i for i,v in enumerate(feature_codes)}
    for module_id,members in enumerate(modules,1):
        selected=[feature_position[v] for v in members]
        # Module-level permutation is raw-input and keeps missing values attached.
        for permutation_repeat in range(repeats):
            changed_prediction,order=raw_module_permutation(model,x_valid,members,strict_feature_codes=feature_codes,rng=np.random.default_rng(seed+module_id*1000+permutation_repeat))
            delta=np.mean(np.abs(y_valid-changed_prediction),axis=0)-base_mae
            for target_position,target in enumerate(target_codes):rows.append({'repeat':repeat,'fold':fold,'module_id':f'M{module_id:03d}','module_size':len(members),'module_members':'|'.join(members),'target_code':target,'permutation_repeat':permutation_repeat,'delta_mae':delta[target_position],'module_source':'E113_fold_local_all_modules_no_validation_gate'})
        for feature in members:
            pos=feature_position[feature];peers=[feature_position[v] for v in members if v!=feature]
            train_feature=x_train.iloc[:,pos].to_numpy(float);observed=np.isfinite(train_feature)
            valid_feature=x_valid.iloc[:,pos].to_numpy(float);valid_observed=np.isfinite(valid_feature)
            if len(peers) and observed.sum()>=20 and valid_observed.any():
                cond_imputer=SimpleImputer(strategy='median',keep_empty_features=True)
                train_cond=cond_imputer.fit_transform(x_train.iloc[:,peers].to_numpy(float))
                valid_cond=cond_imputer.transform(x_valid.iloc[:,peers].to_numpy(float))
                cond=fit_conditional_feature_model(train_cond[observed],train_feature[observed],style_train[observed])
                valid_pos=np.flatnonzero(valid_observed);conditional=cond.predict(valid_cond[valid_pos],style_valid[valid_pos])
                residual=valid_feature[valid_pos]-conditional
                for permutation_repeat in range(repeats):
                    changed=x_valid.copy();changed_values=valid_feature.copy();changed_values[valid_pos]=residual_permutation(conditional,residual,np.random.default_rng(seed+pos*1000+permutation_repeat));changed.iloc[:,pos]=changed_values
                    delta=np.mean(np.abs(y_valid-model.predict(changed)),axis=0)-base_mae
                    for target_position,target in enumerate(target_codes):rows.append({'repeat':repeat,'fold':fold,'module_id':f'M{module_id:03d}','module_size':len(members),'module_members':'|'.join(members),'feature_code':feature,'target_code':target,'permutation_repeat':permutation_repeat,'delta_mae':delta[target_position],'module_source':'E113_fold_local_all_modules_no_validation_gate','conditional_scope':'observed_only_missing_mask_preserved'})
    return pd.DataFrame(rows)

def _sufficiency_rows(model,x_train,x_valid,y_valid,top3,target_codes,repeat,fold):
    baseline=model.predict(x_valid);med=np.nanmedian(x_train.to_numpy(float),axis=0);rows=[];positions={v:i for i,v in enumerate(model.feature_names)}
    for target in target_codes:
        features=top3.get(target,[]);changed=x_valid.copy()
        for feature in features:changed.iloc[:,positions[feature]]=med[positions[feature]]
        altered=model.predict(changed);pos=target_codes.index(target)
        rows.append({'repeat':repeat,'fold':fold,'target_code':target,'top3_features':'|'.join(features),'baseline_mae':float(np.mean(np.abs(y_valid[:,pos]-baseline[:,pos]))),'masked_mae':float(np.mean(np.abs(y_valid[:,pos]-altered[:,pos]))),'delta_mae_masked_minus_baseline':float(np.mean(np.abs(y_valid[:,pos]-altered[:,pos]))-np.mean(np.abs(y_valid[:,pos]-baseline[:,pos]))),'model':'frozen_final_nnls_fixed_feature_mask','mask_value':'outer_training_median','top3_source':'E114_outer_training_partial_effect'})
    return pd.DataFrame(rows)

def run(*,output_dir=RESULT_DIR,repeat_ids=None):
    if not ADAPTER_READY:raise RuntimeError('final NNLS adapter is not ready')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);repeat_ids=list(range(10)) if repeat_ids is None else sorted(set(repeat_ids))
    x,y,style,beer=load_q2();feature_codes=pd.read_csv(RESULT_DIR/'q2_candidate_dictionary.csv').query('eligible_strict == True').feature_code.astype(str).tolist();target_codes=y.columns.astype(str).tolist()
    final=FrozenQ1FinalModel(CANONICAL);final.validate_data(x,y,beer.to_numpy(),style.to_numpy(str))
    folds=[f for f in load_frozen_outer_folds(len(x),10) if f.repeat in repeat_ids]
    field_frames=[];module_frames=[];suff_frames=[];cache=output_dir/'q2_nnls_cache_E115';cache.mkdir(exist_ok=True)
    for number,outer in enumerate(folds,1):
        model,info=final.reconstruct_fold(x,y,beer.to_numpy(),style.to_numpy(str),outer,cache_dir=cache,q2_result_dir=output_dir)
        x_train=x.iloc[outer.train_index];x_valid=x.iloc[outer.validation_index];y_valid=y.to_numpy(float)[outer.validation_index]
        modules=_modules(output_dir,outer.repeat,outer.fold,feature_codes)
        rows=_fold_field_rows(model,x_train,x_valid,y_valid,style.iloc[outer.train_index].to_numpy(str),style.iloc[outer.validation_index].to_numpy(str),modules,feature_codes,target_codes,outer.repeat,outer.fold,20261025+outer.repeat*100+outer.fold)
        module_frames.append(rows[rows.feature_code.isna() if 'feature_code' in rows else np.ones(len(rows),dtype=bool)])
        field_frames.append(rows[rows.feature_code.notna()])
        e114=pd.read_csv(RESULT_DIR/'q2_partial_effect_rankings_E114.csv');top3={t:g.sort_values(['partial_effect_rank','feature_code']).head(3).feature_code.astype(str).tolist() for t,g in e114.groupby('target_code')}
        suff_frames.append(_sufficiency_rows(model,x_train,x_valid,y_valid,top3,target_codes,outer.repeat,outer.fold));print(f'[{number:02d}/{len(folds)}] {outer.split_id} complete max_reconstruction_error={info["max_abs_error"]:.3g}',flush=True)
    modules=pd.concat(module_frames,ignore_index=True);fields=pd.concat(field_frames,ignore_index=True);suff=pd.concat(suff_frames,ignore_index=True)
    module_summary=modules.groupby(['repeat','fold','module_id','module_size','module_members','target_code','module_source'],as_index=False).agg(median_delta_mae=('delta_mae','median'),positive_fraction=('delta_mae',lambda x:float((x>0).mean())),n_permutations=('delta_mae','size'))
    field_summary=fields.groupby(['repeat','fold','module_id','module_size','module_members','feature_code','target_code','module_source','conditional_scope'],as_index=False).agg(median_delta_mae=('delta_mae','median'),positive_fraction=('delta_mae',lambda x:float((x>0).mean()),),n_permutations=('delta_mae','size'))
    paths={'modules':output_dir/'q2_final_nnls_module_importance_E115.csv.gz','fields':output_dir/'q2_final_nnls_conditional_importance_E115.csv.gz','sufficiency':output_dir/'q2_final_nnls_sufficiency_E115.csv','manifest':output_dir/'q2_run_manifest_E115_final_nnls.json'}
    module_summary.to_csv(paths['modules'],index=False,compression='gzip');field_summary.to_csv(paths['fields'],index=False,compression='gzip');suff.to_csv(paths['sufficiency'],index=False)
    manifest={'experiment_id':'E115_final_nnls','status':'complete' if len(repeat_ids)==10 else 'pilot_complete','repeat_ids':repeat_ids,'n_outer_folds':len(folds),'n_features_full':231,'n_features_strict':218,'n_targets':50,'model':'frozen_final_group_NNLS','module_gate':'E113_fold_local_all_modules_no_validation_gate','sufficiency':'fixed_final_NNLS_feature_mask_using_E114_outer_training_Top3','q1_retrained':False,'nnls_refit':False,'hyperparameter_search':False,'gpu_used':False,'adapter_signature':final.signature,'outputs':[str(v) for k,v in paths.items() if k!='manifest']}
    paths['manifest'].write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');return manifest

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repeats',nargs='+',type=int,default=list(range(10)));a=p.parse_args();print(json.dumps(run(repeat_ids=a.repeats),ensure_ascii=False,indent=2))
