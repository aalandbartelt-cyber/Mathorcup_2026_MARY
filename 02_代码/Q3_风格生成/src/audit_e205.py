"""Verify provenance, out-of-fold diagnostics and non-certification boundaries."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sensory_audit import sha,candidate_status,verify_archived_prediction

ROOT=Path(__file__).resolve().parents[3]
RESULT=ROOT/'03_结果/Q3_风格生成'
KEYS=['beer','target_style','scenario','strategy','lambda']


def validate_candidate_table(frame):
    if len(frame)!=891 or frame.duplicated(KEYS).any():raise ValueError('Candidate identities incomplete')
    if frame.physical_safety_certified.any():raise ValueError('Model predictions cannot certify physical safety')
    for row in frame.itertuples():
        available=bool(row.candidate_available)
        expected=candidate_status(available=available,n_failed=float(row.n_failed) if available else 0,
            n_unknown=float(row.n_unknown) if available else 0,target_gain=bool(row.target_gain) if available else False,
            proxy_status=row.proxy_status,support_fraction=float(row.support_repeat_fraction) if available else 0.)
        if row.status!=expected or bool(row.eligible)!=(expected=='eligible_statistical_candidate'):
            raise ValueError('Candidate status bypassed registered evidence gates')
        if available and row.eligible and (not row.primary_proxy_reliable or not row.audit_proxy_reliable):
            raise ValueError('Unreliable proxy marked eligible')
    if frame[frame.strategy=='source_unchanged'].eligible.any():raise ValueError('Zero migration cannot count as success')


def run():
    manifest_path=RESULT/'q3_run_manifest_E205.json'
    m=json.loads(manifest_path.read_text())
    if m.get('status')!='complete' or m.get('n_outer_folds')!=30 or m.get('n_repeats')!=10:
        raise ValueError('Completed 30-fold E205 required')
    if any(m.get(k) is not False for k in ['q1_retrained','nnls_refit','hyperparameter_search','gpu_used','physical_safety_certified']):
        raise ValueError('E205 frozen model or certification policy violated')
    for path,digest in m['input_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Frozen input changed: {path}')
    for name,digest in m['output_sha256'].items():
        if sha(RESULT/name)!=digest:raise ValueError(f'Output changed: {name}')
    cp=RESULT/'q3_checkpoints_E205'
    for name,digest in m['checkpoint_sha256'].items():
        if sha(cp/name)!=digest:raise ValueError(f'Checkpoint changed: {name}')
    oof_path=next(Path(p) for p in m['input_sha256'] if p.endswith('publication_final/final_model_oof.npz'))
    auditor_path=next(Path(p) for p in m['input_sha256'] if p.endswith('q2_oof_predictions_E101_E104_E109_wide.npz'))
    with np.load(oof_path,allow_pickle=False) as a:final={k:a[k] for k in a.files}
    with np.load(auditor_path,allow_pickle=False) as a:auditor=a['E103_ElasticNet']
    migration=pd.read_csv(RESULT/'q3_migration_spectra_E204.csv.gz')
    mapping={int(v):i for i,v in enumerate(final['beer'])};positions=np.array([mapping[int(v)] for v in migration.beer])
    checkpoint_rows=[]
    for repeat in range(10):
        migration_indices=[]
        for fold in range(3):
            name=f'R{repeat:02d}F{fold}';meta=json.loads((cp/(name+'.json')).read_text())
            if meta['signature']!=m['signature']:raise ValueError('Checkpoint signature differs from final manifest')
            valid=np.flatnonzero(final['fold_assignment'][repeat]==fold)
            train=np.flatnonzero(final['fold_assignment'][repeat]!=fold)
            e202_dir=RESULT/f'q3_checkpoints_E202/{name}'
            e202=json.loads((e202_dir/'manifest.json').read_text())
            if (e202['files']['generated.csv.gz']!=sha(e202_dir/'generated.csv.gz')
                or e202['train_beer']!=final['beer'][train].tolist()
                or e202['validation_beer']!=final['beer'][valid].tolist()):
                raise ValueError('E202 generation input differs from frozen outer training split')
            expected_migration=np.flatnonzero(final['fold_assignment'][repeat,positions]==fold)
            with np.load(cp/(name+'.npz'),allow_pickle=False) as a:
                if not np.array_equal(a['validation_index'],valid) or not np.array_equal(a['migration_index'],expected_migration):
                    raise ValueError('Prediction checkpoint changed source-excluded fold identities')
                if np.max(np.abs(a['primary_validation']-final['prediction'][repeat,valid]))>1e-10:
                    raise ValueError('Q1 frozen OOF not reproduced')
                verify_archived_prediction(a['auditor_validation'],auditor[repeat,valid])
                if a['primary_generated'].shape!=(900,50):raise ValueError('Generated profile count changed')
                if any(not np.isfinite(a[k]).all() for k in a.files if a[k].dtype.kind in 'fc'):
                    raise ValueError('Non-finite checkpoint value')
                if (a['grade_generated'].min()<0 or a['grade_generated'].max()>6 or
                    a['grade_migration'].min()<0 or a['grade_migration'].max()>6):
                    raise ValueError('Relative grade outside 0..6')
                migration_indices.extend(a['migration_index'].tolist())
            checkpoint_rows.append({'repeat':repeat,'fold':fold,'n_validation':len(valid),'n_migration':len(expected_migration),
                'primary_max_abs_error':meta['q1_cache']['reconstruction_max_abs_error'],
                'auditor_raw_quantization_error':meta['auditor_reconstruction_max_abs_error'],
                'auditor_archive_exact':True,'e202_training_split_verified':True})
        if sorted(migration_indices)!=list(range(844)):raise ValueError('Migration candidates missing or duplicated within repeat')
    proxy=pd.read_csv(RESULT/'q3_migration_proxy_audit_E205.csv');validate_candidate_table(proxy)
    frozen_summary=pd.read_csv(RESULT/'q3_migration_summary_E204.csv')
    matched=proxy.merge(frozen_summary,on=KEYS,validate='one_to_one',suffixes=('','_frozen'))
    for col in ['candidate_available','target_gain','n_failed','n_unknown','pareto_front']:
        if not matched[col].equals(matched[col+'_frozen']):raise ValueError(f'E204 frozen result changed: {col}')
    profiles=pd.read_csv(RESULT/'q3_generated_profiles_E205.csv.gz')
    summary=pd.read_csv(RESULT/'q3_generated_profile_summary_E205.csv')
    migration_profiles=pd.read_csv(RESULT/'q3_migration_sensory_profiles_E205.csv.gz')
    if (len(profiles)!=45000 or len(summary)!=150 or len(migration_profiles)!=42200
        or profiles.groupby('style').sample_id.nunique().to_dict()!={'Kriek':300,'Lambic':300,'Tripel':300}
        or profiles.duplicated(['style','sample_id','target_code']).any()
        or migration_profiles.duplicated(KEYS+['target_code']).any()):
        raise ValueError('Profile identities/target coverage incomplete')
    unchanged=migration_profiles[migration_profiles.strategy=='source_unchanged']
    zero_error=float(unchanged[['prediction_delta','audit_prediction_delta']].abs().to_numpy().max())
    if zero_error>1e-9:raise ValueError('Unchanged source has nonzero sensory gain')
    biases=pd.read_csv(RESULT/'q3_outer_profile_bias_E205.csv')
    if len(biases)!=450 or not biases.groupby(['repeat','fold','style']).seed_index.nunique().eq(5).all():
        raise ValueError('Outer generated/real profile comparison incomplete')
    checkpoints_path=RESULT/'q3_e205_checkpoint_audit.csv';pd.DataFrame(checkpoint_rows).to_csv(checkpoints_path,index=False)
    report={'status':'complete','n_outer_folds':30,'n_generated_samples':900,'n_generated_profile_values':len(profiles),
        'n_style_target_profile_summary':len(summary),'n_migration_summary':len(proxy),
        'n_migration_profile_values':len(migration_profiles),'n_outer_profile_comparisons':len(biases),
        'candidate_status_counts':{str(k):int(v) for k,v in proxy.status.value_counts().items()},
        'n_eligible_statistical_candidates':int(proxy.eligible.sum()),
        'n_eligible_c1_source_target_pairs':len(proxy[proxy.eligible&proxy.scenario.eq('C1')][['beer','target_style']].drop_duplicates()),
        'max_unchanged_source_prediction_delta':zero_error,
        'frozen_input_hash_count':len(m['input_sha256']),
        'frozen_inputs_unchanged':True,'all_e202_generation_folds_verified':True,
        'primary_max_abs_reconstruction_error':m['max_primary_reconstruction_error'],
        'auditor_float32_archive_exact_all_folds':True,
        'physical_safety_certified':False,'q1_retrained':False,'nnls_refit':False,
        'scope':m['scope'],'source_sha256':sha(__file__),'input_manifest_sha256':sha(manifest_path),
        'output_sha256':{checkpoints_path.name:sha(checkpoints_path)}}
    out=RESULT/'q3_e205_audit_manifest.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':run()
