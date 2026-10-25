"""E202 Gaussian/Copula baselines; the historical pilot remains immutable."""
from pathlib import Path
import hashlib,json,platform,time,yaml
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from gaussian_baselines import StyleGaussianGenerator
from q3_evaluation import distribution_diagnostics
from q1_interface import FrozenQ1V5OOF
from run_e200_contract import validate_cv

ROOT=Path(__file__).resolve().parents[3];CANONICAL=Path('/home/mary/mathorcup');RESULT=ROOT/'03_结果/Q3_风格生成';CONFIG=Path(__file__).resolve().parents[1]/'configs/E202_gaussian_copula_baselines.yaml'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_e201_gate(result_dir,*,expected_q1_oof_sha256):
    path=Path(result_dir)/'q3_run_manifest_E201.json'
    manifest=json.loads(path.read_text())
    signature=manifest.get('signature',{})
    entry=signature.get('entry_gate',{})
    if manifest.get('experiment_id')!='E201' or manifest.get('status')!='complete' or manifest.get('n_outer_folds')!=30:
        raise ValueError('complete 30-fold E201 manifest required')
    if entry.get('scope')!='outer_train_fold' or entry.get('checkpoint')!='R00F0':
        raise ValueError('E201 must carry the verified fold-local graph gate')
    if signature.get('q1_final_oof_sha256')!=expected_q1_oof_sha256:
        raise ValueError('Q1 final OOF identity changed after E201')
    return {'checkpoint':entry['checkpoint'],'scope':entry['scope'],'e113_checkpoint_sha256':entry.get('sha256'),
            'e201_manifest_sha256':sha(path)}

def validate_formal_config(config):
    expected_methods=['style_gaussian_clipped','style_copula_shrunk']
    if config.get('methods')!=expected_methods:
        raise ValueError('formal E202 methods changed')
    if len(config.get('seeds',[]))!=5 or len(set(config['seeds']))!=5:
        raise ValueError('formal E202 requires five unique seeds')
    if config.get('n_repeats')!=10 or config.get('n_folds')!=3:
        raise ValueError('formal E202 requires frozen 10x3 outer CV')
    if config.get('target_styles')!=['Tripel','Kriek','Lambic']:
        raise ValueError('formal E202 target styles changed')
    if config.get('gpu_used') is not False:
        raise ValueError('formal E202 is a CPU baseline')

def compose_metric_row(evaluation_scope,base,metrics):
    metrics=dict(metrics)
    diagnostic_scope=metrics.pop('scope',None)
    row=dict(metrics);row.update(base)
    row['evaluation_scope']=evaluation_scope
    row['diagnostic_scope']=diagnostic_scope
    return row

def gate():
    e201=json.loads((RESULT/'q3_run_manifest_E201.json').read_text())
    e113=json.loads((CANONICAL/'.worktrees/q2-evidence-opt/03_结果/Q2_关键化合物/q2_progress_E113.json').read_text())
    if e201.get('status')!='complete' or e113.get('status')!='complete':raise ValueError('E201 and E113 complete gates required')
    return {'e201':sha(RESULT/'q3_run_manifest_E201.json'),'e113':e113.get('last_split_id'),'q1_final_oof':FrozenQ1V5OOF(CANONICAL).signature}
def run_pilot():
    config=yaml.safe_load(CONFIG.read_text());signature={'config':sha(CONFIG),'gate':gate()}
    chem=pd.read_csv(ROOT/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv');splits=pd.read_csv(ROOT/'03_结果/Q1_感官评分/q1_cv_splits.csv');features=[c for c in chem if c not in ['beer','beer_id','tasting_category_fine']];x=chem[features].to_numpy(float);styles=chem.tasting_category_fine.to_numpy(str)
    row=splits[(splits.repeat==0)&(splits.fold==0)];valid=np.sort(row.row_position.to_numpy(int));train=np.setdiff1d(np.arange(len(x)),valid);imp=SimpleImputer(strategy='median',keep_empty_features=True);tx=imp.fit_transform(x[train]);vx=imp.transform(x[valid]);model=StyleGaussianGenerator().fit(tx,styles[train],mean_shrinkage=float(config['mean_shrinkage']))
    rows=[]
    for style in config['target_styles']:
        mask=styles[valid]==style
        if mask.sum()<2 or (styles[train]==style).sum()<2:rows.append({'style':style,'status':'NA_insufficient_rows'});continue
        for method in config['methods']:
            sample=model.generate(style,int(mask.sum()),method=method,seed=20261025)
            metrics=distribution_diagnostics(vx[mask],sample['values'],tx[styles[train]==style],np.where(np.std(tx,axis=0)>1e-12,np.std(tx,axis=0),1.))
            metrics.pop('w1_by_feature');rows.append({'scope':'pilot_outer_train_to_validation','split_id':'R00F0','style':style,'method':method,'status':'pilot_only',**metrics})
    out=RESULT/'q3_e202_pilot_metrics.csv';pd.DataFrame(rows).to_csv(out,index=False);decision={'status':'pilot_only_no_formal_run','formal_run_started':False,'q1_retrained':False,'gpu_used':False,'methods':config['methods'],'signature':signature}
    (RESULT/'q3_e202_pilot_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n');print(json.dumps(decision,ensure_ascii=False,indent=2))

def run():
    start=time.perf_counter()
    config=yaml.safe_load(CONFIG.read_text());validate_formal_config(config)
    RESULT.mkdir(parents=True,exist_ok=True)
    q1=FrozenQ1V5OOF(CANONICAL)
    entry_gate=verify_e201_gate(RESULT,expected_q1_oof_sha256=q1.signature)
    attachment=ROOT/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv'
    split_path=ROOT/'03_结果/Q1_感官评分/q1_cv_splits.csv'
    chemistry=pd.read_csv(attachment);splits=pd.read_csv(split_path)
    validate_cv(chemistry,splits,n_repeats=10,n_folds=3)
    features=[column for column in chemistry.columns if column not in ['beer','beer_id','tasting_category_fine']]
    if len(features)!=231 or len(chemistry)!=250:
        raise ValueError('Q3 input contract changed')
    x=chemistry[features].to_numpy(float);styles=chemistry.tasting_category_fine.to_numpy(str)
    signature={
        'input_sha256':sha(attachment),'cv_sha256':sha(split_path),'config_sha256':sha(CONFIG),
        'source_sha256':{path.name:sha(path) for path in [Path(__file__),Path(__file__).with_name('gaussian_baselines.py'),Path(__file__).with_name('q3_evaluation.py')]},
        'q1_final_model':q1.registry['final_model'],'q1_final_oof_sha256':q1.signature,'entry_gate':entry_gate,
    }
    checkpoints=RESULT/'q3_checkpoints_E202';checkpoints.mkdir(exist_ok=True)
    tables=[];generated_files=[]
    grouped=splits.groupby(['repeat','fold'],sort=True)
    for number,((repeat,fold),group) in enumerate(grouped,1):
        repeat=int(repeat);fold=int(fold);split_id=f'R{repeat:02d}F{fold}'
        slot=checkpoints/split_id;slot.mkdir(exist_ok=True);commit=slot/'manifest.json'
        if commit.exists():
            old=json.loads(commit.read_text())
            if old.get('signature')!=signature:
                raise ValueError('Q3 E202 checkpoint signature changed')
            for filename,digest in old['files'].items():
                if sha(slot/filename)!=digest:
                    raise ValueError('Q3 E202 checkpoint damaged')
            tables.append(pd.read_csv(slot/'metrics.csv'));generated_files.append(slot/'generated.csv.gz')
            print(f'[{number:02d}/30] {split_id} resumed',flush=True);continue

        valid=np.sort(group.row_position.to_numpy(int));train=np.setdiff1d(np.arange(len(x)),valid)
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        train_x=imputer.fit_transform(x[train]);valid_x=imputer.transform(x[valid])
        scale=np.std(train_x,axis=0);scale=np.where(scale>1e-12,scale,1.)
        model=StyleGaussianGenerator().fit(train_x,styles[train],mean_shrinkage=float(config['mean_shrinkage']))
        rows=[];generated=[]
        for style_index,style in enumerate(config['target_styles']):
            local=np.flatnonzero(styles[train]==style);real=valid_x[styles[valid]==style]
            if len(real)<2 or len(local)<2:
                raise ValueError('style fold too small for registered diagnostics')
            covariance=model.diagnostics(style)
            for seed_index,base_seed in enumerate(config['seeds']):
                seed=int(base_seed)+repeat*1000+fold*100+style_index*10
                for method in config['methods']:
                    sample=model.generate(style,len(real),method=method,seed=seed)
                    metrics=distribution_diagnostics(real,sample['values'],train_x[local],scale)
                    metrics.pop('w1_by_feature')
                    rows.append(compose_metric_row('outer_validation_evaluation',{
                        'repeat':repeat,'fold':fold,'style':style,'method':method,'seed_index':seed_index,
                        'status':'evaluated','pre_clip_cell_fraction':sample['pre_clip_cell_fraction'],
                        'pre_clip_row_fraction':sample['pre_clip_row_fraction'],**covariance,
                    },metrics))
                    table=pd.DataFrame(sample['values'],columns=features)
                    table.insert(0,'sample_position',np.arange(len(real)))
                    for key,value in [('style',style),('seed_index',seed_index),('method',method),('fold',fold),('repeat',repeat),('scope','outer_training_generated')]:
                        table.insert(0,key,value)
                    generated.append(table)

                rng=np.random.default_rng(seed)
                if len(local)>=2*len(real):
                    order=rng.permutation(local)[:2*len(real)]
                    metrics=distribution_diagnostics(train_x[order[:len(real)]],train_x[order[len(real):]],train_x[local],scale)
                    metrics.pop('w1_by_feature')
                    rows.append(compose_metric_row('outer_training_real_real_reference',{
                        'repeat':repeat,'fold':fold,'style':style,'method':'real_real_reference',
                        'seed_index':seed_index,'status':'evaluated','pre_clip_cell_fraction':np.nan,
                        'pre_clip_row_fraction':np.nan,'gaussian_min_eigenvalue':np.nan,
                        'copula_min_eigenvalue':np.nan,'n_training_style':len(local),
                    },metrics))
                else:
                    rows.append({
                        'evaluation_scope':'outer_training_real_real_reference','diagnostic_scope':None,
                        'repeat':repeat,'fold':fold,'style':style,
                        'method':'real_real_reference','seed_index':seed_index,
                        'status':'NA_insufficient_disjoint_matched_training_rows','n_real':len(real),'n_training':len(local),
                    })
        table=pd.DataFrame(rows);metrics_path=slot/'metrics.csv';generated_path=slot/'generated.csv.gz'
        table.to_csv(metrics_path,index=False)
        pd.concat(generated,ignore_index=True).to_csv(generated_path,index=False,compression='gzip')
        record={
            'signature':signature,'train_beer':chemistry.beer.iloc[train].astype(int).tolist(),
            'validation_beer':chemistry.beer.iloc[valid].astype(int).tolist(),
            'files':{path.name:sha(path) for path in [metrics_path,generated_path]},
        }
        temporary=slot/'manifest.json.tmp';temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');temporary.replace(commit)
        tables.append(table);generated_files.append(generated_path)
        progress={
            'experiment_id':'E202','status':'running','completed_folds':number,'n_outer_folds':30,
            'last_split_id':split_id,'elapsed_seconds':time.perf_counter()-start,'entry_gate':entry_gate,
        }
        temporary=RESULT/'q3_progress_E202.json.tmp';temporary.write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n');temporary.replace(RESULT/'q3_progress_E202.json')
        print(f'[{number:02d}/30] {split_id} complete',flush=True)

    metrics=pd.concat(tables,ignore_index=True);metrics_path=RESULT/'q3_distribution_metrics_E202.csv'
    metrics.to_csv(metrics_path,index=False)
    names=['mean_w1','energy_v','coverage','near_duplicate_fraction','spearman_pair_rmse','pre_clip_cell_fraction','pre_clip_row_fraction']
    repeat_table=metrics.groupby(['method','style','repeat'],as_index=False)[names].mean()
    repeat_path=RESULT/'q3_metrics_by_repeat_E202.csv';repeat_table.to_csv(repeat_path,index=False)
    summary=repeat_table.groupby(['method','style'])[names].agg(['mean','std'])
    summary.columns=['_'.join(column) for column in summary.columns]
    summary_path=RESULT/'q3_baseline_summary_E202.csv';summary.reset_index().to_csv(summary_path,index=False)
    decision={
        'status':'statistical_baselines_complete_no_migration','formal_methods':config['methods'],
        'sensory_profiles_generated':False,'chemical_safety_certified':False,'migration_performed':False,
        'q1_retrained':False,'gpu_used':False,
        'notes':['E202只评估折外化学分布拟合，不以单项W1宣布最终模型','五个随机种子不是新增实测样本','支持域截断率仅为模型诊断，不等于缺陷安全率'],
    }
    decision_path=RESULT/'q3_baseline_decision_E202.json';decision_path.write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    outputs=[metrics_path,repeat_path,summary_path,decision_path]
    manifest={
        'experiment_id':'E202','status':'complete','n_outer_folds':30,'target_styles':config['target_styles'],
        'methods':config['methods'],'signature':signature,'gpu_used':False,'python':platform.python_version(),
        'elapsed_seconds':time.perf_counter()-start,
        'aggregation':'first average folds/seeds within repeat/style, then mean and SD across repeats; repeats and seeds are not independent samples',
        'decision':decision,'checkpoint_files':[str(path) for path in generated_files],
        'output_sha256':{path.name:sha(path) for path in outputs},
    }
    manifest_path=RESULT/'q3_run_manifest_E202.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    progress=json.loads((RESULT/'q3_progress_E202.json').read_text());progress.update(status='complete',completed_folds=30)
    (RESULT/'q3_progress_E202.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2));return manifest

if __name__=='__main__':run()
