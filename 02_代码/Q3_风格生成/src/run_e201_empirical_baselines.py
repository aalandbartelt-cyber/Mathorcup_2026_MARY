"""Run fold-safe empirical Q3 baselines after the first verified E113 checkpoint."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,platform,time
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from q3_contract import validate_cv
from q3_evaluation import distribution_diagnostics
from empirical_generation import StyleEmpiricalGenerator

ROOT=Path(__file__).resolve().parents[3]
CANONICAL=Path('/home/mary/mathorcup')
RESULT=ROOT/'03_结果/Q3_风格生成'
CONFIG=Path(__file__).resolve().parents[1]/'configs/E201_empirical_baselines.yaml'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_entry_gate():
    graph=CANONICAL/'.worktrees/q2-evidence-opt/03_结果/Q2_关键化合物'
    progress=json.loads((graph/'q2_progress_E113.json').read_text())
    if progress['completed_folds']<1 or progress['n_features']!=218 or progress['n_outer_folds']!=30:
        raise ValueError('E113 verified first checkpoint not available')
    slot=graph/'q2_graph_checkpoints_E113/R00F0';manifest=json.loads((slot/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(slot/name)!=digest:raise ValueError('E113 checkpoint hash mismatch')
    for name in ['summary','edges','membership']:
        table=pd.read_csv(slot/(name+'.csv.gz'))
        if not table.scope.eq('outer_train_fold').all():raise ValueError('descriptive graph cannot unlock this stage')
    return {'checkpoint':'R00F0','sha256':sha(slot/'manifest.json'),'scope':'outer_train_fold'}

def run():
    start=time.perf_counter();gate=verify_entry_gate()
    config=yaml.safe_load(CONFIG.read_text());RESULT.mkdir(parents=True,exist_ok=True)
    attachment=ROOT/'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv'
    chemistry=pd.read_csv(attachment);split_path=ROOT/'03_结果/Q1_感官评分/q1_cv_splits.csv'
    splits=pd.read_csv(split_path);validate_cv(chemistry,splits,n_repeats=10,n_folds=3)
    features=[x for x in chemistry.columns if x not in ['beer','beer_id','tasting_category_fine']]
    if len(features)!=231 or len(chemistry)!=250:raise ValueError('Q3 input contract changed')
    x=chemistry[features].to_numpy(float);styles=chemistry.tasting_category_fine.to_numpy(str)
    registry_path=CANONICAL/'03_结果/Q1_感官评分/final_model_registry.json'
    registry=json.loads(registry_path.read_text());oof=CANONICAL/registry['oof_path']
    if sha(oof)!=registry['oof_sha256']:raise ValueError('Q1 final OOF hash mismatch')
    signature={'input_sha256':sha(attachment),'cv_sha256':sha(split_path),'config_sha256':sha(CONFIG),
               'source_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('empirical_generation.py'),Path(__file__).with_name('q3_evaluation.py')]},
               'q1_final_oof_sha256':registry['oof_sha256'],'entry_gate':gate}
    slots=RESULT/'q3_checkpoints_E201';slots.mkdir(exist_ok=True)
    tables=[];generated_files=[]
    for number,((repeat,fold),group) in enumerate(splits.groupby(['repeat','fold'],sort=True),1):
        repeat=int(repeat);fold=int(fold);name=f'R{repeat:02d}F{fold}'
        slot=slots/name;slot.mkdir(exist_ok=True);commit=slot/'manifest.json'
        if commit.exists():
            old=json.loads(commit.read_text())
            if old['signature']!=signature:raise ValueError('Q3 checkpoint signature changed')
            for file,digest in old['files'].items():
                if sha(slot/file)!=digest:raise ValueError('Q3 checkpoint damaged')
            tables.append(pd.read_csv(slot/'metrics.csv'));generated_files.append(slot/'generated.csv.gz')
            print(f'[{number:02d}/30] {name} resumed',flush=True);continue
        valid=np.sort(group.row_position.to_numpy(int));train=np.setdiff1d(np.arange(len(x)),valid)
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        tx=imputer.fit_transform(x[train]);vx=imputer.transform(x[valid])
        scale=np.std(tx,axis=0);scale=np.where(scale>1e-12,scale,1.)
        model=StyleEmpiricalGenerator().fit(tx,styles[train],scale=scale)
        rows=[];generated=[]
        for style_index,style in enumerate(config['target_styles']):
            local=np.flatnonzero(styles[train]==style);real=vx[styles[valid]==style]
            if len(real)<2 or len(local)<2:raise ValueError('style fold too small for registered diagnostics')
            for seed_index,seed in enumerate(config['seeds']):
                seed=int(seed)+repeat*1000+fold*100+style_index*10
                for method in config['methods']:
                    sample=model.generate(style,len(real),method=method,seed=seed,neighbors=int(config['neighbors']))
                    metrics=distribution_diagnostics(real,sample['values'],tx[local],scale)
                    metrics.pop('w1_by_feature')
                    rows.append({'scope':'outer_validation_evaluation','repeat':repeat,'fold':fold,'style':style,
                                 'method':method,'seed_index':seed_index,'status':'evaluated',**metrics})
                    table=pd.DataFrame(sample['values'],columns=features)
                    table.insert(0,'sample_position',np.arange(len(real)))
                    for key,value in [('style',style),('seed_index',seed_index),('method',method),('fold',fold),('repeat',repeat),('scope','outer_training_generated')]:table.insert(0,key,value)
                    table['anchor_beer_a']=[int(chemistry.beer.iloc[train[a]]) if a>=0 else None for a in sample['anchor_a']]
                    table['anchor_beer_b']=[int(chemistry.beer.iloc[train[a]]) if a>=0 else None for a in sample['anchor_b']]
                    generated.append(table)
                rng=np.random.default_rng(seed)
                if len(local)>=2*len(real):
                    order=rng.permutation(local)[:2*len(real)]
                    metrics=distribution_diagnostics(tx[order[:len(real)]],tx[order[len(real):]],tx[local],scale)
                    metrics.pop('w1_by_feature')
                    rows.append({'scope':'outer_training_real_real_reference','repeat':repeat,'fold':fold,'style':style,
                                 'method':'real_real_reference','seed_index':seed_index,'status':'evaluated',**metrics})
                else:
                    rows.append({'scope':'outer_training_real_real_reference','repeat':repeat,'fold':fold,'style':style,
                                 'method':'real_real_reference','seed_index':seed_index,'status':'NA_insufficient_disjoint_matched_training_rows',
                                 'n_real':len(real),'n_training':len(local)})
        table=pd.DataFrame(rows);table.to_csv(slot/'metrics.csv',index=False)
        pd.concat(generated,ignore_index=True).to_csv(slot/'generated.csv.gz',index=False,compression='gzip')
        record={'signature':signature,'train_beer':chemistry.beer.iloc[train].astype(int).tolist(),
                'validation_beer':chemistry.beer.iloc[valid].astype(int).tolist(),
                'files':{p.name:sha(p) for p in [slot/'metrics.csv',slot/'generated.csv.gz']}}
        tmp=slot/'manifest.json.tmp';tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');tmp.replace(commit)
        tables.append(table);generated_files.append(slot/'generated.csv.gz')
        progress={'experiment_id':'E201','status':'running','completed_folds':number,'n_outer_folds':30,
                  'last_split_id':name,'elapsed_seconds':time.perf_counter()-start,'entry_gate':gate}
        tmp=RESULT/'q3_progress_E201.json.tmp';tmp.write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n');tmp.replace(RESULT/'q3_progress_E201.json')
        print(f'[{number:02d}/30] {name} complete',flush=True)
    metrics=pd.concat(tables,ignore_index=True);metric_path=RESULT/'q3_distribution_metrics_E201.csv'
    metrics.to_csv(metric_path,index=False)
    names=['mean_w1','energy_v','coverage','near_duplicate_fraction','spearman_pair_rmse']
    by_repeat=metrics.groupby(['method','style','repeat'],as_index=False)[names].mean()
    repeat_path=RESULT/'q3_metrics_by_repeat_E201.csv';by_repeat.to_csv(repeat_path,index=False)
    summary=by_repeat.groupby(['method','style'])[names].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(RESULT/'q3_baseline_summary_E201.csv',index=False)
    decision={'status':'baseline_complete_no_model_selection','sensory_status':'pending_frozen_final_nnls_adapter',
              'chemical_safety_certified':False,'migration_performed':False,
              'q1_retrained':False,'generated_spectra_mechanistically_certified':False,
              'notes':['复制型经验模型与均值负对照均保留，不按单项距离选择赢家','非独立重复：同一啤酒跨重复共享，五种子不作为新增实测样本','真实—真实对照不足同规模不相交样本时记NA']}
    (RESULT/'q3_baseline_decision_E201.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    manifest={'experiment_id':'E201','status':'complete','n_outer_folds':30,'target_styles':config['target_styles'],
              'methods':config['methods'],'signature':signature,'gpu_used':False,'python':platform.python_version(),
              'started_condition':'E113 verified R00F0 checkpoint','elapsed_seconds':time.perf_counter()-start,
              'aggregation':'first average folds/seeds within repeat/style, then mean and SD across repeats; repeats not independent',
              'decision':decision,'checkpoint_files':[str(p) for p in generated_files],
              'output_sha256':{p.name:sha(p) for p in RESULT.glob('*_E201.csv')}}
    (RESULT/'q3_run_manifest_E201.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    progress=json.loads((RESULT/'q3_progress_E201.json').read_text());progress.update(status='complete',completed_folds=30)
    (RESULT/'q3_progress_E201.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2));return manifest

if __name__=='__main__':run()
