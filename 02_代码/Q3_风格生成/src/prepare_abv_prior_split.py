"""Prepare brewery-disjoint external ABV roles; do not train or evaluate priors."""
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

from kaggle_external import select_mapped
from sensory_audit import sha

ROOT=Path(__file__).resolve().parents[3];CANONICAL=Path('/home/mary/mathorcup');RESULT=ROOT/'03_结果/Q3_风格生成'


def brewery_key(name):
    return ' '.join(unicodedata.normalize('NFKC',str(name)).casefold().split())


def build_split(groups,*,seed):
    rows=[];excluded=[]
    for style,group in groups.items():
        if group['key'].duplicated().any():raise ValueError('Unique external keys required within each semantic style')
        for _,row in group.iterrows():
            key=str(row['key']);abv=pd.to_numeric(row.ABV,errors='coerce')
            valid_abv=bool(np.isfinite(abv) and 0<abv<=20)
            valid_brewery=bool(pd.notna(row.Brewery) and brewery_key(row.Brewery))
            if not valid_abv or not valid_brewery:
                excluded.append({'official_style':style,'external_key':key,'reason':'invalid_abv' if not valid_abv else 'missing_brewery'})
                continue
            name=brewery_key(row.Brewery);digest=hashlib.sha256(name.encode()).hexdigest()
            draw=int(hashlib.sha256(f'{int(seed)}|{name}'.encode()).hexdigest(),16)/2**256
            rows.append({'official_style':style,'external_key':key,'brewery_sha256':digest,
                'split':'prior' if draw<.75 else 'holdout','scope':'posthoc_brewery_disjoint_future_training_role_only'})
    columns=['official_style','external_key','brewery_sha256','split','scope']
    result=pd.DataFrame(rows,columns=columns).sort_values(['official_style','external_key']).reset_index(drop=True)
    if len(result) and result.groupby('brewery_sha256').split.nunique().max()>1:raise ValueError('Brewery crosses prior/holdout')
    return result,pd.DataFrame(excluded,columns=['official_style','external_key','reason'])


def run():
    mp=RESULT/'q3_abv_prior_split_manifest.json'
    if mp.exists():
        meta=json.loads(mp.read_text())
        for p,h in meta['input_sha256'].items():
            if sha(p)!=h:raise ValueError('Frozen external split source changed')
        for name,h in meta['output_sha256'].items():
            if sha(RESULT/name)!=h:raise ValueError('Frozen external split output changed')
        print('Existing ABV split verified; no fitting or holdout evaluation');return meta
    external=CANONICAL/'data/2026初赛/公开数据/Kaggle_top_beer_information'
    raw_path=external/'beer_data_set.csv';mapping_path=external/'processed/kaggle_official_style_mapping_draft.csv'
    mapping=pd.read_csv(mapping_path).fillna({'subset':''});raw=pd.read_csv(raw_path)
    if raw.shape!=(5558,21) or raw['key'].duplicated().any():raise ValueError('External source contract changed')
    groups=select_mapped(raw,mapping,['Tripel','Kriek','Lambic']);split,excluded=build_split(groups,seed=20261026)
    counts=split.groupby(['official_style','split']).agg(n_rows=('external_key','size'),n_breweries=('brewery_sha256','nunique')).reset_index()
    if len(counts)!=6 or counts.n_breweries.min()<2:raise ValueError('Fixed split too sparse; report instead of silently reseeding')
    outputs=[]
    for name,frame in [('q3_abv_prior_split.csv',split),('q3_abv_prior_split_exclusions.csv',excluded),('q3_abv_prior_split_counts.csv',counts)]:
        path=RESULT/name;frame.to_csv(path,index=False);outputs.append(path)
    inputs=[Path(__file__),Path(__file__).with_name('kaggle_external.py'),raw_path,mapping_path,RESULT/'q3_run_manifest_E212.json']
    meta={'status':'split_prepared_only','seed':20261026,'prior_probability':.75,
        'group_assignment':'sha256(seed|NFKC_casefold_whitespace_normalized_brewery)','n_rows':len(split),
        'n_excluded':len(excluded),'brewery_cross_split_count':0,'counts':counts.to_dict('records'),
        'input_sha256':{str(p.resolve()):sha(p) for p in inputs},'output_sha256':{p.name:sha(p) for p in outputs},
        'external_whole_data_previously_described':True,'independent_unseen_confirmation':False,
        'abv_values_in_split_artifact':False,'holdout_used_for_fitting_or_selection':False,'prior_model_trained':False,
        'q1_retrained':False,'q2_modified':False,'q3_generator_replaced':False,
        'notes':['静态外部数据无动作转移；本划分不提供强化学习环境',
            '先前已查看全体外部描述，不可称完全未见测试；这里只隔离未来拟合与评估角色',
            'ABV单维先验不需要评论风味计数；保留ABV有效而评论风味总数为零的记录',
            '不因各风格比例不齐或结果好坏改变固定酒厂划分；尚未拟合或查看holdout模型表现']}
    mp.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');print(json.dumps(meta,ensure_ascii=False,indent=2));return meta


if __name__=='__main__':run()
