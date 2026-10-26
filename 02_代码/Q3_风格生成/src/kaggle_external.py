"""Static external profiles: semantic mapping and shared-brewery cluster uncertainty."""
import numpy as np
import pandas as pd

FLAVORS=['Astringency','Body','Alcohol','Bitter','Sweet','Sour','Salty','Fruits','Hoppy','Spices','Malty']


def normalized_flavors(frame):
    values=frame[FLAVORS].to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Finite nonnegative review descriptor counts required')
    out=frame.copy();total=values.sum(axis=1);usable=total>0
    out['flavor_total']=total;out['flavor_usable']=usable
    for i,name in enumerate(FLAVORS):out['prop_'+name]=np.divide(values[:,i],total,out=np.full(len(total),np.nan),where=usable)
    return out


def select_mapped(frame,mapping,styles):
    if frame['key'].duplicated().any():raise ValueError('Unique external beer keys required')
    selected={};text=(frame.Name.fillna('')+' '+frame.Description.fillna('')).str.lower()
    for style in styles:
        definitions=mapping[mapping.official_style.eq(style)&mapping.primary_pool.eq(1)]
        if not len(definitions):raise ValueError('Missing predefined semantic style mapping')
        mask=np.zeros(len(frame),bool)
        for _,row in definitions.iterrows():
            if str(row.get('subset',''))=='kriek':
                mask|=(frame.Style.eq('Lambic - Fruit')&text.str.contains('kriek|cherry|cerise',regex=True)).to_numpy()
            else:
                if row.kaggle_style not in set(frame.Style):raise ValueError('Mapping references absent raw style')
                mask|=frame.Style.eq(row.kaggle_style).to_numpy()
        selected[style]=frame.loc[mask].copy()
        if not len(selected[style]):raise ValueError('Empty mapped external style')
    return selected


def cluster_profiles(groups,columns,n_bootstrap,*,seed):
    breweries=sorted(set().union(*(set(g.Brewery.astype(str)) for g in groups.values())))
    if not breweries or n_bootstrap<2:raise ValueError('Brewery clusters and repeated resamples required')
    weights=np.random.default_rng(seed).multinomial(len(breweries),np.full(len(breweries),1/len(breweries)),size=n_bootstrap)
    estimates={}
    for style,group in groups.items():
        if group[columns].isna().any().any():raise ValueError('Only usable external profile rows may enter the bootstrap')
        sums=group.groupby('Brewery')[columns].sum().reindex(breweries,fill_value=0.).to_numpy(float)
        counts=group.groupby('Brewery').size().reindex(breweries,fill_value=0).to_numpy(float)
        den=weights@counts;num=weights@sums
        estimates[style]=np.divide(num,den[:,None],out=np.full(num.shape,np.nan),where=den[:,None]>0)
    return estimates


def direction_status(lower,upper,model_delta):
    if not np.isfinite([lower,upper,model_delta]).all():return 'insufficient_evidence'
    if lower<=0<=upper:return 'external_uncertain'
    if abs(model_delta)<=1e-12:return 'model_tie'
    return 'direction_agrees' if np.sign(model_delta)==np.sign(lower+upper) else 'direction_disagrees'
