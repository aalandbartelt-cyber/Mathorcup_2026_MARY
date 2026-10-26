"""Empirical beta dependence and whole-row local/Copula mixture candidates."""
import numpy as np
from scipy.spatial.distance import cdist,pdist

from gaussian_baselines import StyleGaussianGenerator
from empirical_generation import StyleEmpiricalGenerator
from q3_evaluation import distribution_diagnostics

CANDIDATES=['copula','local','beta_linear','beta_step','mix_25','mix_50','mix_75']


def balanced_inner_splits(styles,*,seed):
    labels=np.asarray(styles,str);rng=np.random.default_rng(seed);parts=[[],[]]
    for style in np.unique(labels):
        ids=rng.permutation(np.flatnonzero(labels==style))
        for i,block in enumerate(np.array_split(ids,2)):parts[i].extend(block.tolist())
    result=[];all_ids=np.arange(len(labels))
    for part in parts:
        valid=np.sort(part);train=np.setdiff1d(all_ids,valid)
        result.append((train,valid))
    return result


def diversity_diagnostics(real,generated,training,scale):
    base=distribution_diagnostics(real,generated,training,scale)
    real=np.asarray(real,float)/scale;generated=np.asarray(generated,float)/scale
    spread_real=float(pdist(real).mean());spread_generated=float(pdist(generated).mean())
    ratio=spread_generated/spread_real if spread_real>1e-12 else np.nan
    base['pairwise_spread_ratio']=ratio
    base['log_spread_error']=float(abs(np.log(ratio))) if np.isfinite(ratio) and ratio>0 else np.inf
    base['collapsed']=bool(spread_generated<=1e-12)
    rr=cdist(real,real);np.fill_diagonal(rr,np.inf);radii=rr.min(axis=1)
    rg=cdist(real,generated)
    base['density_k1']=float((rg<=radii[:,None]).sum(axis=0).mean())
    return base


class JointStyleGenerator:
    def fit(self,x,styles,*,scale,tie_seed=20261026,target_styles=None):
        values=np.asarray(x,float);labels=np.asarray(styles,str);scale=np.asarray(scale,float)
        if (values.ndim!=2 or labels.shape!=(len(values),) or not np.isfinite(values).all()
            or scale.shape!=(values.shape[1],) or not np.isfinite(scale).all() or (scale<=0).any()):
            raise ValueError('Finite training rows, aligned styles and positive training scale required')
        targets=list(np.unique(labels)) if target_styles is None else list(target_styles)
        self.blocks_={};self.ranks_={};self.copulas_={};rng=np.random.default_rng(tie_seed)
        self.local_=StyleEmpiricalGenerator().fit(values,labels,scale=scale)
        for style in targets:
            block=values[labels==style]
            if len(block)<2:
                if target_styles is not None:raise ValueError('Requested style has fewer than two training observations')
                continue
            ranks=np.empty(block.shape,int)
            for column in range(block.shape[1]):
                # Each tied group receives a random permutation of its ordinal
                # positions; this does not fabricate concentrations or an LOD.
                order=np.lexsort((rng.random(len(block)),block[:,column]))
                ranks[order,column]=np.arange(1,len(block)+1)
            self.blocks_[style]=block.copy();self.ranks_[style]=ranks
            self.copulas_[style]=StyleGaussianGenerator().fit(block,np.repeat(style,len(block)))
        return self

    def generate(self,style,n,method,*,seed):
        if style not in self.blocks_ or not isinstance(n,(int,np.integer)) or n<1:
            raise ValueError('Fitted style and positive sample count required')
        if method not in CANDIDATES:raise ValueError('Unregistered generation candidate')
        if method=='copula':
            return self.copulas_[style].generate(style,n,method='style_copula_shrunk',seed=seed)['values']
        if method=='local':return self.local_.generate(style,n,method='local_interpolation',seed=seed)['values']
        rng=np.random.default_rng(seed)
        if method.startswith('mix_'):
            probability=int(method.split('_')[1])/100
            copula=self.generate(style,n,'copula',seed=seed)
            local=self.generate(style,n,'local',seed=seed)
            mask=rng.random(n)<probability
            return np.where(mask[:,None],local,copula)
        block=self.blocks_[style];n_train=len(block)
        anchor=rng.integers(n_train,size=n);ranks=self.ranks_[style][anchor]
        uniform=rng.beta(ranks,n_train+1-ranks)
        sorted_values=np.sort(block,axis=0)
        if method=='beta_step':
            indices=np.minimum((uniform*n_train).astype(int),n_train-1)
            return sorted_values[indices,np.arange(block.shape[1])[None,:]]
        return np.column_stack([np.quantile(sorted_values[:,j],uniform[:,j],method='linear')
                                for j in range(block.shape[1])])
