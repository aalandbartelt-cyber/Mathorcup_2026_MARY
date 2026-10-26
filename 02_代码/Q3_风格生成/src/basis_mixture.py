"""Training-only chemical geometry and independent-stream whole-spectrum mixtures."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist,pdist
from sklearn.cluster import KMeans

from gaussian_baselines import StyleGaussianGenerator


def mixture_seeds(seed):
    return tuple(int(s.generate_state(1,dtype=np.uint32)[0]) for s in np.random.SeedSequence(seed).spawn(3))


def neighbor_jaccard(a,b):
    return np.array([len(set(x)&set(y))/len(set(x)|set(y)) for x,y in zip(a,b)],float)


class ChemicalBasisGeometry:
    def fit(self,x,geometry_indices,config,*,coordinate_dim=None,scale_kind='iqr'):
        values=np.asarray(x,float);indices=np.asarray(geometry_indices,int)
        if (values.ndim!=2 or len(values)<2 or not np.isfinite(values).all() or indices.ndim!=1
            or not len(indices) or len(np.unique(indices))!=len(indices) or indices.min()<0 or indices.max()>=values.shape[1]):
            raise ValueError('Finite training matrix and unique valid geometry fields required')
        if scale_kind not in ['iqr','std']:raise ValueError('Unsupported chemical geometry scale')
        self.n_features_=values.shape[1];self.indices_=indices.copy();self.config_=dict(config)
        block=values[:,indices];self.lower_,self.upper_=np.quantile(block,config['winsor_quantiles'],axis=0)
        clipped=np.clip(block,self.lower_,self.upper_);self.center_=np.median(clipped,axis=0)
        std=clipped.std(axis=0);iqr=np.subtract(*np.quantile(clipped,[.75,.25],axis=0))
        scale=std if scale_kind=='std' else np.where(iqr>1e-12,iqr,std)
        self.scale_=np.where(scale>1e-12,scale,1.)
        standardized=(clipped-self.center_)/self.scale_;self.pca_mean_=standardized.mean(axis=0)
        _,_,vt=np.linalg.svd(standardized-self.pca_mean_,full_matrices=False)
        dim=int(config['coordinate_dim'] if coordinate_dim is None else coordinate_dim)
        if dim<1:raise ValueError('Positive coordinate dimension required')
        self.components_=vt[:min(dim,len(values)-1,len(indices))].copy()
        coords=self.transform(values)
        unique=np.unique(np.round(coords,12),axis=0);n=min(int(config['prototype_count']),len(unique))
        if n<1:raise ValueError('At least one chemical prototype required')
        if n==len(unique):self.prototypes_=unique.copy()
        else:self.prototypes_=KMeans(n_clusters=n,random_state=int(config['prototype_seed']),n_init=5).fit(coords).cluster_centers_
        distances=pdist(self.prototypes_);positive=distances[distances>1e-12]
        base=float(np.median(positive)) if len(positive) else 1.
        self.bandwidths_=base*np.asarray(config['bandwidth_multipliers'],float)
        return self

    def transform(self,x):
        values=np.asarray(x,float)
        if values.ndim!=2 or values.shape[1]!=self.n_features_ or not np.isfinite(values).all():
            raise ValueError('Finite full-field query matrix required')
        z=(np.clip(values[:,self.indices_],self.lower_,self.upper_)-self.center_)/self.scale_-self.pca_mean_
        return z@self.components_.T

    def basis(self,z):
        coords=np.asarray(z,float)
        if coords.ndim!=2 or coords.shape[1]!=self.prototypes_.shape[1] or not np.isfinite(coords).all():
            raise ValueError('Finite fitted-dimension coordinates required')
        squared=cdist(coords,self.prototypes_,metric='sqeuclidean')
        return np.concatenate([np.exp(-np.minimum(squared/(2*h*h),745.)) for h in self.bandwidths_],axis=1)

    def neighborhood(self,x,*,mode):
        values=np.asarray(x,float);n=len(values)
        if n<2 or mode not in ['pca','basis']:raise ValueError('At least two same-style points and known geometry required')
        coords=self.transform(values);dist=cdist(coords,coords);np.fill_diagonal(dist,np.inf)
        order=np.argsort(dist,axis=1,kind='stable');k=min(int(self.config_['neighbors']),n-1)
        support=min(int(self.config_['pca_support_neighbors']),n-1)
        neighbors=order[:,:k].copy();fallback=np.full(n,mode=='pca');effective=np.zeros(n,bool)
        if mode=='basis':
            phi=self.basis(coords);bd=cdist(phi,phi)
            for i in range(n):
                allowed=order[i,:support]
                invalid=(phi[i].max()<self.config_['basis_zero_tolerance']
                    or np.all(bd[i,allowed]<1e-12) or np.ptp(dist[i,allowed])<1e-12)
                fallback[i]=invalid
                if not invalid:
                    chosen=np.lexsort((allowed,bd[i,allowed]))[:k];neighbors[i]=allowed[chosen]
                    effective[i]=support>k
        diagnostic=pd.DataFrame({'anchor_local_position':np.arange(n),'fallback':fallback,'basis_effective':effective})
        return neighbors,diagnostic


class ChemicalMixtureGenerator:
    def fit(self,x,styles,geometry_indices,config,*,target_styles):
        values=np.asarray(x,float);labels=np.asarray(styles,str)
        if values.ndim!=2 or labels.shape!=(len(values),) or not np.isfinite(values).all():
            raise ValueError('Finite current-training rows with style labels required')
        self.values_=values.copy();self.styles_=labels.copy();self.config_=dict(config)
        self.scale_=np.std(values,axis=0);self.scale_=np.where(self.scale_>1e-12,self.scale_,1.)
        self.geometry_=ChemicalBasisGeometry().fit(values,geometry_indices,config)
        self.neighbors_={};self.copulas_={};self.positions_={};self.diagnostics_={};self.unstable_={}
        original_basis={};stability=[]
        for style in target_styles:
            ids=np.flatnonzero(labels==style)
            if len(ids)<2:raise ValueError('Requested style requires at least two training rows')
            block=values[ids];k=min(int(config['neighbors']),len(ids)-1)
            raw=cdist(block/self.scale_,block/self.scale_);np.fill_diagonal(raw,np.inf)
            rn=np.argsort(raw,axis=1,kind='stable')[:,:k]
            pn,_=self.geometry_.neighborhood(block,mode='pca');bn,diagnostic=self.geometry_.neighborhood(block,mode='basis')
            original_basis[style]=bn;self.positions_[style]=ids
            self.neighbors_[style]={'raw':ids[rn],'pca':ids[pn],'basis':ids[bn]}
            self.diagnostics_[style]=diagnostic
            self.copulas_[style]=StyleGaussianGenerator().fit(block,np.repeat(style,len(block)))
        rng=np.random.default_rng(config['prototype_seed'])
        perturbations=[]
        for b in range(config['stability_bootstraps']):
            draw=rng.integers(len(values),size=len(values));perturbations.append((f'bootstrap_{b}',values[draw],None,'iqr'))
        for dim in config['stability_dimensions']:perturbations.append((f'dimension_{dim}',values,dim,'iqr'))
        perturbations.append(('std_scale',values,None,'std'))
        for name,training,dim,scale in perturbations:
            perturbed=ChemicalBasisGeometry().fit(training,geometry_indices,config,coordinate_dim=dim,scale_kind=scale)
            for style,ids in self.positions_.items():
                other,_=perturbed.neighborhood(values[ids],mode='basis');scores=neighbor_jaccard(original_basis[style],other)
                stability.append({'style':style,'perturbation':name,'mean_jaccard':float(scores.mean()),
                    'median_jaccard':float(np.median(scores)),'minimum_jaccard':float(scores.min()),'n_anchors':len(ids)})
        self.stability_table_=pd.DataFrame(stability)
        for style in target_styles:
            median=float(self.stability_table_.loc[self.stability_table_['style']==style,'median_jaccard'].median())
            unstable=median<config['stability_minimum_median_jaccard'];self.unstable_[style]=unstable
            if unstable:
                self.neighbors_[style]['basis']=self.neighbors_[style]['pca'].copy()
                self.diagnostics_[style]['fallback']=True;self.diagnostics_[style]['basis_effective']=False
            self.diagnostics_[style]['stability_median_jaccard']=median
            self.diagnostics_[style]['style_stability_fallback']=unstable
        return self

    def local(self,style,n,geometry,*,seed):
        if style not in self.positions_ or geometry not in ['raw','pca','basis'] or not isinstance(n,(int,np.integer)) or n<1:
            raise ValueError('Known style/geometry and positive sample count required')
        rng=np.random.default_rng(seed);positions=self.positions_[style]
        a=rng.choice(positions,size=n,replace=True);index={int(value):i for i,value in enumerate(positions)}
        b=np.array([rng.choice(self.neighbors_[style][geometry][index[int(value)]]) for value in a])
        t=rng.uniform(.2,.8,size=n);sample=(1-t[:,None])*self.values_[a]+t[:,None]*self.values_[b]
        return {'values':sample,'anchor_a':a,'anchor_b':b,'mixture':t}

    def generate(self,style,n,geometry,*,alpha,seed):
        if not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('Mixture probability in [0,1] required')
        selector,copula_seed,local_seed=mixture_seeds(seed)
        local=self.local(style,n,geometry,seed=local_seed)
        copula=self.copulas_[style].generate(style,n,method='style_copula_shrunk',seed=copula_seed)['values']
        chosen=np.random.default_rng(selector).random(n)<alpha
        return {'values':np.where(chosen[:,None],local['values'],copula),'local_selected':chosen,
            'anchor_a':local['anchor_a'],'anchor_b':local['anchor_b'],'interpolation_t':local['mixture'],
            'selector_seed':selector,'copula_seed':copula_seed,'local_seed':local_seed}
