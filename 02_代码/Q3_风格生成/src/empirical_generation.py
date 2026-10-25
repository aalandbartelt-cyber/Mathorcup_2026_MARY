"""Training-only style baselines with explicit copying and collapse controls."""
import numpy as np
from scipy.spatial.distance import cdist

class StyleEmpiricalGenerator:
    def fit(self,x,styles,*,scale=None):
        values=np.asarray(x,float);labels=np.asarray(styles,str)
        if values.ndim!=2 or len(values)<2 or labels.shape!=(len(values),) or not np.isfinite(values).all():
            raise ValueError('finite training rows and matched style labels required')
        if values.shape[1]<1:raise ValueError('at least one feature required')
        scale=np.std(values,axis=0) if scale is None else np.asarray(scale,float)
        if scale.shape!=(values.shape[1],) or not np.isfinite(scale).all() or (scale<0).any():
            raise ValueError('finite nonnegative training scale required')
        self.scale_=np.where(scale>1e-12,scale,1.)
        self.values_=values.copy();self.styles_=labels.copy()
        return self

    def generate(self,style,n,*,method,seed,neighbors=3):
        if not hasattr(self,'values_'):raise ValueError('fit is required before generation')
        if not isinstance(n,(int,np.integer)) or n<1:raise ValueError('positive integer sample count required')
        positions=np.flatnonzero(self.styles_==str(style))
        if len(positions)<2:raise ValueError('at least two training rows of the requested style required')
        rng=np.random.default_rng(seed);a=rng.choice(positions,size=n,replace=True);b=a.copy()
        mixture=np.zeros(n)
        if method=='empirical_resample':samples=self.values_[a].copy()
        elif method=='style_mean':
            samples=np.tile(self.values_[positions].mean(0),(n,1));a[:]=-1;b[:]=-1
        elif method=='local_interpolation':
            if not isinstance(neighbors,(int,np.integer)) or neighbors<1:raise ValueError('invalid neighbor count')
            distances=cdist(self.values_[positions]/self.scale_,self.values_[positions]/self.scale_)
            np.fill_diagonal(distances,np.inf)
            nearest=np.argsort(distances,axis=1,kind='stable')[:,:min(neighbors,len(positions)-1)]
            local={int(value):i for i,value in enumerate(positions)}
            b=np.asarray([positions[rng.choice(nearest[local[int(value)]])] for value in a])
            mixture=rng.uniform(.2,.8,size=n)
            samples=(1-mixture[:,None])*self.values_[a]+mixture[:,None]*self.values_[b]
        else:raise ValueError('unsupported baseline method')
        return {'values':samples,'anchor_a':a,'anchor_b':b,'mixture':mixture}
