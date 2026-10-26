"""Empirical-marginal Gaussian Copula with training-selected rank correlation shrinkage."""
import numpy as np
from scipy.stats import rankdata,norm


class LatentCorrelationGenerator:
    def fit(self,x,styles,*,target_styles):
        values=np.asarray(x,float);labels=np.asarray(styles,str)
        if values.ndim!=2 or labels.shape!=(len(values),) or not np.isfinite(values).all():
            raise ValueError('Finite aligned training matrix required')
        self.models={}
        for style in target_styles:
            block=values[labels==style]
            if len(block)<2:raise ValueError('At least two training rows for each requested style')
            latent=norm.ppf((rankdata(block,axis=0,method='average')-.5)/len(block))
            latent-=latent.mean(axis=0)
            std=latent.std(axis=0);active=(std>1e-12)&(np.ptp(block,axis=0)>1e-12)
            h=np.zeros_like(latent);h[:,active]=latent[:,active]/std[active]
            correlation=h.T@h/len(block);np.fill_diagonal(correlation,1.)
            correlation=(correlation+correlation.T)/2.
            eigenvalues,eigenvectors=np.linalg.eigh(correlation)
            self.models[style]={'sorted':np.sort(block,axis=0),'correlation':correlation,
                'eigenvalues':np.maximum(eigenvalues,0.),'eigenvectors':eigenvectors}
        return self

    def correlation(self,style,shrinkage):
        if style not in self.models or not np.isfinite(shrinkage) or not 0<=shrinkage<=1:
            raise ValueError('Fitted style and correlation shrinkage in [0,1] required')
        matrix=self.models[style]['correlation']
        return (1-shrinkage)*matrix+shrinkage*np.eye(len(matrix))

    def generate(self,style,n,*,shrinkage,seed):
        self.correlation(style,shrinkage)
        if not isinstance(n,(int,np.integer)) or n<1:raise ValueError('Positive generation size required')
        model=self.models[style];p=len(model['eigenvalues'])
        factor=model['eigenvectors']*np.sqrt((1-shrinkage)*model['eigenvalues']+shrinkage)
        latent=np.random.default_rng(seed).normal(size=(n,p))@factor.T
        uniform=norm.cdf(latent)
        return np.column_stack([np.quantile(model['sorted'][:,j],uniform[:,j],method='linear') for j in range(p)])
