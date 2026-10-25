"""Fold-local full Gaussian and empirical-Copula style baselines for Q3."""
import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.stats import norm

class StyleGaussianGenerator:
    def fit(self,x,styles,*,shrinkage='ledoit_wolf',mean_shrinkage=.25):
        values=np.asarray(x,float);labels=np.asarray(styles,str)
        if values.ndim!=2 or labels.shape!=(len(values),) or len(values)<2 or not np.isfinite(values).all():raise ValueError('finite training matrix and labels required')
        self.global_mean=values.mean(0);self.models={};self.n_features=values.shape[1]
        for style in np.unique(labels):
            block=values[labels==style]
            if len(block)<2:raise ValueError('each style needs at least two training rows')
            estimator=LedoitWolf().fit(block)
            mean=(1-mean_shrinkage)*block.mean(0)+mean_shrinkage*self.global_mean
            self.models[str(style)]={'mean':mean,'cov':estimator.covariance_,'rows':len(block),'style_values':block.copy()}
        if shrinkage!='ledoit_wolf':raise ValueError('only registered Ledoit-Wolf covariance is supported')
        return self

    def generate(self,style,n,*,method='style_gaussian',seed=20261025):
        if str(style) not in self.models:raise ValueError('unseen style')
        if n<1:raise ValueError('positive sample count required')
        model=self.models[str(style)];rng=np.random.default_rng(seed)
        if method=='style_gaussian': values=rng.multivariate_normal(model['mean'],model['cov'],size=n,check_valid='warn')
        elif method=='global_gaussian': values=rng.multivariate_normal(self.global_mean,np.eye(self.n_features),size=n,check_valid='warn')
        elif method=='style_copula':
            block=model['style_values'];z=np.empty_like(block)
            for j in range(self.n_features):
                order=np.argsort(block[:,j],kind='stable');ranks=np.empty(len(block));ranks[order]=np.arange(len(block));z[:,j]=norm.ppf((ranks+.5)/len(block))
            corr=np.corrcoef(z,rowvar=False);corr=np.nan_to_num(corr,nan=0.);corr=(corr+corr.T)/2.;np.fill_diagonal(corr,1.)
            latent=rng.multivariate_normal(np.zeros(self.n_features),corr,size=n,check_valid='warn');values=np.empty_like(latent)
            for j in range(self.n_features):
                sorted_values=np.sort(block[:,j]);u=np.clip(norm.cdf(latent[:,j]),0,1);values[:,j]=np.quantile(sorted_values,u,method='linear')
        else:raise ValueError('unsupported Gaussian baseline')
        return {'values':values,'style':str(style),'method':method,'n_training_style':model['rows']}
