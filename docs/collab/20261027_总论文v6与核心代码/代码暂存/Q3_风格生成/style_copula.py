"""Style-conditioned empirical margins and shrinkage Gaussian Copula.

Exact computational functions retained from the archived implementation.
No input reading, model cache loading or top-level fitting occurs.
"""
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.covariance import LedoitWolf


def _psd_factor(matrix):
    """Return a symmetric PSD matrix and a cached sampling factor."""
    symmetric=(np.asarray(matrix,float)+np.asarray(matrix,float).T)/2.
    eigenvalues,eigenvectors=np.linalg.eigh(symmetric)
    eigenvalues=np.clip(eigenvalues,0.,None)
    psd=(eigenvectors*eigenvalues)@eigenvectors.T
    psd=(psd+psd.T)/2.
    return psd,eigenvectors*np.sqrt(eigenvalues)


def _correlation_from_covariance(covariance):
    covariance=np.asarray(covariance,float)
    scale=np.sqrt(np.clip(np.diag(covariance),1e-12,None))
    correlation=covariance/np.outer(scale,scale)
    correlation=np.nan_to_num(correlation,nan=0.,posinf=0.,neginf=0.)
    correlation=(correlation+correlation.T)/2.
    np.fill_diagonal(correlation,1.)
    correlation,_=_psd_factor(correlation)
    scale=np.sqrt(np.clip(np.diag(correlation),1e-12,None))
    correlation=correlation/np.outer(scale,scale)
    correlation=(correlation+correlation.T)/2.
    np.fill_diagonal(correlation,1.)
    return correlation


class StyleGaussianGenerator:
    def fit(self,x,styles,*,shrinkage='ledoit_wolf',mean_shrinkage=.25):
        values=np.asarray(x,float);labels=np.asarray(styles,str)
        if values.ndim!=2 or labels.shape!=(len(values),) or len(values)<2 or not np.isfinite(values).all():
            raise ValueError('finite training matrix and labels required')
        if shrinkage!='ledoit_wolf':
            raise ValueError('only registered Ledoit-Wolf covariance is supported')
        if not 0.<=float(mean_shrinkage)<=1.:
            raise ValueError('mean_shrinkage must be in [0, 1]')
        self.global_mean=values.mean(0)
        self.global_scale=np.std(values,axis=0)
        self.global_scale=np.where(self.global_scale>1e-12,self.global_scale,1.)
        standardized=(values-self.global_mean)/self.global_scale
        self.models={};self.n_features=values.shape[1]
        for style in np.unique(labels):
            block=values[labels==style]
            block_standardized=standardized[labels==style]
            if len(block)<2:
                raise ValueError('each style needs at least two training rows')

            # Raw covariance is retained only to reproduce the immutable pilot.
            raw_estimator=LedoitWolf().fit(block)
            raw_mean=(1-mean_shrinkage)*block.mean(0)+mean_shrinkage*self.global_mean

            estimator=LedoitWolf().fit(block_standardized)
            standardized_mean=(1-mean_shrinkage)*block_standardized.mean(0)
            gaussian_covariance,gaussian_factor=_psd_factor(estimator.covariance_)

            latent=np.zeros_like(block_standardized)
            for column in range(self.n_features):
                if np.ptp(block[:,column])<=1e-12:
                    continue
                ranks=rankdata(block[:,column],method='average')
                latent[:,column]=norm.ppf((ranks-.5)/len(block))
            latent_estimator=LedoitWolf().fit(latent)
            copula_covariance=_correlation_from_covariance(latent_estimator.covariance_)
            copula_covariance,copula_factor=_psd_factor(copula_covariance)

            self.models[str(style)]={
                'mean':raw_mean,
                'cov':raw_estimator.covariance_,
                'standardized_mean':standardized_mean,
                'gaussian_covariance':gaussian_covariance,
                'gaussian_factor':gaussian_factor,
                'copula_covariance':copula_covariance,
                'copula_factor':copula_factor,
                'rows':len(block),
                'style_values':block.copy(),
                'lower':block.min(0),
                'upper':block.max(0),
            }
        return self

    def diagnostics(self,style):
        if str(style) not in self.models:
            raise ValueError('unseen style')
        model=self.models[str(style)]
        return {
            'gaussian_min_eigenvalue':float(np.linalg.eigvalsh(model['gaussian_covariance']).min()),
            'copula_min_eigenvalue':float(np.linalg.eigvalsh(model['copula_covariance']).min()),
            'n_training_style':int(model['rows']),
        }

    def generate(self,style,n,*,method='style_gaussian',seed=20261025):
        if str(style) not in self.models:
            raise ValueError('unseen style')
        if n<1:
            raise ValueError('positive sample count required')
        model=self.models[str(style)];rng=np.random.default_rng(seed)
        pre_clip_cell_fraction=0.;pre_clip_row_fraction=0.
        if method=='style_gaussian':
            values=rng.multivariate_normal(model['mean'],model['cov'],size=n,check_valid='warn')
        elif method=='style_gaussian_clipped':
            latent=rng.normal(size=(n,self.n_features))@model['gaussian_factor'].T
            raw=(latent+model['standardized_mean'])*self.global_scale+self.global_mean
            outside=(raw<model['lower'])|(raw>model['upper'])
            pre_clip_cell_fraction=float(outside.mean())
            pre_clip_row_fraction=float(outside.any(axis=1).mean())
            values=np.clip(raw,model['lower'],model['upper'])
        elif method=='global_gaussian':
            values=rng.multivariate_normal(self.global_mean,np.eye(self.n_features),size=n,check_valid='warn')
        elif method in {'style_copula','style_copula_shrunk'}:
            block=model['style_values']
            if method=='style_copula':
                z=np.empty_like(block)
                for column in range(self.n_features):
                    order=np.argsort(block[:,column],kind='stable')
                    ranks=np.empty(len(block));ranks[order]=np.arange(len(block))
                    z[:,column]=norm.ppf((ranks+.5)/len(block))
                corr=np.corrcoef(z,rowvar=False);corr=np.nan_to_num(corr,nan=0.)
                corr=(corr+corr.T)/2.;np.fill_diagonal(corr,1.)
                latent=rng.multivariate_normal(np.zeros(self.n_features),corr,size=n,check_valid='warn')
            else:
                latent=rng.normal(size=(n,self.n_features))@model['copula_factor'].T
            values=np.empty_like(latent)
            for column in range(self.n_features):
                sorted_values=np.sort(block[:,column])
                probabilities=np.clip(norm.cdf(latent[:,column]),0.,1.)
                values[:,column]=np.quantile(sorted_values,probabilities,method='linear')
        else:
            raise ValueError('unsupported Gaussian baseline')
        return {
            'values':values,
            'style':str(style),
            'method':method,
            'n_training_style':model['rows'],
            'pre_clip_cell_fraction':pre_clip_cell_fraction,
            'pre_clip_row_fraction':pre_clip_row_fraction,
        }
