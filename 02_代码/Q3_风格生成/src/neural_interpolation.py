"""Small shared denoising representation with anchored curvature and residual noise."""
import time

import numpy as np
import torch
from torch import nn
from sklearn.covariance import LedoitWolf

from empirical_generation import StyleEmpiricalGenerator
from gaussian_baselines import StyleGaussianGenerator,_psd_factor


def curvature_delta(a,b,t,decode):
    a=np.asarray(a,float);b=np.asarray(b,float);t=np.asarray(t,float)
    if a.shape!=b.shape or a.ndim!=2 or t.shape!=(len(a),) or not np.isfinite([*a.ravel(),*b.ravel(),*t]).all() or (t<0).any() or (t>1).any():
        raise ValueError('Aligned finite latent endpoints and coefficients in [0,1] required')
    return decode((1-t[:,None])*a+t[:,None]*b)-(1-t[:,None])*decode(a)-t[:,None]*decode(b)


class SmallAutoencoder(nn.Module):
    def __init__(self,p,hidden,latent):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(p,hidden),nn.Tanh(),nn.Linear(hidden,latent),nn.Tanh())
        self.decoder=nn.Sequential(nn.Linear(latent,hidden),nn.Tanh(),nn.Linear(hidden,p))

    def forward(self,x):return self.decoder(self.encoder(x))


class ResidualInterpolator:
    def _setup(self,x,styles,config,target_styles):
        values=np.asarray(x,float);labels=np.asarray(styles,str)
        if values.ndim!=2 or labels.shape!=(len(values),) or len(values)<2 or not np.isfinite(values).all():
            raise ValueError('Finite current-training chemical rows required')
        self.values_=values.copy();self.styles_=labels.copy();self.config_=dict(config);self.targets_=list(target_styles)
        self.lower_,self.upper_=np.quantile(values,config['winsor_quantiles'],axis=0)
        clipped=np.clip(values,self.lower_,self.upper_);self.center_=np.median(clipped,axis=0)
        iqr=np.subtract(*np.quantile(clipped,[.75,.25],axis=0));std=clipped.std(axis=0)
        scale=np.where(iqr>1e-12,iqr,std);self.scale_=np.where(scale>1e-12,scale,1.)
        self.u_=self.transform(values);self.pca_mean_=self.u_.mean(axis=0)
        _,_,vt=np.linalg.svd(self.u_-self.pca_mean_,full_matrices=False)
        self.pca_components_=vt[:min(config['latent_dim'],len(values)-1,values.shape[1])].copy()
        raw_scale=values.std(axis=0);raw_scale=np.where(raw_scale>1e-12,raw_scale,1.)
        self.local_=StyleEmpiricalGenerator().fit(values,labels,scale=raw_scale)
        self.positions_={s:np.flatnonzero(labels==s) for s in target_styles}
        if any(len(v)<2 for v in self.positions_.values()):raise ValueError('Each requested style requires at least two training rows')
        self.copulas_={s:StyleGaussianGenerator().fit(values[ids],np.repeat(s,len(ids))) for s,ids in self.positions_.items()}
        return self

    def transform(self,x):
        values=np.asarray(x,float)
        if values.ndim!=2 or values.shape[1]!=self.values_.shape[1] or not np.isfinite(values).all():
            raise ValueError('Finite full-field chemical matrix required')
        return (np.clip(values,self.lower_,self.upper_)-self.center_)/self.scale_

    def fit(self,x,styles,config,*,target_styles,train_seed,device):
        self._setup(x,styles,config,target_styles);started=time.perf_counter()
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Registered CUDA device unavailable')
        devices=[torch.cuda.current_device()] if device=='cuda' else []
        previous=torch.are_deterministic_algorithms_enabled()
        self.history_=[]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(train_seed));torch.use_deterministic_algorithms(True)
            try:
                network=SmallAutoencoder(self.values_.shape[1],config['hidden_dim'],config['latent_dim']).to(device=device,dtype=torch.float32)
                optimizer=torch.optim.AdamW(network.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
                tensor=torch.as_tensor(self.u_,dtype=torch.float32,device=device)
                generator=torch.Generator(device=device).manual_seed(int(train_seed)+1)
                network.train()
                for step in range(config['train_steps']):
                    noise=torch.randn(tensor.shape,generator=generator,device=device,dtype=tensor.dtype)*config['denoising_std']
                    optimizer.zero_grad(set_to_none=True);prediction=network(tensor+noise)
                    loss=nn.functional.smooth_l1_loss(prediction,tensor)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite denoising training loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(network.parameters(),config['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();self.history_.append({'step':step+1,'training_huber':float(loss.detach().cpu()),'gradient_norm':float(norm.detach().cpu())})
                self.network_=network.to(device='cpu',dtype=torch.float64).eval()
            finally:torch.use_deterministic_algorithms(previous)
        self.train_seed_=int(train_seed);self.gpu_used_=device=='cuda';self.training_seconds_=time.perf_counter()-started
        self.n_parameters_=sum(p.numel() for p in self.network_.parameters())
        self._residual_factors();return self

    def _decode(self,z):
        with torch.inference_mode():return self.network_.decoder(torch.as_tensor(np.asarray(z,float),dtype=torch.float64)).numpy()

    def _encode(self,u):
        with torch.inference_mode():return self.network_.encoder(torch.as_tensor(np.asarray(u,float),dtype=torch.float64)).numpy()

    def reconstruct_u(self,u,family):
        if family=='pca':return self.pca_mean_+(u-self.pca_mean_)@self.pca_components_.T@self.pca_components_
        if family=='neural':return self._decode(self._encode(u))
        raise ValueError('Unknown residual representation')

    def reconstruct(self,x,family):return self.reconstruct_u(self.transform(x),family)*self.scale_+self.center_

    def _residual_factors(self):
        self.factors_={};self.residual_diagnostics_=[]
        for family in ['pca','neural']:
            residual=self.u_-self.reconstruct_u(self.u_,family);within=residual.copy()
            for style in np.unique(self.styles_):within[self.styles_==style]-=residual[self.styles_==style].mean(axis=0)
            global_cov=LedoitWolf().fit(within).covariance_;self.factors_[family]={}
            for style,ids in self.positions_.items():
                local_cov=LedoitWolf().fit(residual[ids]).covariance_
                weight=len(ids)/(len(ids)+self.config_['covariance_pseudocount'])
                cov=(1-weight)*global_cov+weight*local_cov;_,factor=_psd_factor(cov)
                self.factors_[family][style]=factor
                self.residual_diagnostics_.append({'family':family,'style':style,'style_covariance_weight':weight,
                    'residual_trace':float(np.trace(cov)),'n_training_style':len(ids)})

    def local(self,style,n,family,*,seed,noise_seed,beta,gamma):
        if family not in ['raw','pca','neural'] or min(beta,gamma)<0 or not np.isfinite([beta,gamma]).all():
            raise ValueError('Known representation and nonnegative finite correction strengths required')
        base=self.local_.generate(style,n,method='local_interpolation',seed=seed,neighbors=self.config_['neighbors'])
        linear=base['values'];a=base['anchor_a'];b=base['anchor_b'];t=base['mixture']
        if family=='raw':
            if beta!=0 or gamma!=0:raise ValueError('Raw baseline cannot include hidden corrections')
            raw=linear.copy();values=raw.copy()
        else:
            correction=np.zeros_like(linear)
            if family=='neural' and beta:
                correction=curvature_delta(self._encode(self.u_[a]),self._encode(self.u_[b]),t,self._decode)
            elif family=='pca' and beta:raise ValueError('PCA affine curvature is identically zero')
            noise=np.random.default_rng(noise_seed).normal(size=linear.shape)@self.factors_[family][style].T
            raw=linear+beta*correction*self.scale_+gamma*np.sqrt(2*t*(1-t))[:,None]*noise*self.scale_
            block=self.values_[self.positions_[style]];low=block.min(axis=0);high=block.max(axis=0);constant=high-low<=1e-12
            raw[:,constant]=low[constant];values=np.clip(raw,low,high)
        block=self.values_[self.positions_[style]];low=block.min(axis=0);high=block.max(axis=0)
        outside=(raw<low-1e-12)|(raw>high+1e-12)
        return {'values':values,'unclipped_values':raw,'anchor_a':a,'anchor_b':b,'mixture':t,
            'preclip_cell_fraction':outside.mean(axis=1),'preclip_row':outside.any(axis=1)}

    def generate(self,style,n,family,*,alpha,beta,gamma,seed):
        if not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('Mixture probability in [0,1] required')
        seeds=tuple(int(s.generate_state(1,dtype=np.uint32)[0]) for s in np.random.SeedSequence(seed).spawn(4))
        if len(set(seeds))!=4:raise ValueError('Independent stream seed collision')
        local=self.local(style,n,family,seed=seeds[2],noise_seed=seeds[3],beta=beta,gamma=gamma)
        copula=self.copulas_[style].generate(style,n,method='style_copula_shrunk',seed=seeds[1])['values']
        selected=np.random.default_rng(seeds[0]).random(n)<alpha
        return {'values':np.where(selected[:,None],local['values'],copula),
            'unclipped_values':np.where(selected[:,None],local['unclipped_values'],copula),
            'local_selected':selected,'anchor_a':local['anchor_a'],'anchor_b':local['anchor_b'],'interpolation_t':local['mixture'],
            'local_preclip_cell_fraction':local['preclip_cell_fraction'],'local_preclip_row':local['preclip_row'],
            'stream_seeds':seeds}

    def state_arrays(self):
        state={f'weight__{k}':v.detach().cpu().numpy().copy() for k,v in self.network_.state_dict().items()}
        state.update(lower=self.lower_,upper=self.upper_,center=self.center_,scale=self.scale_,
            pca_mean=self.pca_mean_,pca_components=self.pca_components_)
        return state

    @classmethod
    def from_arrays(cls,x,styles,config,*,target_styles,arrays):
        result=cls()._setup(x,styles,config,target_styles)
        for name in ['lower','upper','center','scale','pca_mean','pca_components']:
            np.testing.assert_allclose(getattr(result,name+'_'),arrays[name],atol=1e-12,rtol=0.)
        with torch.random.fork_rng(devices=[]):
            result.network_=SmallAutoencoder(result.values_.shape[1],config['hidden_dim'],config['latent_dim']).double().eval()
        weights={k.removeprefix('weight__'):torch.as_tensor(np.asarray(v).copy()) for k,v in arrays.items() if k.startswith('weight__')}
        result.network_.load_state_dict(weights,strict=True);result.history_=[];result.gpu_used_=False
        result.n_parameters_=sum(p.numel() for p in result.network_.parameters());result._residual_factors();return result
