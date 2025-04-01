import torch
from tqdm import tqdm
from src.models.nfdm.nfdm import t_dir
from src.their_utils.test_util import denoised_fn_round
import torch
from torch import nn
from types import SimpleNamespace
from torch import Tensor
import torch as th
###
# Orchestrator
###

def sample_loop(z, ts, tf, n_steps, model, mode, clamping = False):
    if mode == 'sde':
        return solve_de(z, ts, tf, n_steps, model, mode, clamping)
    elif mode == 'ode':
        return solve_de(z, ts, tf, n_steps, model, mode, clamping)
    elif mode == 'marginal':
        return discrete_sampling(z, ts, tf, n_steps, model, mode, clamping)
    elif mode == 'star':
        return discrete_sampling(z, ts, tf, n_steps, model, mode, clamping)
    else:
        raise ValueError("mode must be either 'sde', 'ode', 'marginal', or 'star'")
    

####
# utility Functions
####

def clamp(model, x):
    #create a namespace called args
    args = SimpleNamespace(model_arch='iygyjgiy')
    t = "an-un-used-variable"

    #we need to modify the model somehow
    model3 = model.pred.model.word_embedding

    x = denoised_fn_round(args, model3, x, t)
    return x

###
#functions that run the outer loop
###

@torch.no_grad()
def solve_de(z, ts, tf, n_steps, model, mode, clamping = False):
    assert mode in ['sde', 'ode'], "mode must be either 'sde' or 'ode'"

    bs = z.shape[0]
    
    tt = torch.linspace(ts, tf, n_steps + 1)[:-1].to(z.device) #[:-1].to(z.device)
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5
    
    path = [z]
    for t in tqdm(tt):
        t = t.expand(bs,1,1)
        
        if mode == 'sde':
            if hasattr(model, 'vol_eta'):
                f, g = sde_drift_ndm(z, t, model, clamping)
            else:
                f, g = sde_drift(z, t, model, clamping)
        else:
            if hasattr(model, 'vol_eta'):
                f, g = ode_drift_ndm(z, t, model, clamping)
            else:
                f, g = ode_drift(z, t, model, clamping)

        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2

        path.append(z)
    
    if bs > 128:
        path = path[1:3]
        
    return z, torch.stack(path)

@torch.no_grad()
def discrete_sampling(
        z: Tensor,
        ts: float,
        tf: float,
        n_steps: int,
        model: nn.Module,
        mode: str,
        clamping: bool
):
    bs = z.shape[0]

    if mode == 'marginal':
        t_steps =  torch.linspace(ts, tf, n_steps + 1).to(z.device)[:-1]
    if mode == 'star':
        t_steps = torch.linspace(ts, tf, n_steps + 1).to(z.device)[1:]
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5

    if clamping:
        denoised_fn = clamp
    else:
        print("no clamping ---------------------------------------")
        denoised_fn = None

    path = [z]
    pbar = tqdm
    for t in pbar(t_steps):
        t = t.expand(bs, 1, 1)
        
        #I understand I am doing 2x-1 the number of needed forward pass through gamma now, Ill fix that before putting it into the actual code.
        if mode == 'marginal': 
            if all(t == 0):
                continue       
            z = get_next_marginal(prev_sample=z, t=t, s=t+dt, model=model, denoised_fn=denoised_fn)
        elif mode == 'star':
            z = get_next_star(x=z, t=t, model=model, denoised_fn=denoised_fn)

        path.append(z)

    if bs > 128:
        path = path[1:3]

    return z, torch.stack(path)

###
# functions that run the inner loop
###

def get_next_star(x, t, model, denoised_fn=None):
    def process_xstart(x):
        if denoised_fn is not None:
            # print(denoised_fn)
            x = denoised_fn(model, x)
        if False:# clip_denoised:
            return x.clamp(-1, 1)
        return x

    #print(x, "x")
    x_ = model.pred(x, t) 
    #print(x_, "x_")   
    x_start = process_xstart(x_)

    if hasattr(model, "gamma"):
        gamma, _ =  model.gamma(t)
        alpha = model.gamma.alpha_2(gamma) ** 0.5
        sigma2 =  model.gamma.sigma_2(gamma)

        m, _ = model.transform.get_m_s(x_start, t)

        mean = alpha*m
        log_variance = th.log(sigma2)
    else:
        f_m, f_s, _ = model.affine(x_start, t)
        mean = f_m
        log_variance = th.log(f_s**2)
    
    """
    if top_p is not None and top_p > 0:
        # print('top_p sampling')
        noise = th.randn_like(x)
        replace_mask = th.abs(noise) > top_p
        while replace_mask.any():
            noise[replace_mask] = th.randn_like(noise[replace_mask])
            replace_mask = th.abs(noise) > top_p
        assert (th.abs(noise) <= top_p).all()
    """
  
    noise = th.randn_like(x)

    #t.squeeze(-1).squeeze(-1)

    nonzero_mask = (
        (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    )  # no noise when t == 0
    
    sample = mean + nonzero_mask * th.exp(0.5 * log_variance) * noise #Maybe this is key!!!!!
    return sample

def get_next_marginal(prev_sample, t, s, model, denoised_fn=None):
    def process_xstart(x):
        if denoised_fn is not None:
            # print(denoised_fn)
            x = denoised_fn(model, x)
        if False:# clip_denoised:
            return x.clamp(-1, 1)
        return x
    
    #step 1 do prediction 
    x_ = model.pred(prev_sample, t) 
    
    x_start = process_xstart(x_)

    #step 2 get epsilon
    if hasattr(model, "gamma"):
        gmm, _ = model.gamma(t)
        alpha2 = model.gamma.alpha_2(gmm)
        sigma2 =  model.gamma.sigma_2(gmm)
        alpha = alpha2 ** 0.5
        sigma = sigma2 ** 0.5

        m_ , _ = model.transform.get_m_s(x_start, t)

        eps = (prev_sample - alpha * m_) / sigma
    else:
        f_m, sigma, alpha = model.affine(x_start, t)
        sigma2 = sigma ** 2
        alpha2 = alpha ** 2
        eps = (prev_sample - f_m) / sigma
        m_ = x_start

    #step 3 get epsilon s|t
    #we need stepsize for this?
    noise = torch.randn_like(prev_sample)

    if hasattr(model, "gamma"):
        gmm_s, _ = model.gamma(s)
        alpha2_s = model.gamma.alpha_2(gmm_s)
        sigma2_s =  model.gamma.sigma_2(gmm_s)
        alpha_s = alpha2_s ** 0.5
        sigma_s = sigma2_s ** 0.5
        
        m_s , _ = model.transform.get_m_s(x_start, s)
    else:
        f_m_s, sigma_s, alpha_s = model.affine(x_start, s)
        sigma2_s = sigma_s ** 2
        alpha2_s = alpha_s ** 2
        m_s = x_start

    #print(gmm_s)
    snr_t = (alpha2/sigma2)
    snr_s = (alpha2_s/sigma2_s)

    sigma2_tilde_s_t = 1 -  (snr_t / snr_s) 
    print(sigma2_tilde_s_t, "sigma2_tilde_s_t")
    #sigma2_tilde_s_t = torch.ones_like(sigma2_tilde_s_t)
    epsilon_tilde_s_t = torch.sqrt(1 - sigma2_tilde_s_t) * eps + (sigma2_tilde_s_t ** 0.5) * noise

    #print("snr_t", snr_t[0])
    #print("snr_s", snr_s[0])
    #print("sigma", sigma2_tilde_s_t) #this should be positive always but isnt so im doing something wrong. 

    #step 4 get z_s
        
    #if (s == 0).all():
    #    print("last step ")
    #    sample = alpha_s * m_s + sigma_s * torch.sqrt(1 - sigma2_tilde_s_t) * eps 
    #else:
    if all(s == 0):
        sample = alpha_s * m_s
    else:
        sample = alpha_s * m_s + sigma_s * epsilon_tilde_s_t
    
    #if we want to match appendix 1 of ndm paper I think it should instead be
    #sample = alpha_s * m_s +  torch.sqrt(sigma2 - sigma2_tilde_s_t) * eps + (sigma2_tilde_s_t ** 0.5) * noise

    return sample

###
#drift functions for DE's
###

def sde_drift(z, t, model, clamping):
    x = model.pred(z, t)

    if clamping: # and (t > 0.7).all():
        x = clamp(z, model, x)
        
    def f(t_in):
        return model.affine(x, t_in)

    (m, s, _), (dm, ds, _) = t_dir(f, t)

    g = model.vol(t)
    g2 = g ** 2

    dz = dm + ds / s * (z - m)
    score = (m - z) / s ** 2
    drift = dz - 0.5 * g2 * score

    return drift, g

def ode_drift(z, t, model, clamping):
    x = model.pred(z, t)
    #print(x, "x")
    if clamping: # and (t > 0.7).all():
        x = clamp(z, model, x)

    def f(t_in):
        return model.affine(x, t_in)

    (m, s, _), (dm, ds, _) = t_dir(f, t)
    
    dz = dm + ds / s * (z - m)

    return dz, 0

def sde_drift_ndm(z_in, t_in, model, clamping):
    gmm, d_gmm = model.gamma(t_in)
    alpha_2 = model.gamma.alpha_2(gmm)
    sigma_2 = model.gamma.sigma_2(gmm)
    alpha = alpha_2 ** 0.5

    eta = model.vol_eta(t_in)

    g = (sigma_2 * d_gmm * eta) ** 0.5

    x_ = model.pred(z_in, t_in)

    if clamping: # and (t_in > 0.7).all():
        x = clamp(z, model, x)

    (m_, _), (d_m_, _) = model.transform(x_, t_in)

    drift = -alpha * d_gmm * (1 + eta) / 2 * m_ + \
            alpha * d_m_ + \
            0.5 * d_gmm * (alpha_2 + eta) * z_in

    return drift, g

def ode_drift_ndm(z_in, t_in, model, clamping):
    gmm, d_gmm = model.gamma(t_in)
    alpha_2 = model.gamma.alpha_2(gmm)
    sigma_2 = model.gamma.sigma_2(gmm)
    alpha = alpha_2 ** 0.5
    sigma = sigma_2 ** 0.5

    x_ = model.pred(z_in, t_in)

    if clamping: # and (t_in > 0.7).all():
        x = clamp(z, model, x)

    (m_, _), (d_m_, _) = model.transform(x_, t_in)

    eps = (z_in - alpha * m_) / sigma
    alpha_prime = - d_gmm * 0.5 * alpha * (1- alpha_2) 
    sigma_prime = 0.5 * d_gmm * sigma * (1 - sigma_2)
    #dz = -alpha * d_gmm + alpha * d_m_ + sigma * d_gmm * eps
    dz = alpha_prime * m_ + alpha * d_m_ + sigma_prime * eps

    return dz, 0

