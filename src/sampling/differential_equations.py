from math import nan
import torch
from tqdm import tqdm
from src.models.nfdm.nfdm import t_dir
from src.their_utils.test_util import denoised_fn_round
from src.models.ndm.components.context import VaeContext
from src.models.ndm.components.gamma import GammaTheirs, GammaLinear
import torch
from torch import nn
from types import SimpleNamespace
from torch import Tensor
import torch as th

from contextlib import contextmanager
from src.models.nfdm.components.forward_process import NFDM_gaussian

@contextmanager
def double_precision():
    # Save the current dtype
    current_dtype = torch.get_default_dtype()
    
    # Set default dtype to float64 (double precision)
    torch.set_default_dtype(torch.float64)
    
    try:
        yield
    finally:
        # Revert to the original dtype after the block
        torch.set_default_dtype(current_dtype)

###
# Orchestrator
###
from src.models.ndm.components.context import NoneContext

def sample_loop(z, ts, tf, n_steps, model, mode, clamping = False):
    with torch.no_grad():
        if not hasattr(model, 'context'):
            model.context = NoneContext(None)
        if isinstance(model.context, VaeContext):
            context = model.context.sample_context(z)
        else:
            context = None

        if mode == 'sde':
            return solve_de(z, ts, tf, n_steps, model, mode, clamping, context)
        if mode == "better_sde":
            return better_solve_de(z, ts, tf, n_steps, model, mode, clamping, context)
        elif mode == 'ode':
            return solve_de(z, ts, tf, n_steps, model, mode, clamping, context)
        elif mode == 'marginal':
            return discrete_sampling(z, ts, tf, n_steps, model, mode, clamping, context)
        elif mode == 'star':
            return discrete_sampling(z, ts, tf, n_steps, model, mode, clamping, context)
        else:
            raise ValueError("mode must be either 'sde', 'ode', 'marginal', or 'star'")
    

####
# utility Functions
####

def clamp(model, x, t):
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
def corrector_step(
    z: torch.Tensor,
    t: torch.Tensor,
    model,
    snr: float = 0.2,
    n_steps: int = 1,
    clamp_fn: callable = None,
    context=None,
):
    """
    One or more Langevin‐corrector steps at time t.

    Args:
        z         (bs×…): current latent
        t         (bs×1×1): current time
        model          : your score model with `.pred(z,t)` and `.affine(x,tt)`
        snr            : signal‐to‐noise ratio for step‐size
        n_steps        : how many MCMC updates to do
        clamp_fn       : optional clamp(x) to keep predictions in‐bounds
        context        : if your sde_drift needs extra context

    Returns:
        z_new   (bs×…): updated latent after corrector steps
    """
    bs = z.shape[0]

    for _ in range(n_steps):
        # 1) Predict x_t = E[x|z,t]
        x = model.pred(z, t)
        if clamp_fn is not None:
            x = clamp_fn(x)

        # 2) Get (m,s) from your affine decoder via t_dir
        #    assuming you have already imported t_dir
        (m, s, _), _ = t_dir(lambda tt: model.affine(x, tt), t)

        # 3) Compute the score: ∇ log p_t(z) ≈ (m – z) / s²
        score = (m - z) / (s * s)

        # 4) Compute step‐size α via SNR heuristic
        #    ‖ξ‖ and ‖score‖ averaged over the batch
        noise = torch.randn_like(z)
        grad_norm  = torch.norm(score.view(bs, -1),  dim=-1).mean()
        noise_norm = torch.norm(noise.view(bs, -1),  dim=-1).mean()
        alpha = (snr * noise_norm / grad_norm)**2 * 2

        # 5) Langevin update: z ← z + α score + √(2α) ξ
        z = z + alpha * score + torch.sqrt(2 * alpha) * noise

    return z

@torch.no_grad()
def solve_de(z, ts, tf, n_steps, model, mode, clamping = False, context=None):
    assert mode in ['sde', 'ode'], "mode must be either 'sde' or 'ode'"

    bs = z.shape[0]
    
    tt = torch.linspace(ts, tf, n_steps + 1)[:-1].to(z.device) #[:-1].to(z.device)
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5


    
    if bs > 64:
        path = None
    else:
        path = [z]
    for t in tqdm(tt):
        t = t.expand(bs,1,1)
        
        if mode == 'sde':
            if hasattr(model, 'vol_eta'):
                f, g = sde_drift_ndm(z, t, model, clamping, context)
            else:
                f, g = sde_drift(z, t, model, clamping)
        else:
            if hasattr(model, 'vol_eta'):
                f, g = ode_drift_ndm(z, t, model, clamping, context)
            else:
                f, g = ode_drift(z, t, model, clamping)

        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2

        #langevin corrector step
        corrector_step(z, t, model, n_steps=4, clamp_fn=None, context=context)

        if path is not None:
            path.append(z)
    
    ret_path = torch.stack(path) if path is not None else None
    return z, ret_path

@torch.no_grad()
def discrete_sampling(
        z: Tensor,
        ts: float,
        tf: float,
        n_steps: int,
        model: nn.Module,
        mode: str,
        clamping: bool,
        context = None
):
    bs = z.shape[0]

    if mode == 'marginal':
        t_steps =  torch.linspace(ts, tf, n_steps + 1).to(z.device)[:-1]
    if mode == 'star':
        t_steps = torch.linspace(ts, tf, n_steps + 1).to(z.device)[1:]
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5

    if clamping:
        print("clamping ---------------------------------------")
        denoised_fn = clamp
    else:
        print("no clamping ---------------------------------------")
        def unk_remover(model, x, t):
            #the idea is to clamp find the indexes that are currently defined as unk tokens -> unk token index is 2
            # x gives the current word embeddings of this step of the diffusion language model 
            #clamp the x to the word embeddings
            if True: #torch.all(t<0.1):
                all_embeddings = model.pred.model.word_embedding.weight
                clamped = torch.argmax(x @ all_embeddings.T, dim=-1)

                randn = torch.randn_like(x)
                x = torch.where((clamped == 2).unsqueeze(-1), randn, x)
            return x

        denoised_fn = None


    print(bs, "bs")
    if bs > 64:
        path = None
    else:
        path = [z]
    print("path", path)
    pbar = tqdm
    for t in pbar(t_steps):
        t = t.expand(bs, 1, 1)
        
        #I understand I am doing 2x-1 the number of needed forward pass through gamma now, Ill fix that before putting it into the actual code.
        #print(t+dt, "should be 0.99 somthing to 0 or I guess until dt")
        if mode == 'marginal': 
            if all(t == 0):
                continue       
            z, z_mean = get_next_marginal(prev_sample=z, t=t, s=t+dt, model=model, denoised_fn=denoised_fn, context=context)
        elif mode == 'star':
            z = get_next_star(x=z, t=t, model=model, denoised_fn=denoised_fn, context=context)

        if path is not None:
            path.append(z_mean)


    ret_path = torch.stack(path) if path is not None else None

    return z, ret_path 
###
# functions that run the inner loop
###

def get_next_star(x, t, model, denoised_fn=None, context=None):
    def process_xstart(x):
        if denoised_fn is not None:
            # print(denoised_fn)
            x = denoised_fn(model, x, t)
        if False:# clip_denoised:
            return x.clamp(-1, 1)
        return x

    #print(x, "x")
    x_ = model.pred(x, t, context) 
    #print(x_, "x_")   
    x_start = process_xstart(x_)

    if hasattr(model, "gamma"):
        if context is None: 
            context = model.context.sample_context(x_)
        

        if context is None:
            gamma, _ = model.gamma(t)
        else:
            gamma, _ = model.gamma(t, context)

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

def get_next_marginal(prev_sample, t, s, model, denoised_fn=None, context=None):
    with double_precision():
        def process_xstart(x):
            if denoised_fn is not None:
                # print(denoised_fn)
                x = denoised_fn(model, x, t)
            if False:# clip_denoised:
                return x.clamp(-1, 1)
            return x
        
        #step 1 do prediction 
        x_ = model.pred(prev_sample, t, context) 
        
        x_start = process_xstart(x_)

        #step 2 get epsilon
        if hasattr(model, "gamma"):
            if context is None:
                context = model.context.sample_context(x_)

            if context is None:
                gmm, _ = model.gamma(t)
            else:
                gmm, _ = model.gamma(t, context)

            alpha2 = model.gamma.alpha_2(gmm)
            sigma2 =  model.gamma.sigma_2(gmm)
            alpha = alpha2 ** 0.5
            sigma = sigma2 ** 0.5

            m_ , _ = model.transform.get_m_s(x_start, t)

            eps = (prev_sample - alpha * m_) / sigma
        else:
            f_m, sigma, alpha = model.affine(x_start, t)
            if isinstance(model.affine, NFDM_gaussian):
                gamma_ref = GammaTheirs()
                gmm = gamma_ref.get_gamma(t)
                if False:
                    eta     = 1.0
                    gamma_0 = torch.tensor(-10.0, device=t.device, dtype=f_m.dtype)
                    N       = 100
                    bs = t.shape[0]

                    # 1) build a uniform grid from 0 to t
                    zero_to_t = torch.linspace(0.0, t[0,0,0], N, device=t.device,  dtype=f_m.dtype).unsqueeze(-1)        # shape (N,)
                    dt        = zero_to_t[1] - zero_to_t[0]                       # scalar

                    # 2) approximate the integral G(t) = ∫0^t g(s)^2 ds
                    g_vals       = model.vol(zero_to_t)                            # shape (N,)
                    g2_cumsum    = torch.cumsum(g_vals**2, dim=0) * dt             # shape (N,)
                    G_t_approx   = g2_cumsum[-1].unsqueeze(-1)                                   # scalar

                    # 3) form the softplus argument
                    sp_arg = (G_t_approx / eta) + torch.nn.functional.softplus(gamma_0)

                    # 4) invert softplus in a numerically stable way
                    threshold = 20.0
                    gmm = torch.where(
                        sp_arg > threshold,
                        sp_arg,
                        torch.log(torch.expm1(sp_arg))
                    )  # this is your γ(t)

                # 5) now plug into your GammaLinear
                gamma_ref = GammaLinear()
                alpha2 = gamma_ref.alpha_2(gmm)
                sigma2 = gamma_ref.sigma_2(gmm)
            else:
                sigma2 = sigma ** 2
                alpha2 = alpha ** 2
            eps = (prev_sample - f_m) / sigma

        #step 3 get epsilon s|t
        #we need stepsize for this?
        noise = torch.randn_like(prev_sample)

        if hasattr(model, "gamma"):
            if context is None:
                context = model.context.sample_context(x_)
                #This is wrong!!!!! it should never resample context ever only the first time
                #wait maybe im wrong
            #TODO, in case of NN = a,b,c context this is currently incorrect wrong x_. cause hmm actually does that mean also m_s is incorrect?

            if context is None:
                gmm_s, _ = model.gamma(s)
            else:
                gmm_s, _ = model.gamma(s, context)

            alpha2_s = model.gamma.alpha_2(gmm_s)
            sigma2_s =  model.gamma.sigma_2(gmm_s)
            alpha_s = alpha2_s ** 0.5
            sigma_s = sigma2_s ** 0.5
            
            m_s , _ = model.transform.get_m_s(x_start, s)
        else:
            f_m_s, sigma_s, alpha_s = model.affine(x_start, s)
            if isinstance(model.affine, NFDM_gaussian):
                gamma_ref = GammaTheirs()
                gmm_s = gamma_ref.get_gamma(s)
                if False:
                    eta     = 1.0
                    gamma_0 = torch.tensor(-10.0, device=t.device, dtype=f_m_s.dtype)
                    N       = 100
                    bs = t.shape[0]

                    # 1) build a uniform grid from 0 to t
                    zero_to_t = torch.linspace(0.0, s[0,0,0], N, device=t.device, dtype=f_m_s.dtype).unsqueeze(-1)        # shape (N,)
                    
                    dt        = zero_to_t[1] - zero_to_t[0]                       # scalar

                    # 2) approximate the integral G(t) = ∫0^t g(s)^2 ds
                    g_vals       = model.vol(zero_to_t)                            # shape (N,)
                    g2_cumsum    = torch.cumsum(g_vals**2, dim=0) * dt             # shape (N,)
                    G_t_approx   = g2_cumsum[-1].unsqueeze(-1)                            # scalar

                    # 3) form the softplus argument
                    sp_arg = (G_t_approx / eta) + torch.nn.functional.softplus(gamma_0)

                    # 4) invert softplus in a numerically stable way
                    threshold = 20.0
                    gmm_s = torch.where(
                        sp_arg > threshold,
                        sp_arg,
                        torch.log(torch.expm1(sp_arg))
                    )  # this is your γ(t)

                    gamma_ref = GammaLinear()
                alpha2_s = gamma_ref.alpha_2(gmm_s)
                sigma2_s = gamma_ref.sigma_2(gmm_s)
            else:
                sigma2_s = sigma_s ** 2
                alpha2_s = alpha_s ** 2
            m_s = f_m_s #should not be this in general nfdm case only works if static. 
    

        #print(gmm_s) cast to higher precision
        snr_t = (alpha2/sigma2).double()
        snr_s = (alpha2_s/sigma2_s).double()

        #sigma2_tilde_s_t = 1 -  (snr_t / snr_s) 
        #print(sigma2_tilde_s_t, "sigma2_tilde_s_t")
        #sigma2_tilde_s_t = torch.ones_like(sigma2_tilde_s_t)
        if hasattr(model, "gamma"):
            if torch.any(gmm_s > gmm):
                pass
            #    print(t, s)
            if model.gamma.around_reference:
                # get reference gammas of shape [bs]
                ref_s = model.gamma.get_reference_gamma(s)
                ref_t = model.gamma.get_reference_gamma(t)

                # compute a view shape [bs, 1, 1, …] matching gmm_s.dim()
                view_shape = [ref_s.size(0)] + [1] * (gmm_s.dim() - 1)
                ref_s = ref_s.view(*view_shape)
                ref_t = ref_t.view(*view_shape)

                # expand to exactly match gmm_s and gmm
                gmm_s_r = ref_s.expand_as(gmm_s)
                gmm_r   = ref_t.expand_as(gmm)
                sigma2_tilde_s_t = -torch.expm1(gmm_s_r - gmm_r)
            else:
                #sigma2_tilde_s_t = -torch.expm1(gmm_s - gmm) #-(torch.exp(gmm_s - gmm)-1) = 1-torch.exp(gmm_s - gmm) => gmm > gmm_s so quantity should be positive
                sigma2_tilde_s_t = 1+torch.exp(gmm_s - gmm)
                #the plus 1 is the fix 
            #print(sigma2_tilde_s_t, "sigma2_tilde_s_t, before clamp")
            #print(torch.all(gmm == gmm_s), "gmm == gmm_s")
            #print(gmm_s, "gmm_s")
            #print(s, t, "s, t")
            sigma2_tilde_s_t = torch.clamp(sigma2_tilde_s_t, 0, 1)
            if torch.any(sigma2_tilde_s_t > 1) or torch.any(sigma2_tilde_s_t < 0):
                #print("sigma2_tilde_s_t out of bounds", sigma2_tilde_s_t[sigma2_tilde_s_t > 1], sigma2_tilde_s_t[sigma2_tilde_s_t < 0])
                #print("snr_t", snr_t[0])
                #print("snr_s", snr_s[0])

                #print(gmm, "gmm")
                #print(gmm_s, "gmm_s")
                #print(sigma2_tilde_s_t)
                #raise ValueError("sigma2_tilde_s_t out of bounds")
                pass
                
            if torch.any(sigma2_tilde_s_t == nan):
                #print("sigma2_tilde_s_t is None", sigma2_tilde_s_t[sigma2_tilde_s_t == nan])
                raise ValueError("sigma2_tilde_s_t is None")
        else:
            sigma2_tilde_s_t = (snr_t/snr_s).float() #(1 - (snr_t / snr_s)).float()
            #print(sigma2_tilde_s_t, "sigma2_tilde_s_t, before clamp")
            sigma2_tilde_s_t = torch.clamp(sigma2_tilde_s_t, 0, 1)
            #print(sigma2_tilde_s_t, "sigma2_tilde_s_t")

        sigma2_tilde_s_t = torch.zeros_like(sigma2_tilde_s_t) + 0.8  # -> this works quite well it did a mauve of 0.99
        if torch.all(s < 0.1):
                sigma2_tilde_s_t = torch.zeros_like(sigma2_tilde_s_t) + 0.3
        epsilon_tilde_s_t = torch.sqrt(1 - sigma2_tilde_s_t) * eps + (sigma2_tilde_s_t.sqrt()) * noise

        #print("snr_t", snr_t[0])
        #print("snr_s", snr_s[0])
        #print("sigma", sigma2_tilde_s_t) #this should be positive always but isnt so im doing something wrong. 

        #step 4 get z_s
            
        #if (s == 0).all():
        #    print("last step ")
        #    sample = alpha_s * m_s + sigma_s * torch.sqrt(1 - sigma2_tilde_s_t) * eps 
        #else:
        if hasattr(model, "gamma"):
            mean = alpha_s * m_s
        else:
            mean = f_m_s

        if all(s == 0):
            if hasattr(model, "gamma"):
                sample = alpha_s * m_s
            else:
                sample = f_m_s
            #sample 
        else:
            if hasattr(model,"gamma"):
                sample = alpha_s * m_s + sigma_s * epsilon_tilde_s_t
            else:
                sample = f_m_s + sigma_s * epsilon_tilde_s_t
    
        #if we want to match appendix 1 of ndm paper I think it should instead be
        #sample = alpha_s * m_s +  torch.sqrt(sigma2 - sigma2_tilde_s_t) * eps + (sigma2_tilde_s_t ** 0.5) * noise
    return sample, mean

import torch
import torchsde
from torchsde import BrownianInterval

class ReverseSDE(torch.nn.Module):

    def __init__(self, model, drift_fn, drift_shape, clamping=False, context=None):
        super().__init__()
        self.model = model
        self.drift_fn = drift_fn
        self.clamping = clamping
        self.context = context
        self.noise_type = "diagonal"   # or "diagonal" if you ever switch back
        self.sde_type   = "ito"
        self.drift_shape = drift_shape

    def f(self, u, y):
        # u goes from 0 → 1, so t goes from 1 → 0
        t = 1.0 - u
        #print(t==1)
        t_ = t.expand(y.shape[0]).unsqueeze(-1)
        # sde_drift returns the _reverse_ drift already:
        #print("y", y.shape)
        #print(t_)
        
        #unflatten last dim so it fits into the model
        y = y.view(self.drift_shape)

        drift, _ = self.drift_fn(y, t_, model=self.model, clamping=self.clamping, context=self.context)
        #this drift has shape bs, seqlen, hidden_dim -> should be bs,  state_size ----> flatten the last dim 
        drift = drift.flatten(start_dim=1)

        return -drift

    def g(self, u, y):
        bs, D = y.shape
        t = 1.0 - u
        t_ = t.expand(y.shape[0]).unsqueeze(-1)
        # model.vol(t_) is your learned scalar volatility
        g_val = self.model.vol(t_)
        
        return g_val.view(bs, 1).expand(bs, D)

def better_solve_de(z, ts, tf, n_steps, model, mode, clamping = False, context=None):	
    if mode == 'sde':
        if hasattr(model, 'vol_eta'):
            drift_fn = sde_drift_ndm
        else:
            drift_fn = sde_drift
    else:
        if hasattr(model, 'vol_eta'):
            drift_fn = ode_drift_ndm
        else:
            drift_fn = ode_drift

    sde  = ReverseSDE(model=model, drift_fn=drift_fn, drift_shape = z.shape, clamping=clamping, context=context)
    us   = torch.linspace(0.0, 1.0, 40, device=z.device)

    print(us[0], us[1], us[-1])
    dt   = 1.0 / n_steps
    print(dt)
    print(dt == us[1])
    z_old = z
    z = z.flatten(start_dim=1)

    # z0 should be pure noise at u=0
    with torch.no_grad():
        z_path = torchsde.sdeint(
            sde=sde,
            y0=z, #this z has shape bs, seqlen, hidden_dim -> should be bs,  state_size ----> flatten the last dim and then unflatten
            ts=us,
            method="euler",    # or milstein, srk, etc.
            dt=dt
        )
    
    print(z_path.shape, "z_path shape")
    z_path = z_path.view(-1, z_old.shape[0], z_old.shape[1], z_old.shape[2])

    
    return z_path[-1], z_path

###
#drift functions for DE's
###

def sde_drift(z, t, model, clamping, context=None):
    x = model.pred(z, t)

    if clamping: # and (t > 0.7).all():
        x = clamp(model, x)
        
    def f(t_in):
        return model.affine(x, t_in)

    (m, s, _), (dm, ds, _) = t_dir(f, t)

 
    g = model.vol(t)
    g2 = g ** 2

    dz = dm + ds / s * (z - m)
    score = (m - z) / s ** 2
    drift = dz - 0.5 * g2 * score

    return drift, g

def ode_drift(z, t, model, clamping, context=None):
    x = model.pred(z, t)
    #print(x, "x")
    if clamping: # and (t > 0.7).all():
        x = clamp(model, x)

    def f(t_in):
        return model.affine(x, t_in)

    (m, s, _), (dm, ds, _) = t_dir(f, t)
    
    dz = dm + ds / s * (z - m)

    return dz, 0

def sde_drift_ndm(z_in, t_in, model, clamping, context):
    x_ = model.pred(z_in, t_in)
    if context is None:
        context = model.context.sample_context(x_)

    if context is None:
        gmm, d_gmm = model.gamma(t_in)
    else:
        gmm, d_gmm = model.gamma(t_in, context)

    alpha_2 = model.gamma.alpha_2(gmm)
    sigma_2 = model.gamma.sigma_2(gmm)
    alpha = alpha_2 ** 0.5

    eta = model.vol_eta(t_in)

    g = (sigma_2 * d_gmm * eta) ** 0.5

    
    if clamping: # and (t_in > 0.7).all():
        x_ = clamp(model, x_)

    (m_, _), (d_m_, _) = model.transform(x_, t_in)

    drift = -alpha * d_gmm * (1 + eta) / 2 * m_ + \
            alpha * d_m_ + \
            0.5 * d_gmm * (alpha_2 + eta) * z_in

    return drift, g

def ode_drift_ndm(z_in, t_in, model, clamping, context):
    x_ = model.pred(z_in, t_in)

    if context is None:
        context = model.context.sample_context(x_)

    if context is None:
        gmm, d_gmm = model.gamma(t_in)
    else:
        gmm, d_gmm = model.gamma(t_in, context)

    alpha_2 = model.gamma.alpha_2(gmm)
    sigma_2 = model.gamma.sigma_2(gmm)
    alpha = alpha_2 ** 0.5
    sigma = sigma_2 ** 0.5

    x_ = model.pred(z_in, t_in)

    if clamping: # and (t_in > 0.7).all():
        x_ = clamp(model, x_)

    (m_, _), (d_m_, _) = model.transform(x_, t_in)

    eps = (z_in - alpha * m_) / sigma
    alpha_prime = - d_gmm * 0.5 * alpha * (1- alpha_2) 
    sigma_prime = 0.5 * d_gmm * sigma * (1 - sigma_2)
    #dz = -alpha * d_gmm + alpha * d_m_ + sigma * d_gmm * eps
    dz = alpha_prime * m_ + alpha * d_m_ + sigma_prime * eps

    return dz, 0

