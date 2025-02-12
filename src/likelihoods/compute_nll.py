import torch
import numpy as np
from scipy import integrate
from models.diffusions import t_dir

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))


def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator.
     Computes nabla f_thingy(x) i think, using the 
  """

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      #print(eps.shape, "eps")
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn

def get_prob_flow_ode_drift(model, z, t):
    #print(z.shape)
    
    
    x = model.pred(z, t)

    def f(t_in):
        return model.affine(x, t_in)

    (m, s), (dm, ds) = t_dir(f, t)

    dz = dm + ds / s * (z - m)
    return dz

def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: get_prob_flow_ode_drift(model, xx, tt))(x, t, noise)

def get_prior_logp(z):
    """
    Computes the log density of z under a standard Gaussian.
    
    Args:
      z (torch.Tensor): A tensor of shape [batch_size, ...] representing the latent codes.
    
    Returns:
      log_prob (torch.Tensor): A tensor of shape [batch_size] with the log probability for each sample.
    """
    # Flatten z except for the batch dimension
    z_flat = z.view(z.shape[0], -1)
    d = z_flat.shape[1]  # dimensionality of the latent space
    
    # Compute the squared L2 norm of z for each sample in the batch
    norm_sq = torch.sum(z_flat ** 2, dim=1)
    
    # Compute log probability using the formula for a standard Gaussian
    log_prob = -0.5 * (d * torch.log(torch.tensor(2 * np.pi, dtype=z.dtype)) + norm_sq)
    
    return log_prob


def get_likelihood_fn(model, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):

    def likelihood_fn(data):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
        data: A PyTorch tensor.

        Returns:
        bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
        z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
        nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        with torch.no_grad():
            data = model.encoder(data)
            shape = data.shape
            if hutchinson_type == 'Gaussian':
                epsilon = torch.randn_like(data)
            elif hutchinson_type == 'Rademacher':
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
            
            #the point of this is to set up the joint ode, so that means it should include both the drift ode and the logp ode.
            #since we need both of these jointly. 

            def ode_func(t, x):
                #idk if this line is correct 
                sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
                #print(sample.shape, "sample")
                vec_t = (torch.ones(sample.shape[0], device=sample.device) * t).unsqueeze(-1) #I added the unsqueeze
                #print(vec_t.shape)
                #print(x.shape)

                # get the drift 
                drift = to_flattened_numpy(get_prob_flow_ode_drift(model, sample, vec_t))
                #get the logp_gradient 
                logp_grad = to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
                return np.concatenate([drift, logp_grad], axis=0)

            
            init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
            solution = integrate.solve_ivp(ode_func, (eps, 1), init, rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
            delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)

            #compute the prior logp
            #In our case.....
            prior_logp = get_prior_logp(z)
            
            bpd = -(prior_logp + delta_logp) / np.log(2)
            N = np.prod(shape[1:])
            bpd = bpd / N
            # A hack to convert log-likelihoods to bits/dim -> this line makes no sense. I think its a typo or something
            #inverse scalar is somethign we would need if we would scale the datapoints or embeddings in any way. 
            #but we do not do that.
            #offset = 7. - inverse_scaler(-1.)
            #bpd = bpd + offset
            return bpd, z, nfe
        
    return likelihood_fn