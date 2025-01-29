import torch
from tqdm import tqdm

from src.models.diffusions import t_dir


@torch.no_grad()
def solve_de(z, ts, tf, n_steps, module, mode, clamping = False):
    #tf = time start
    #ts = time end
    assert mode in ['sde', 'ode'], "mode must be either 'sde' or 'ode'"

    bs = z.shape[0]
    
    tt = torch.linspace(ts, tf, n_steps + 1)[:-1].to(z.device)
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5
    
    path = [z]
    for t in tqdm(tt):
        t = t.expand(bs, 1,1)
        
        if mode == 'sde':
            f, g = sde_drift(z, t, module)
        else:
            f, g = ode_drift(z, t, module)

        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2

        if clamping:
            #clamp z to the nearest actual embedding in the embedding matrix, use the dot product to find the closest embedding
            logits = module.model.decoder(z, module.model.encoder.embedding.weight)
            indices = logits.argmax(dim=-1).squeeze(-1)
            z = module.encoder.embedding(indices)

        path.append(z)
        
    return z, torch.stack(path)

def sde_drift(z, t, module):
    x = module.model.pred(z, t)

    def f(t_in):
        return module.model.affine(x, t_in)

    (m, s), (dm, ds) = t_dir(f, t)
    g = module.model.vol(t)
    g2 = g ** 2

    dz = dm + ds / s * (z - m)
    score = (m - z) / s ** 2
    drift = dz - 0.5 * g2 * score

    return drift, g

def ode_drift(z, t, module):
    x = module.model.pred(z, t)

    def f(t_in):
        return module.model.affine(x, t_in)

    (m, s), (dm, ds) = t_dir(f, t)

    dz = dm + ds / s * (z - m)

    return dz, 0