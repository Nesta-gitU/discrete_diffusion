import torch
import torch.autograd
import torch.autograd.functional
import torch.nn as nn

def jvp(f, x, v):
    return torch.autograd.functional.jvp(
        f, x, v, 
        create_graph=torch.is_grad_enabled()
    )

def t_dir(f, t):
    return jvp(f, t, torch.ones_like(t))

class NeuralDiffusion(nn.Module):
    def __init__(self, affine, pred, vol, encoder, decoder):
        super().__init__()
        
        self.affine = affine
        self.pred = pred
        self.vol = vol

        self.encoder = encoder
        self.decoder = decoder
    
    def f(self, x_in):
            def f_(t_in):
                return self.affine(x_in, t_in)
            return f_
        
    def forward(self, x, t, 
                compute_diffusion_loss=True,
                compute_prior_loss=False,
                compute_reconstruction_loss=False,
                reconstruction_loss_type = "diff_anchor"):
        bs = x.size(0)

        #sanity check the input
        if compute_reconstruction_loss:
            if reconstruction_loss_type not in ["diff_anchor", "collapse"]:
                raise ValueError("Invalid reconstruction loss type")
        
        if not compute_diffusion_loss and not compute_reconstruction_loss:  
            raise ValueError("At least one of compute_diffusion_loss or compute_reconstruction_loss must be True")
        
        #encode the word indices x to embeddings
        embeddings = self.encoder(x)

        # compute parameters of q(z_t | x) and corresponding time derivatives
        # that means here we get the parameters of: F(x, t, eps) = _F(e(x), t, eps)
        if compute_diffusion_loss:
            g2 = self.vol(t) ** 2
            (f_m, f_s), (f_dm, f_ds) = t_dir(self.f(embeddings), t) #(function output), (jvp) == (mean, sigma), (mean derivative, sigma derivative)
        else:
            f_m, f_s = self.affine(embeddings, t)

        # sample z_t from q(z_t | x)
        # z_t should obtained from putting epsilon into the forward process
        eps = torch.randn_like(embeddings)
        z = f_m + f_s * eps # function evaluation of F(e(x), t, eps)

        # predict x from z_t
        embeddings_ = self.pred(z, t)

        # compute the diffusion loss
        if compute_diffusion_loss:
            diffusion_loss = self.diffusion_loss(f_dm, f_ds, f_s, eps, g2, embeddings_ , z, t)
        else:
            diffusion_loss = torch.zeros(bs, dtype=embeddings.dtype, device=x.device)
        
        # compute the reconstruction loss
        if compute_reconstruction_loss:
            reconstruction_loss = self.reconstruction_loss(x, embeddings, embeddings_, bs, reconstruction_loss_type)
        else:
            reconstruction_loss = torch.zeros(bs, dtype=embeddings.dtype, device=x.device)

        # compute the prior loss
        if compute_prior_loss:
            prior_loss = self.prior_loss(z, bs)
        else:
            prior_loss = torch.zeros(bs, dtype=embeddings.dtype, device=x.device)


        
        return diffusion_loss, reconstruction_loss, prior_loss

    def diffusion_loss(self, f_dm, f_ds, f_s, eps, g2, x_, z, t):
        # compute the drift term of the forward process based on eps
        # f_drift is what is then used in the loss and is the forward process drift.
        f_dz = f_dm + f_ds * eps  # ODE drift ---> this works because gaussians are nice and linear so the derivative of F(x, t, eps) can be written like this. 
        f_score = - eps / f_s  # score function
        f_drift = f_dz - 0.5 * g2 * f_score  # SDE drift 

        
        # substitute predicted \hat{x} into the forward process to parameterise the reverse process
        (r_m, r_s), (r_dm, r_ds) = t_dir(self.f(x_), t)

        # compute the drift term of the reverse process based on z_t
        r_dz = r_dm + r_ds / r_s * (z - r_m)  # ODE drift
        r_score = (r_m - z) / r_s ** 2  # score function
        r_drift = r_dz - 0.5 * g2 * r_score  # SDE drift

        # compute the diffusion loss
        loss = 0.5 * (f_drift - r_drift) ** 2 / g2
        loss = loss.sum(dim=1)

        return loss

    def reconstruction_loss(self, x, embeddings, embeddings_, bs, loss_type):
        """
        I need to have a good look at the math of this part. 
        """
        if loss_type == "collapse":
            logits = self.decoder(embeddings, self.encoder.weight)
        elif loss_type == "anchor_diff":
            # e prediction used in the reverse process
            logits = self.decoder(embeddings_, self.encoder.weight)

        # Reshape logits and x to compute token-wise cross-entropy loss
        block_size = x.shape[1]
        
        logits = logits.view(bs * block_size, -1)  # [batch_size * block_size, vocab_size]
        x = x.view(-1)  # [batch_size * block_size]

        # Compute cross-entropy loss for each token
        token_loss = torch.nn.functional.cross_entropy(logits, x, reduction="none")  # [batch_size * block_size]

        # Reshape back to [batch_size, block_size] and sum losses over tokens
        token_loss = token_loss.view(bs, block_size).sum(dim=1)  # [batch_size]

        return token_loss

    #this should only be computed if the forward process isnt implemented to have exactly n(0,1) at time 1. 
    def prior_loss(self, embeddings, bs):
        # compute the prior loss 
        #not implemented error
        return torch.zeros(bs, dtype=embeddings.dtype, device=x.device)