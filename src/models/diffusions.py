import torch
import torch.autograd
import torch.autograd.functional
import torch.nn as nn

from pathlib import Path
import pickle

def jvp(f, x, v):
    v.requires_grad = True
    x.requires_grad = True
    return torch.autograd.functional.jvp(
        f, x, v, 
        create_graph=torch.is_grad_enabled(),
        strict=True
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
        if hasattr(self.decoder, "lm_head"):
            with torch.no_grad():
                self.decoder.lm_head.weight = self.encoder.embedding.weight
        
    def forward(self, x, t, 
                compute_diffusion_loss=True,
                compute_prior_loss=False,
                compute_reconstruction_loss=True,
                reconstruction_loss_type = "diff_anchor"):

        def f(x_in):
            def f_(t_in):
                return self.affine(x_in, t_in)
            return f_

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
        #here it crahses
        #print("t requires grad: ", t.requires_grad)
        #print("embeddings requires grad: ", embeddings.requires_grad)

        #require grad on the embeddings to make the jvp work correctly during test and val
        #print(t.requires_grad, "t requires grad")
        #print(embeddings.requires_grad, "embeddings requires grad")

        if compute_diffusion_loss:
            g2 = self.vol(t) ** 2
            (f_m, f_s), (f_dm, f_ds) = t_dir(f(embeddings), t) #(function output), (jvp) == (mean, sigma), (mean derivative, sigma derivative)
        else:
            f_m, f_s = self.affine(embeddings, t)

        # sample z_t from q(z_t | x)
        # z_t should obtained from putting epsilon into the forward process
        eps = torch.randn_like(embeddings)
        #fix epsilon to some value
        #eps = torch.ones_like(embeddings)/10
        z = f_m + f_s * eps # function evaluation of F(e(x), t, eps)

        #also deocde z to see if pred is doing a  hard job?
        #with torch.no_grad():
        #    z_logits = self.decoder(z, self.encoder.embedding.weight)
        #    for i in range(3):
        #        print("z sequence: ", torch.argmax(z_logits[i], dim=1))

        # predict x from z_t
        #print(z, "z")
        embeddings_ = self.pred(z, t) # z is not neccerily a word embedding here.
        
        #for decoding purposes decode the embeddings to see if the pred is doing a good job.
        #print("embeddings:", embeddings)
        #print("embeddings_pred:", embeddings_)
        #with torch.no_grad():
        #    embeddings_logits = self.decoder(embeddings_, self.encoder.embedding.weight)
        #    print("embeddings sequence: ", torch.argmax(embeddings_logits[0], dim=1))

        # compute the diffusion loss
        if compute_diffusion_loss:         #self, f_dm, f_ds, f_s, eps, g2, x_,           x, z, t
            diffusion_loss = self.diffusion_loss(f_dm, f_ds, f_s, eps, g2, embeddings_ , x, z, t, f)
        else:
            diffusion_loss = torch.zeros(bs, dtype=embeddings.dtype, device=x.device)
        
        # compute the reconstruction loss
        if compute_reconstruction_loss:
            reconstruction_loss = self.reconstruction_loss(x, t, embeddings, embeddings_, bs, reconstruction_loss_type)
        else:
            reconstruction_loss = torch.zeros(bs, dtype=embeddings.dtype, device=x.device).mean()

        # compute the prior loss
        if compute_prior_loss:
            prior_loss = self.prior_loss(z, bs)
        else:
            prior_loss = torch.zeros(bs, dtype=embeddings.dtype, device=x.device)

        return diffusion_loss, reconstruction_loss, prior_loss

    def diffusion_loss(self, f_dm, f_ds, f_s, eps, g2, x_, x, z, t, f):
        # compute the drift term of the forward process based on eps
        # f_drift is what is then used in the loss and is the forward process drift.
        f_dz = f_dm + f_ds * eps  # ODE drift ---> this works because gaussians are nice and linear so the derivative of F(x, t, eps) can be written like this. 
        f_score = - eps / f_s  # score function
        f_drift = f_dz - 0.5 * g2 * f_score  # SDE drift 

        #print(f_dm, "f_dm")

        #require grad because otherwise during val it wouldnt require grad and jvp wouldnt work right
        #x_.requires_grad = True
        
        # substitute predicted \hat{x} into the forward process to parameterise the reverse process
        
        (r_m, r_s), (r_dm, r_ds) = t_dir(f(x_), t)

        # compute the drift term of the reverse process based on z_t
        r_dz = r_dm + r_ds / r_s * (z - r_m)  # ODE drift
        r_score = (r_m - z) / r_s ** 2  # score function
        r_drift = r_dz - 0.5 * g2 * r_score  # SDE drift


        # compute the diffusion loss
        loss = 0.5 * (f_drift - r_drift) ** 2 / g2
        # mask out special tokens
        #mask = x != 0 #false for [UNK] tokens which should have id zero. 
        #not_mask = ~mask
        #print(not_mask.sum(), "number of unks")
        #mask_expanded = mask.unsqueeze(-1).expand(-1, -1, loss.shape[2])
        #loss = loss * mask_expanded
        
        #do not comment out this line!!!!!!!!!!!!!!!!
        loss = loss.sum(dim=(1,2))

        return loss

    def reconstruction_loss(self, x, t, embeddings, embeddings_, bs, loss_type):
        """
        I need to have a good look at the math of this part. 
        """
        

        if loss_type == "collapse":
            f_m, f_s = self.affine(embeddings, torch.zeros_like(t))
            z_0 = f_m + f_s * torch.randn_like(embeddings)

            logits = self.decoder(z_0, self.encoder.embedding.weight) #check 
            #this doesnt make any sense, because the embeddins are being decoded by similarity to all, but I just got them from there so this will always be zero. 
        elif loss_type == "diff_anchor":
            # e prediction used in the reverse process
            logits = self.decoder(embeddings_, self.encoder.embedding.weight)
            
        #for the first 3 samples in the batch, print the actual sequence, and the sequence created by logits of the decoder
        #for i in range(3):
        #    print("Actual sequence: ", x[i])
        #    print("Predicted sequence: ", torch.argmax(logits[i], dim=1))
        #    print("\n")
        #folded_logits = logits.view(-1, logits.size(-1))
        #print(folded_logits.shape, "folded_logits_shape")
        #loss = torch.nn.functional.cross_entropy(folded_logits, x.view(-1), reduction="sum", ignore_index=0) #use ignore_index here if we ever do any padding
        #print(loss.shape, "loss_shape")
        #loss = loss/bs

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), x.view(-1)).view(x.shape)
        # print(decoder_nll.shape)
        decoder_nll = decoder_nll.mean(dim=-1)
        decoder_nll = decoder_nll / embeddings.size(-1)

        return decoder_nll

    #this should only be computed if the forward process isnt implemented to have exactly n(0,1) at time 1. 
    def prior_loss(self, embeddings, bs):
        # compute the prior loss 
        #not implemented error
        return torch.zeros(bs, dtype=embeddings.dtype, device=embeddings.device)
    

