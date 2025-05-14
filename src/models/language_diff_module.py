from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch._C import device
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import inspect
from torch.optim.lr_scheduler import LambdaLR

import copy
from collections import OrderedDict
from their_utils.nn import mean_flat
from torch.optim.swa_utils import AveragedModel
from src.models.time_samplers.time_samplers import TimeSampler, UniformBucketSampler, ContSampler
import math
#from muon import Muon 
from src.utils.utils import MuonLightning
from torch import nn
#from src.likelihoods.compute_nll import get_likelihood_fn



class DiffusionModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        diffusion,
        total_steps: int,
        optimizer: torch.optim.Optimizer,
        compile: bool,
        time_sampler,
        use_scheduler: bool = True,
        grad_clip: float = float('inf'),
        grad_clipping_type: str = "always", #options are always, warmup, dynamic
        beta_vae_anneal: int = 1,
        use_full_elbo_in_is: bool = True,
        compute_diffusion_loss: bool = True,
        compute_prior_loss: bool = False,
        compute_reconstruction_loss: bool = True,
        reconstruction_loss_type: str = "diff_anchor",
        mask_padding: bool = False,
        enable_matmul_tf32: bool = False,
        enable_cudnn_tf32: bool = False,
        switch_to_rescaled: int = None,
        reduction_type: str = "mean", #reduction type should be specified if you want to do summing instead ----->>>>>>>
        use_muon: bool = False,
        do_lr_warmup: bool = False,
        muon_params: Dict[str, Any] = None,
    ) -> None:

        """Initialize a `Diffusion Module`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.use_full_elbo_in_is = use_full_elbo_in_is
        self.switch_to_rescaled = switch_to_rescaled

        if use_full_elbo_in_is and not isinstance(time_sampler, TimeSampler):
            self.use_full_elbo_in_is = False
            print("use_full_elbo_in_is is set to False because time_sampler is not a TimeSampler")

        self.time_sampler = time_sampler
        #hack to make it so you can continue a training run with a different sampler
        if not hasattr(self.time_sampler, "_logits"):
            self.time_sampler._logits = nn.Parameter(torch.ones(100), requires_grad=False)
        self.model = diffusion
        self.max_steps = total_steps
        self.mask_padding = mask_padding
        self.grad_clip = grad_clip
        self.beta_vae_anneal = beta_vae_anneal
        self.beta = 0.0
        self.reduction_type = reduction_type
        self.use_muon = use_muon
        self.do_lr_warmup = do_lr_warmup
        self.muon_params = muon_params
        self.clip_warmup = 6000 if self.use_muon else 3000
        print(self.model)
        print(self.model.pred)

        #self.ema = copy.deepcopy(self.model)
        #self.ema.to("cpu")
        #for p in self.ema.parameters():
        #    p.requires_grad = False

        #self.update_ema(self.ema, self.model, decay=0) 
        #self.ema.eval()
        self.ema = AveragedModel(self.model, avg_fn=lambda avg, new, _: 0.9999 * avg + (1 - 0.9999) * new)
        self.avg_grad_norm = MeanMetric()
        for p in self.ema.parameters():
            p.requires_grad = False

        self.automatic_optimization = not use_muon
        self.flag = False
        
        

    def forward(self, t, x: torch.Tensor, compute_diffusion_loss, compute_reconstruction_loss, compute_prior_loss, reconstruction_loss_type) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        #t = torch.rand(x.size(0), 1).unsqueeze(2).to(x.device) #sample a random time for each example in the batch
        #fix t to 0.5 for debug
        #t = torch.ones(x.size(0), 1).unsqueeze(2).to(x.device)/100
        t.requires_grad = True


        #print(t, "t ")
        #print(t.requires_grad, "t requires grad")
        #for i in range(3):
        #    print(f"t: {t[i]}")
        # sample just t = 1 the most corupted noise level
        #t = torch.ones(x.size(0), 1).unsqueeze(2).to(x.device)

        #add diffusion_loss_full_elbo here, also add masking of padding to it
        #also find everywhere forward is being called and change the output signature there
        #also add a None checker on diffusion_loss_full_elbo and if it is none 
        #and the setting for use diffusion loss full elbo is was set, then throw an error for incompatible combination of settings
        #also put correct ifstatements and stuff in train
        #also finally if use diffusion_loss_full_elbo and IS not used automatically set use diff loss to False


        
        if self.switch_to_rescaled is not None and (self.switch_to_rescaled == "now" or self.global_step >= self.switch_to_rescaled):
            if (self.switch_to_rescaled == self.global_step) or (self.switch_to_rescaled=="now"):
                self.switch_to_rescaled = 0
                #turn of gradients on all but the predictor 
                print("switching to rescaled")
                if hasattr(self.model, "affine"):
                    noise_params = list(self.model.affine.parameters()) + list(self.model.vol.parameters())
                else:
                    noise_params = list(self.model.transform.parameters()) + list(self.model.gamma.parameters()) + list(self.model.vol_eta.parameters()) + list(self.model.context.parameters())
                for p in noise_params: p.requires_grad_(False)
            diffusion_loss, context_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss = self.model.get_losses(x, t, 
                                                                                None, 
                                                                                compute_prior_loss, 
                                                                                compute_reconstruction_loss, 
                                                                                reconstruction_loss_type,
                                                                                compute_their_loss=True)  
        else:
            diffusion_loss, context_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss = self.model.get_losses(x, t, 
                                                                                    compute_diffusion_loss, 
                                                                                    compute_prior_loss, 
                                                                                    compute_reconstruction_loss, 
                                                                                    reconstruction_loss_type,
                                                                                    compute_their_loss=False)  

        if context_loss is None:
            context_loss = torch.zeros_like(diffusion_loss)

        #keep everything unreduced before this 
        #then use an if statement to choose to either mean or sum reduce everything 

        def sum_reduce(x):
            #sum over all but the first dimension
            return x.flatten(start_dim=1).sum(dim=1)
        
        reduce_function = sum_reduce if self.reduction_type == "sum" else mean_flat

                         
        #reduce correctly 
        if self.mask_padding:
            #print("masking padding")
            pad_mask = x == 3 #shape [B, Seqlen]
            diffusion_loss = diffusion_loss.masked_fill(pad_mask.unsqueeze(-1), 0) # shape [B, Seqlen, Embed]
            N_over_S = diffusion_loss.shape[1] / (~pad_mask).sum(dim=-1).float()
            diffusion_loss = mean_flat(diffusion_loss)* N_over_S

            if diffusion_loss_full_elbo is not None:
                diffusion_loss_full_elbo = diffusion_loss_full_elbo.masked_fill(pad_mask.unsqueeze(-1), 0)
                N_over_S = diffusion_loss_full_elbo.shape[1] / (~pad_mask).sum(dim=-1).float()
                diffusion_loss_full_elbo = mean_flat(diffusion_loss_full_elbo)* N_over_S

        else:
            diffusion_loss = reduce_function(diffusion_loss)   
            diffusion_loss_full_elbo = reduce_function(diffusion_loss_full_elbo) if diffusion_loss_full_elbo is not None else None
        

        prior_loss = reduce_function(prior_loss)
        reconstruction_loss = reduce_function(reconstruction_loss)
        context_loss = reduce_function(context_loss)

        

        return diffusion_loss, context_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        print("haloooooooooooooooooooooooooooo")
        print(f"[RANK {self.global_rank}] on device {self.device}")
        print(self.device, "device is a device is a device and stuff")
    

    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        t, p = self.time_sampler(bs=batch.size(0), device=batch.device)
        t = t.unsqueeze(-1)
        if p is not None:
            p = p.unsqueeze(-1).unsqueeze(-1)
            p = p.detach()
            #dnew_t = torch.arange(0, 1, 0.1).unsqueeze(-1).to(t.device)
            #print("probability dist over t", self.time_sampler.prob(new_t))        
        
        t = t.detach()

        #with torch.no_grad():
        diffusion_loss, context_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss = self.forward(t, batch,
                                            compute_diffusion_loss=self.hparams.compute_diffusion_loss,
                                            compute_prior_loss=self.hparams.compute_prior_loss,
                                            compute_reconstruction_loss=self.hparams.compute_reconstruction_loss,
                                            reconstruction_loss_type = self.hparams.reconstruction_loss_type)
        
        

        if isinstance(self.time_sampler, TimeSampler):
            if (diffusion_loss_full_elbo is not None) and self.use_full_elbo_in_is:
                #print(diffusion_loss_full_elbo.shape)
                #print(diffusion_loss.shape)
                is_loss = diffusion_loss / p + self.time_sampler.loss(diffusion_loss_full_elbo.detach(), t)
                #this is super suspicious I need to study the use of importance sampling in this way 
            else:
                is_loss = diffusion_loss / p + self.time_sampler.loss(diffusion_loss.detach(), t)
            self.log("train/is_loss", is_loss.mean(), on_step=True, prog_bar=True, logger=True)
            elbo = is_loss + reconstruction_loss + prior_loss + context_loss
        else:
            elbo = diffusion_loss + reconstruction_loss + prior_loss + context_loss
        
        diffusion_loss = diffusion_loss.mean()
        reconstruction_loss = reconstruction_loss.mean()
        prior_loss = prior_loss.mean()
        context_loss = self.beta * context_loss.mean()
        elbo = elbo.mean()
        #print(diffusion_loss, "----------------------diffusion_loss")

        #print(f"diffusion_loss: {self.hparams.compute_diffusion_loss}, reconstruction_loss: {self.hparams.compute_reconstruction_loss}, prior_loss: {self.hparams.compute_prior_loss}")

        #note elbo may or may not be valid depending on what we actualy calculatte in the forward pass
        
        # update and log metrics
        self.log("train/diffusion_loss", diffusion_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/reconstruction_loss", reconstruction_loss, on_step=True, prog_bar=True,logger=True, sync_dist=True)
        self.log("train/prior_loss", prior_loss, on_step=True, prog_bar=True,logger=True, sync_dist=True)
        self.log("train/elbo", elbo, on_step=True, prog_bar=False,logger=True, sync_dist=True)
        self.log("train/context_loss", context_loss, on_step=True, prog_bar=False,logger=True, sync_dist=True)

        '''
        missing = []
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is None:
                missing.append(name)
        if missing:
            print(f"[!] No grad for {len(missing)} params:", missing)

        still_have = []
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                still_have.append(name)
        if still_have:
            print(f"[!] Still have grad for {len(still_have)} params:", still_have)
        '''
        # return loss or backpropagation will fail
        if self.automatic_optimization:
            return elbo

        # (3) otherwise, manual optimization (Muon + AdamW)
        #    manual_backward() will invoke your on_after_backward() hook,
        #    so gradient‐clipping & EMA update still happen there.
        optimizers = self.optimizers()        # [optim_muon, optim_adamw]
        schedulers = self.lr_schedulers()     # [sched_muon, sched_adamw]

        #print("I do think we are getting here?")
        self.manual_backward(elbo) #-> manual backward should call on_after_backward, but clearly it doesnt 
        self.on_after_backward()

        #print(optimizers[0].optimizer.param_groups)

        # step & zero out each optimizer
        optimizers[0].step()
        optimizers[0].zero_grad()
        optimizers[1].step()
        optimizers[1].zero_grad()

        # step each scheduler
        for sch in schedulers:
            sch.step()

        return elbo
    
    def on_after_backward(self) -> None:
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is None:
                # this parameter didn’t get a gradient
                print(f"[UNUSED] {name}")

        #print("I dont think we are calling this function at all ")
        if self.flag:
            optimizers = self.optimizers()
            for group in optimizers[0].optimizer.param_groups:
                group["update_buffer"] = group["update_buffer"].to(self.device)
                group["update_buffer_views"] = [
                    group["update_buffer"].to(self.device) for tensor in group["update_buffer_views"]
                ]
            self.flag = False
        
        #optimizers = self.optimizers()
        #for group in optimizers[0].optimizer.param_groups:
        #    print("test", group["update_buffer"] == group["update_buffer_views"][0])
                
        self.ema.update_parameters(self.model)

        # Compute gradient norm

        if self.hparams.grad_clipping_type == "always":
            new_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
            #if new_grad_norm > self.grad_clip:
                #print("we should have clipped")

        elif self.hparams.grad_clipping_type == "warmup":
            if self.global_step > self.clip_warmup:
                new_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
                #if new_grad_norm > self.grad_clip:
                    #print("we should have clipped")
            else:
                new_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
        elif self.hparams.grad_clipping_type == "dynamic":
            alpha = 0.999

            if self.global_step == 0:
                self.current_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
                new_grad_norm=self.current_grad_norm
                self.log("current_clip", self.current_grad_norm*2, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            if self.global_step < 50:
                new_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
                self.log("current_clip", self.current_grad_norm*2, on_step=True, on_epoch=False, prog_bar=True, logger=True)
                self.current_grad_norm = alpha * self.current_grad_norm + (1 - alpha) * new_grad_norm
            else:
                new_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.current_grad_norm*1.5)
                if new_grad_norm > self.current_grad_norm*2:
                    print("we should have clipped")
                    new_grad_norm = self.current_grad_norm*2 #add some downward bias to the gradient norm so it cannot blow up over the course of a couple steps 
                self.log("current_clip", self.current_grad_norm*2, on_step=True, on_epoch=False, prog_bar=True, logger=True)
                self.current_grad_norm = alpha * self.current_grad_norm + (1 - alpha) * new_grad_norm
            
        self.log("grad_norm", new_grad_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        # anneal beta
        self.beta = float(min(1.0, self.global_step / float(self.beta_vae_anneal)))
        self.log("beta", self.beta, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.grad_clipping_type == "dynamic":
            # Save the current gradient norm to the checkpoint
            checkpoint['current_grad_norm'] = self.current_grad_norm

        
            

        if self.use_muon:
            #optimizer state should already be saved
            #print(self.optimizers()[0])
            #checkpoint["muon_param_groups"] = self.optimizers()[0].optimizer.param_groups
            pass
            

    def on_load_checkpoint(self, checkpoint):
        self.current_grad_norm = checkpoint.get('current_grad_norm', 0.22) #if its not there then its not used so None would also be fine
        
        """
        print("did this happen?")
        for opt_state in checkpoint["optimizer_states"]:
            print(opt_state)
            for state in opt_state["state"].values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        """
        if self.use_muon:
            self.flag = True
            #save optimizer state as txt file so I can look at it 
            #save a string as txt file by using 
            #create that file
            with open("optimizer_state.txt", "w") as f:
                f.write(str(checkpoint["optimizer_states"]))
            
            #print(self.optimizers())
            #print(checkpoint["optimizer_states"])
            self._manual_optim_state = checkpoint["optimizer_states"]
            #self._muon_param_groups = checkpoint["muon_param_groups"]
            #print(self._muon_param_groups)


    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        t = torch.rand(batch.size(0), 1).unsqueeze(2).to(batch.device) 
        diffusion_loss, context_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss = self.forward(t, batch,
                                            compute_diffusion_loss=self.hparams.compute_diffusion_loss,
                                            compute_prior_loss=self.hparams.compute_prior_loss,
                                            compute_reconstruction_loss=self.hparams.compute_reconstruction_loss,
                                            reconstruction_loss_type = self.hparams.reconstruction_loss_type)


        diffusion_loss = diffusion_loss.mean()
        reconstruction_loss = reconstruction_loss.mean()
        prior_loss = prior_loss.mean()

        #note elbo may or may not be valid depending on what we actualy calculatte in the forward pass
        elbo = diffusion_loss + reconstruction_loss + prior_loss

        #compute the bits per character 

        # update and log metrics
        self.log("val/diffusion_loss", diffusion_loss, on_step=True, prog_bar=False)
        self.log("val/reconstruction_loss", reconstruction_loss, on_step=True, prog_bar=True)
        self.log("val/prior_loss", prior_loss, on_step=True, prog_bar=False)
        self.log("val/elbo", elbo, on_step=True, prog_bar=True)


       
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Test‑time MC‑ELBO with unbiased variance + SE in **nats / dimension**.

        Notes
        -----
        * **No antithetic sampling** (plain MC).
        * `K` i.i.d. draws of the diffusion term per example.
        * Computes:
            - `mc_var_per_example` : Var[\bar{ℓ}_i]  (nats²)  ← variance of the MC **mean** per sample.
            - `dataset_se`         : SE of the reported dataset mean (nats / dim).
        """
        import math, torch

        B = batch.size(0)                         # batch size
        d = batch.shape[1]                        # sequence length (dimensions per example)
        K = 64                                    # MC samples per example
        device = batch.device

        elbo_model = self.ema.module              # use EMA weights for evaluation

        # -------------------------------------------------------------------
        # 1.   CONSTANT TERMS (reconstruction + prior) – no dependence on t
        # -------------------------------------------------------------------
        t_dummy = torch.zeros(B, 1, 1, device=device)  # placeholder only – functions ignore t
        recon_loss = elbo_model.get_elbo_reconstruction_loss(batch, t_dummy)   # (B,)
        prior_loss = elbo_model.get_elbo_prior_loss(batch, t_dummy)            # (B,)

        # -------------------------------------------------------------------
        # 2.   MONTE‑CARLO SAMPLES OF THE DIFFUSION TERM
        # -------------------------------------------------------------------
        elbo_samples = torch.empty(K, B, device=device)   # nats, per‑sample/per‑example
        diff_samples = torch.empty(K, B, device=device)

        for k in range(K):
            # --- plain U[0,1] sampling over t --------------------------------
            t = torch.rand(B, 1, 1, device=device)

            # diffusion + (optional) context loss; both are unreduced
            diff_loss, ctx_loss = elbo_model.get_elbo_diffusion_loss(batch, t)
            if ctx_loss is None:
                ctx_loss = torch.zeros_like(prior_loss)

            diff_per_ex  = diff_loss.flatten(1).sum(1)    # (B,)
            ctx_per_ex   = ctx_loss                       # already (B,)

            elbo_samples[k] = diff_per_ex + recon_loss + prior_loss + ctx_per_ex
            diff_samples[k] = diff_per_ex

        # -------------------------------------------------------------------
        # 3.   PER‑EXAMPLE MEAN & VARIANCE (unbiased) ------------------------
        # -------------------------------------------------------------------
        elbo_mean_per_ex = elbo_samples.mean(0)                       # (B,)
        mc_var_per_ex    = elbo_samples.var(0, unbiased=True) / K     # Var[ \bar{ℓ}_i ] (nats²)

        # -------------------------------------------------------------------
        # 4.   DATASET‑LEVEL STATISTICS --------------------------------------
        # -------------------------------------------------------------------
        dataset_mean_elbo = elbo_mean_per_ex.mean()                   # scalar (nats)
        dataset_var_across_ex = elbo_mean_per_ex.var(unbiased=True)   # Var_data[ℓ̄] (nats²)
        dataset_se = torch.sqrt(dataset_var_across_ex / B)            # SE of mean (nats)

        # Convert to nats per *dimension* (sequence element)
        nats_per_dim_mean = dataset_mean_elbo / d                     # scalar
        nats_per_dim_se   = dataset_se / d                            # scalar

        # Same for diffusion term (for monitoring)
        diff_mean_per_ex = diff_samples.mean(0)                       # (B,)
        dataset_mean_diff = diff_mean_per_ex.mean()

        # -------------------------------------------------------------------
        # 5.   LOGGING --------------------------------------------------------
        # -------------------------------------------------------------------
        self.log("test/diffusion_loss",         dataset_mean_diff,      on_step=False, prog_bar=False)
        self.log("test/reconstruction_loss",    recon_loss.mean(),      on_step=False, prog_bar=False)
        self.log("test/prior_loss",             prior_loss.mean(),      on_step=False, prog_bar=False)
        self.log("test/context_loss",           ctx_per_ex.mean(),      on_step=False, prog_bar=False)
        self.log("test/elbo",                   dataset_mean_elbo,      on_step=False, prog_bar=True)
        self.log("test/elbo_mc_var",           mc_var_per_ex.mean(),   on_step=False, prog_bar=False)
        self.log("test/elbo_dataset_se",        dataset_se,             on_step=False, prog_bar=False)
        self.log("test/nats_per_dim",           nats_per_dim_mean,      on_step=False, prog_bar=True)
        self.log("test/nats_per_dim_se",        nats_per_dim_se,        on_step=False, prog_bar=False)



    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        if self.hparams.enable_matmul_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        if self.hparams.enable_cudnn_tf32:
            torch.backends.cudnn.allow_tf32 = True
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        #torch.optim.Adam([*model.parameters(), *time_sampler.parameters()], lr=1e-4)

        if not self.use_muon:
            if isinstance(self.time_sampler, TimeSampler):
                optimizer = self.hparams.optimizer([*self.model.parameters(), *self.time_sampler.parameters()])
            else:
                optimizer = self.hparams.optimizer(self.model.parameters())

            def linear_anneal_lambda(step, total_steps):
                if self.do_lr_warmup:
                    warmup_steps = 10000
                    if step < warmup_steps:
                        return step / warmup_steps
                return 1 - (step / total_steps)

            total_steps = self.max_steps  # Replace with your total annealing steps
            if self.hparams.use_scheduler:
                scheduler = LambdaLR(optimizer, lr_lambda=lambda step: linear_anneal_lambda(step, total_steps))
                return_dict = {"optimizer": optimizer, "lr_scheduler": scheduler}
            else:
                return_dict = {"optimizer": optimizer}

            return return_dict
        else:
            muon_params = []
            adamw_params = []

            # this is very important since this is where we decide which parameters go to which optimizer 
            # from the repo: 
            # Muon is intended to optimize only the internal ≥2D parameters of a network. Embeddings, classifier heads, and scalar or vector parameters should be optimized using AdamW.
            # So I will only use Muon for parameters in self.model.pred.model and in affine.net is forward_process exists, if we use the gamma representation it should only work on the transfom parameters
            # so for pred it should only work on self.model.pred.model.input_transformers
            # and for the affine it should only work on self.model.affine.net.encoder

            pred_trans = self.model.pred.model.input_transformers
            for p in pred_trans.parameters():
                if not p.requires_grad:
                    continue
                if p.ndim >= 2:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

            # 2) Affine: only encoder if using forward_process
            affine_net = self.model.affine.net
            if hasattr(affine_net, "forward_process"):
                for p in affine_net.encoder.parameters():
                    if not p.requires_grad:
                        continue
                    if p.ndim >= 2:
                        muon_params.append(p)
                    else:
                        adamw_params.append(p)

            # 3) Everything else → AdamW
            #    Skip any p already in one of the two lists
            seen = {id(p) for p in muon_params + adamw_params}
            for p in self.model.parameters():
                if not p.requires_grad or id(p) in seen:
                    continue
                adamw_params.append(p)

            # 2) Instantiate optimizers
            optim_muon = MuonLightning(
                muon_params,
                lr=self.muon_params.muon_lr,
                momentum=self.muon_params.muon_momentum,
                ns_steps=self.muon_params.muon_ns_steps,
                weight_decay=self.muon_params.muon_weight_decay,
                rank=0,
                world_size=1
            )
            optim_adamw = self.hparams.optimizer([*adamw_params, *self.time_sampler.parameters()])

            # 3) Create a shared LR schedule with warmup + cosine decay
            def linear_anneal_lambda(step, total_steps):
                if self.do_lr_warmup:
                    warmup_steps = 10000
                    if step < warmup_steps:
                        return step / warmup_steps
                return 1 - (step / total_steps)

            total_steps = self.max_steps/2  # Replace with your total annealing steps
            scheduler_muon = LambdaLR(optim_muon, lr_lambda=lambda step: linear_anneal_lambda(step, total_steps))
            scheduler_adamw = LambdaLR(optim_adamw,  lr_lambda=lambda step: linear_anneal_lambda(step, total_steps))

            self._optimizers = [optim_muon, optim_adamw]

            if hasattr(self, "_manual_optim_state"):
                for opt, state in zip(self._optimizers, self._manual_optim_state):
                    opt.load_state_dict(state)

            
            optimizers = self._optimizers
            for group in optimizers[0].param_groups:
                group["update_buffer"] = group["update_buffer"].to(self.device)
                group["update_buffer_views"] = [group["update_buffer"][i] for i in range(1)]
                    
            #if hasattr(self, "_muon_param_groups"):
                #Im not 100% sure this is correct
            #    print(self._optimizers[0].param_groups)
            #    self._optimizers[0].param_groups = self._muon_param_groups
            
            return (
                self._optimizers,
                [
                    {"scheduler": scheduler_muon, "interval": "step"},
                    {"scheduler": scheduler_adamw, "interval": "step"},
                ],
            )

if __name__ == "__main__":
    _ = DiffusionModule(None, None, None, None)