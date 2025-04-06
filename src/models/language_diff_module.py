from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import inspect
from torch.optim.lr_scheduler import LambdaLR

import copy
from collections import OrderedDict
from their_utils.nn import mean_flat
from torch.optim.swa_utils import AveragedModel
from src.models.time_samplers.time_samplers import TimeSampler, UniformBucketSampler, ContSampler


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
        use_full_elbo_in_is: bool = True,
        compute_diffusion_loss: bool = True,
        compute_prior_loss: bool = False,
        compute_reconstruction_loss: bool = True,
        reconstruction_loss_type: str = "diff_anchor",
        mask_padding: bool = False,
        enable_matmul_tf32: bool = False,
        enable_cudnn_tf32: bool = False
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

        if use_full_elbo_in_is and not isinstance(time_sampler, TimeSampler):
            self.use_full_elbo_in_is = False
            print("use_full_elbo_in_is is set to False because time_sampler is not a TimeSampler")

        self.time_sampler = time_sampler
        self.model = diffusion
        self.max_steps = total_steps
        self.mask_padding = mask_padding
        self.grad_clip = grad_clip
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

    @torch.no_grad()
    def update_ema(self, ema_model, model, decay=0.9999):
        """
        Step the EMA model towards the current model.
        """
        device = next(model.parameters()).device
        model.to("cpu")
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())
        
        
        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

        model.to(device)
        

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
        
        diffusion_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss = self.model.get_losses(x, t, 
                                                                                compute_diffusion_loss, 
                                                                                compute_prior_loss, 
                                                                                compute_reconstruction_loss, 
                                                                                reconstruction_loss_type,
                                                                                compute_their_loss=False)   
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
            diffusion_loss = mean_flat(diffusion_loss)   
            diffusion_loss_full_elbo = mean_flat(diffusion_loss_full_elbo) if diffusion_loss_full_elbo is not None else None
        

        prior_loss = mean_flat(prior_loss)

        return diffusion_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass
    

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
            new_t = torch.arange(0, 1, 0.1).unsqueeze(-1).to(t.device)
            #print("probability dist over t", self.time_sampler.prob(new_t))        
        
        t = t.detach()

        #with torch.no_grad():
        diffusion_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss = self.forward(t, batch,
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
            elbo = is_loss + reconstruction_loss + prior_loss 
        else:
            elbo = diffusion_loss + reconstruction_loss + prior_loss
        
        diffusion_loss = diffusion_loss.mean()
        reconstruction_loss = reconstruction_loss.mean()
        prior_loss = prior_loss.mean()
        elbo = elbo.mean()
        #print(diffusion_loss, "----------------------diffusion_loss")

        #print(f"diffusion_loss: {self.hparams.compute_diffusion_loss}, reconstruction_loss: {self.hparams.compute_reconstruction_loss}, prior_loss: {self.hparams.compute_prior_loss}")

        #note elbo may or may not be valid depending on what we actualy calculatte in the forward pass
        

        # update and log metrics
        self.log("train/diffusion_loss", diffusion_loss, on_step=True, prog_bar=True, logger=True)
        self.log("train/reconstruction_loss", reconstruction_loss, on_step=True, prog_bar=True,logger=True)
        self.log("train/prior_loss", prior_loss, on_step=True, prog_bar=True,logger=True)
        self.log("train/elbo", elbo, on_step=True, prog_bar=False,logger=True)

        # return loss or backpropagation will fail
        return elbo
    
    def on_after_backward(self) -> None:
        self.ema.update_parameters(self.model)

        # Compute gradient norm

        if self.hparams.grad_clipping_type == "always":
            new_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
        elif self.hparams.grad_clipping_type == "warmup":
            if self.global_step > 3000:
                new_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
            else:
                new_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
        elif self.hparams.grad_clipping_type == "dynamic":
            alpha = 0.99

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

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.grad_clipping_type == "dynamic":
            # Save the current gradient norm to the checkpoint
            checkpoint['current_grad_norm'] = self.current_grad_norm

    def on_load_checkpoint(self, checkpoint):
        self.current_grad_norm = checkpoint.get('current_grad_norm', 0.22) #if its not there then its not used so None would also be fine
        

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        t = torch.rand(batch.size(0), 1).unsqueeze(2).to(batch.device) 
        diffusion_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss = self.forward(t, batch,
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
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        #this is because I did runs without this attribute and thus it would not checkpoint load correctly otherwise...
        t = torch.rand(batch.size(0), 1).unsqueeze(2).to(batch.device) 

        diffusion_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss  = self.forward(t,batch,
                                            compute_diffusion_loss=self.hparams.compute_diffusion_loss,
                                            compute_prior_loss=self.hparams.compute_prior_loss,
                                            compute_reconstruction_loss=self.hparams.compute_reconstruction_loss,
                                            reconstruction_loss_type = self.hparams.reconstruction_loss_type)

        diffusion_loss = diffusion_loss.mean()
        reconstruction_loss = reconstruction_loss.mean()
        prior_loss = prior_loss.mean()

        #note elbo may or may not be valid depending on what we actualy calculatte in the forward pass
        elbo = diffusion_loss + reconstruction_loss + prior_loss 
        print(diffusion_loss)

        # update and log metrics
        self.log("test/diffusion_loss", diffusion_loss, on_step=True, prog_bar=True)
        self.log("test/reconstruction_loss", reconstruction_loss, on_step=True, prog_bar=True)
        self.log("test/prior_loss", prior_loss, on_step=True, prog_bar=True)
        self.log("test/elbo", elbo, on_step=True, prog_bar=True)
        self.log("test/nfe", nfe, on_step=True, prog_bar=True)

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

        if isinstance(self.time_sampler, TimeSampler):
            optimizer = self.hparams.optimizer([*self.model.parameters(), *self.time_sampler.parameters()])
        else:
            optimizer = self.hparams.optimizer(self.model.parameters())

        def linear_anneal_lambda(step, total_steps):
            return 1 - (step / total_steps)

        total_steps = self.max_steps  # Replace with your total annealing steps
        if self.hparams.use_scheduler:
            scheduler = LambdaLR(optimizer, lr_lambda=lambda step: linear_anneal_lambda(step, total_steps))
            return_dict = {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return_dict = {"optimizer": optimizer}

        return return_dict

if __name__ == "__main__":
    _ = DiffusionModule(None, None, None, None)