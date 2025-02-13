from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import inspect

from src.models.diffusions import NeuralDiffusion
import copy
from collections import OrderedDict

from src.likelihoods.compute_nll import get_likelihood_fn


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
        diffusion: NeuralDiffusion,
        optimizer: torch.optim.Optimizer,
        compile: bool,
        scheduler: torch.optim.lr_scheduler = None,
        compute_diffusion_loss: bool = True,
        compute_prior_loss: bool = False,
        compute_reconstruction_loss: bool = True,
        reconstruction_loss_type: str = "diff_anchor",
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

        self.model = diffusion
        self.ema = copy.deepcopy(self.model)
        for p in self.ema.parameters():
            p.requires_grad = False

        self.update_ema(self.ema, self.model, decay=0) 
        self.ema.eval()

        #initialize a mean metric for tracking nll over the entire dataset
        #not that in the future I should add the option to iterate over the test set multiple times to the eval script. 
        #I suppose a hacky easy way to do this would be to just make the len larger and than use module operator on the index in the get_item


        # initialize the metrics to track:
        # recon_loss, diffusion_loss, prior_loss, elbo, bits-per-character 

    @torch.no_grad()
    def update_ema(self, ema_model, model, decay=0.9999):
        """
        Step the EMA model towards the current model.
        """
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def forward(self, x: torch.Tensor, compute_diffusion_loss, compute_reconstruction_loss, compute_prior_loss, reconstruction_loss_type) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        t = torch.rand(x.size(0), 1).unsqueeze(2).to(x.device) #sample a random time for each example in the batch
        #fix t to 0.5 for debug
        #t = torch.ones(x.size(0), 1).unsqueeze(2).to(x.device)/100
        t.requires_grad = True


        #print(t, "t ")
        #print(t.requires_grad, "t requires grad")
        #for i in range(3):
        #    print(f"t: {t[i]}")
        # sample just t = 1 the most corupted noise level
        #t = torch.ones(x.size(0), 1).unsqueeze(2).to(x.device)
        diffusion_loss, reconstruction_loss, prior_loss = self.model(x, t, compute_diffusion_loss, compute_prior_loss, compute_reconstruction_loss, reconstruction_loss_type)
        return diffusion_loss, reconstruction_loss, prior_loss

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
        #with torch.no_grad():
        diffusion_loss, reconstruction_loss, prior_loss = self.forward(batch,
                                            compute_diffusion_loss=self.hparams.compute_diffusion_loss,
                                            compute_prior_loss=self.hparams.compute_prior_loss,
                                            compute_reconstruction_loss=self.hparams.compute_reconstruction_loss,
                                            reconstruction_loss_type = self.hparams.reconstruction_loss_type)

        diffusion_loss = diffusion_loss.mean()
        #print(reconstruction_loss, "----------------------reconstruction_loss")
        reconstruction_loss = reconstruction_loss.mean()
        prior_loss = prior_loss.mean()
        #print(diffusion_loss, "----------------------diffusion_loss")

        #print(f"diffusion_loss: {self.hparams.compute_diffusion_loss}, reconstruction_loss: {self.hparams.compute_reconstruction_loss}, prior_loss: {self.hparams.compute_prior_loss}")

        #note elbo may or may not be valid depending on what we actualy calculatte in the forward pass
        elbo = diffusion_loss + reconstruction_loss + prior_loss 

        # update and log metrics
        self.log("train/diffusion_loss", diffusion_loss, on_step=True, prog_bar=True)
        self.log("train/reconstruction_loss", reconstruction_loss, on_step=True, prog_bar=True)
        self.log("train/prior_loss", prior_loss, on_step=True, prog_bar=False)
        self.log("train/elbo", elbo, on_step=True, prog_bar=False)

        # return loss or backpropagation will fail
        return elbo
    
    def on_after_backward(self) -> None:
        self.update_ema(self.ema, self.model, decay=0.999)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        diffusion_loss, reconstruction_loss, prior_loss = self.forward(batch,
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
        if batch_idx == 0:
            print("batch_idx == 0")
            self.loglikelihood = MeanMetric()
            self.likelihood_fn = get_likelihood_fn(self.ema)

        bpd, z, nfe = self.likelihood_fn(batch)        
        bpc = bpd * self.embedding.weight.shape[1]
        print(self.embedding.weight.shape[1])
        self.loglikelihood.update(bpc)

        diffusion_loss, reconstruction_loss, prior_loss  = self.forward(batch,
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
        self.log("test/likelihood", self.loglikelihood.compute(), on_step=True, prog_bar=True)
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

        optimizer = self.hparams.optimizer(self.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
