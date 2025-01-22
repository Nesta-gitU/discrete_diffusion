from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import inspect

from src.models.diffusions import NeuralDiffusion



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
        weight_decay: float,
        scheduler: torch.optim.lr_scheduler = None,
        compute_diffusion_loss: bool = True,
        compute_prior_loss: bool = False,
        compute_reconstruction_loss: bool = True,
        reconstruction_loss_type: str = "diff_anchor"
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

        # initialize the metrics to track:
        # recon_loss, diffusion_loss, prior_loss, elbo, bits-per-character 

    def forward(self, x: torch.Tensor, compute_diffusion_loss, compute_reconstruction_loss, compute_prior_loss, reconstruction_loss_type) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        t = torch.rand(x.size(0), 1).unsqueeze(2) #sample a random time for each example in the batch
        return self.model(x, t, compute_diffusion_loss, compute_reconstruction_loss, compute_prior_loss, reconstruction_loss_type)

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

        # update and log metrics
        self.log("train/diffusion_loss", diffusion_loss, on_step=True, prog_bar=False)
        self.log("train/reconstruction_loss", reconstruction_loss, on_step=True, prog_bar=False)
        self.log("train/prior_loss", prior_loss, on_step=True, prog_bar=False)
        self.log("train/elbo", elbo, on_step=True, prog_bar=True)

        # return loss or backpropagation will fail
        return elbo.sum()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        diffusion_loss, reconstruction_loss, prior_loss = self.forward(batch, 
                                            compute_diffusion_loss=True,
                                            compute_prior_loss=True,
                                            compute_reconstruction_loss=True,
                                            reconstruction_loss_type =self.hparams.reconstruction_loss_type)
        diffusion_loss = diffusion_loss.mean()
        reconstruction_loss = reconstruction_loss.mean()
        prior_loss = prior_loss.mean()

        #note elbo may or may not be valid depending on what we actualy calculatte in the forward pass
        elbo = diffusion_loss + reconstruction_loss + prior_loss 

        # update and log metrics
        self.log("val/diffusion_loss", diffusion_loss, on_step=True, prog_bar=False)
        self.log("val/reconstruction_loss", reconstruction_loss, on_step=True, prog_bar=False)
        self.log("val/prior_loss", prior_loss, on_step=True, prog_bar=False)
        self.log("val/elbo", elbo, on_step=True, prog_bar=True)
       

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        diffusion_loss, reconstruction_loss, prior_loss = self.forward(batch, 
                                            compute_diffusion_loss=True,
                                            compute_prior_loss=True,
                                            compute_reconstruction_loss=True,
                                            reconstruction_loss_type =self.hparams.reconstruction_loss_type)
        diffusion_loss = diffusion_loss.mean()
        reconstruction_loss = reconstruction_loss.mean()
        prior_loss = prior_loss.mean()

        #note elbo may or may not be valid depending on what we actualy calculatte in the forward pass
        elbo = diffusion_loss + reconstruction_loss + prior_loss 

        # update and log metrics
        self.log("test/diffusion_loss", diffusion_loss, on_step=True, prog_bar=False)
        self.log("test/reconstruction_loss", reconstruction_loss, on_step=True, prog_bar=False)
        self.log("test/prior_loss", prior_loss, on_step=True, prog_bar=False)
        self.log("test/elbo", elbo, on_step=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.trainer.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.hparams.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        print(f"using fused AdamW: {use_fused}")


        optimizer = self.hparams.optimizer(params=optim_groups)
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
