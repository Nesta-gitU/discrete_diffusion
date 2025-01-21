
class DiffusionImageSampleLogger(Callback):
    def __init__(
        self,
        odeint_params: Dict[str, Any],
        n_steps: int = 300,
        batch_size: int = 25,
        log_train: bool = True,
        log_validation: bool = True,
        log_train_every_n_epochs: Optional[int] = None,
        sample_seed: Optional[int] = 42
    ):
        self._odeint_params = odeint_params
        self._n_steps = n_steps

        self._train_val_batch_size_counter = TrainValBatchSizeCounter(
            default_batch_size=batch_size,
            collect_train=log_train,
            collect_validation=log_validation
        )

        self._schedule = Schedule(
            act_on_train=log_train,
            act_on_validation=log_validation,
            act_on_train_every_n_epochs=log_train_every_n_epochs
        )

        self._sample_seed = sample_seed

    @torch.no_grad()
    def sample_from_diffusion(
        self,
        module: LitDiffusion,
        bs: Optional[int]
    ) -> Tuple[Tensor, Tensor]:
        with model_in_eval_mode(module):
            with preserve_rng_state(seed=self._sample_seed):
                x_sde = module.diff.r_rsde.sample(
                    bs=bs, t=0., mode='sde', ts=1.,
                    n_steps=self._n_steps
                ).cpu()
                x_ode = module.diff.r_rsde.sample(
                    bs=bs, t=0., mode='ode', ts=1.,
                    odeint_params=self._odeint_params
                ).cpu()

        return x_sde, x_ode

    @rank_zero_only
    def setup(self, trainer: Trainer, module: LitDiffusion, stage: Optional[str] = None) -> None:
        self._schedule.setup_act_on_train_every_n_epochs(trainer)

    @rank_zero_only
    def on_train_batch_start(self, trainer: Trainer, module: LitDiffusion, batch: Tensor, batch_idx: int) -> None:
        self._train_val_batch_size_counter.update_train_batch_size(batch)

    @rank_zero_only
    def on_validation_batch_start(self, trainer: Trainer, module: LitDiffusion,
                                  batch: Tensor, batch_idx: int, dataloader_idx: int) -> None:
        self._train_val_batch_size_counter.update_validation_batch_size(batch)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, module: LitDiffusion) -> None:
        
        if not self._schedule.should_act_on_train(trainer):
            return

        logger = get_wandb_logger(trainer)
        if logger is None:
            return

        batch_size = self._train_val_batch_size_counter.train_batch_size
        img_sde, img_ode = self.sample_from_diffusion(module, batch_size)

        text_table = wandb.Table(columns=["epoch", "text_ode", "text_sde"])
        #hacky way to turn this into word logging would be take the image, argmax the first dimension, then convert using the meta file back to words from a list of indexes. 
        def img_to_word(img):
            tensor = img.squeeze(1)  # Shape becomes [25, 28, 28]
            tensor = tensor[:, :27, :]

            word_indices = tensor.argmax(dim=1)  # Shape becomes [25, 28]
            print(word_indices.shape, "word_indices.shape")

            with open("/teamspace/studios/this_studio/nesta_ndm_clone/data/text8/meta.pkl", 'rb') as f:
                meta = pickle.load(f)

            itos = meta['itos']

            # Step 3: Convert word indices to text using the vocabulary mapping
            decoded_texts = []
            for sequence in word_indices:
                words = [itos[idx.item()] for idx in sequence]
                decoded_texts.append("".join(words))  # Combine words into a single string

            return decoded_texts

        w_sde = img_to_word(img_sde)
        w_ode = img_to_word(img_ode)

        epoch = trainer.current_epoch

        text_table.add_data(epoch, w_ode, w_sde) 

        logger.experiment.log({"generated_samples": text_table})

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, module: LitDiffusion) -> None:
        if not self._schedule.should_act_on_validation():
            return

        logger = get_wandb_logger(trainer)
        if logger is None:
            return

        batch_size = self._train_val_batch_size_counter.validation_batch_size
        img_sde, img_ode = self.sample_from_diffusion(module, batch_size)
        log_images(logger, 'validation_images/samples_sde', img_sde)
        log_images(logger, 'validation_images/samples_ode', img_ode)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'train_val_batch_size_counter': self._train_val_batch_size_counter,
            'schedule': self._schedule
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._train_val_batch_size_counter.load_state_dict(state_dict['train_val_batch_size_counter'])
        self._schedule = state_dict['schedule']