import lightning as pl
from lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import Define
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .callbacks.progressbar import GlobalProgressBar


class System(pl.LightningModule):
    """ Abstract base class for all systems. """

    default_monitor: str = "val_loss"

    def __init__(
        self, data_configs, model_config, train_config, algorithm_config,
        log_dir, result_dir, ckpt_dir=None, *args, **kwargs
    ):
        super().__init__()
        self.data_configs = data_configs
        self.model_config = model_config
        self.train_config = train_config
        self.algorithm_config = algorithm_config
        self.build_configs(*args, **kwargs)  # Customize config hook

        self.log_dir = log_dir
        self.result_dir = result_dir
        self.ckpt_dir = ckpt_dir
        self.save_hyperparameters()

        self.build_model(*args, **kwargs)
        if Define.DEBUG:
            print("Model structure:")
            print(self)
            pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print("Total trainable params: ", pytorch_total_params)

            print("====================================================================")
            print("[Data Config]", self.data_configs)
            print("[Model Config]", self.model_config)
            print("[Train Config]", self.train_config)
            print("[Algorithm Config]", self.algorithm_config)
            print("====================================================================")

    def build_configs(self):
        """ Parser additional information """
        pass

    def build_model(self):
        """ Build all components here. """
        raise NotImplementedError
    
    def build_optimized_model(self):
        """ Return modules to be updated in training loop, if meta learning, consider only outer loop. """
        raise NotImplementedError    

    def build_saver(self):
        """ Return a saver class. """
        raise NotImplementedError
    
    def configure_callbacks(self):
        # Checkpoint saver
        save_step = self.train_config["step"]["save_step"]
        checkpoint = ModelCheckpoint(
            dirpath=self.ckpt_dir,
            monitor="Val/Total Loss", mode="min",
            every_n_train_steps=save_step, save_top_k=-1
        )

        # Progress bars (step/epoch)
        outer_bar = GlobalProgressBar(process_position=1)

        # Monitor learning rate / gpu stats
        lr_monitor = LearningRateMonitor()
        # gpu_monitor = GPUStatsMonitor(  stablize!
        #     memory_utilization=True, gpu_utilization=True, intra_step_time=True, inter_step_time=True
        # )
        
        # Save figures/audios/csvs
        saver = self.build_saver()
        if isinstance(saver, list):
            callbacks = [checkpoint, outer_bar, lr_monitor, *saver]
        else:
            callbacks = [checkpoint, outer_bar, lr_monitor, saver]
        return callbacks

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        optimized_modules = self.build_optimized_model()
        cnt = sum([p.numel() for p in optimized_modules.parameters() if p.requires_grad])
        print(f"Optimiable parameters: {cnt}")
        self.optimizer = get_optimizer(optimized_modules, self.model_config, self.train_config)

        self.scheduler = {
            "scheduler": get_scheduler(self.optimizer, self.train_config),
            'interval': 'step', # "epoch" or "step"
            'frequency': 1,
            'monitor': self.default_monitor,
        }

        return [self.optimizer], [self.scheduler]

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        return checkpoint

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.test_global_step = checkpoint["global_step"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        state_dict_pop_keys = []
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    if self.local_rank == 0:
                        print(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[k].shape}, "
                                    f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                # if self.local_rank == 0:
                #     print(f"Dropping parameter {k}")
                state_dict_pop_keys.append(k)
                is_changed = True

        # modify state_dict format to model_state_dict format
        for k in state_dict_pop_keys:
            state_dict.pop(k)
        for k in model_state_dict:
            if k not in state_dict:
                # print("Reinitialize: ", k)
                state_dict[k] = model_state_dict[k]

        if is_changed:
            checkpoint.pop("optimizer_states", None)


class BaseSystem(pl.LightningModule):
    """ Abstract base class for all systems. (v2) """
    def __init__(self, config):
        super().__init__()
        self._info = {}
        self.config = config
        self.build_configs()  # Customize config hook

        self.build_model()
        if Define.DEBUG:
            print("Model structure:")
            print(self)
            pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print("Total trainable params: ", pytorch_total_params)
        
        self.save_hyperparameters()

    def build_configs(self):
        """ Parser additional information """
        pass

    def build_model(self):
        """ Build all components here. """
        raise NotImplementedError
    
    def build_saver(self) -> list:
        """ Return a saver class. """
        return []
    
    def configure_callbacks(self):
        # Progress bars (step/epoch)
        outer_bar = GlobalProgressBar(process_position=1)

        # Monitor learning rate / gpu stats
        lr_monitor = LearningRateMonitor()

        callbacks = [outer_bar, lr_monitor, *self.build_saver()]

        return callbacks

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        return checkpoint

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.test_global_step = checkpoint["global_step"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        state_dict_pop_keys = []
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    if self.local_rank == 0:
                        print(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[k].shape}, "
                                    f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                # if self.local_rank == 0:
                #     print(f"Dropping parameter {k}")
                state_dict_pop_keys.append(k)
                is_changed = True

        # modify state_dict format to model_state_dict format
        for k in state_dict_pop_keys:
            state_dict.pop(k)
        for k in model_state_dict:
            if k not in state_dict:
                # print("Reinitialize: ", k)
                state_dict[k] = model_state_dict[k]

        if is_changed:
            checkpoint.pop("optimizer_states", None)
