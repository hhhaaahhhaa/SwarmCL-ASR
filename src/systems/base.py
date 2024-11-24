import lightning as pl


class System(pl.LightningModule):
    """ Abstract base class for all systems. """
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.build_configs(*args, **kwargs)

        self.log_dir = config["output_dir"]["log_dir"]
        self.result_dir = config["output_dir"]["result_dir"]
        self.ckpt_dir = config["output_dir"]["ckpt_dir"]
        self.save_hyperparameters()

        self.build_model(*args, **kwargs)

    def build_configs(self, *args, **kwargs):
        """ Parse additional information """
        pass

    def build_model(self, *args, **kwargs):
        """ Build all components here. """
        pass

    def build_saver(self):
        """ Return a list of savers(callbacks). """
        return []

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
