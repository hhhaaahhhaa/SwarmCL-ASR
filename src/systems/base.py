import lightning as pl


class System(pl.LightningModule):
    """ Abstract base class for all systems. """
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.log_dir = config["output_dir"]["log_dir"]
        self.result_dir = config["output_dir"]["result_dir"]
        self.ckpt_dir = config["output_dir"]["ckpt_dir"]

        self.build_configs()
        self.build_model()

    def build_configs(self):
        """ Parse additional information """
        pass

    def build_model(self):
        """ Build all components here. """
        pass

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # support loading to a different structure by matching names
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
        
        if is_changed:
            print("You try to load the checkpoint to a different architecture, make sure you know what you are doing.")

        # modify state_dict format to model_state_dict format
        for k in state_dict_pop_keys:
            state_dict.pop(k)
        for k in model_state_dict:
            if k not in state_dict:
                # print("Reinitialize: ", k)
                state_dict[k] = model_state_dict[k]

        if is_changed:
            checkpoint.pop("optimizer_states", None)
