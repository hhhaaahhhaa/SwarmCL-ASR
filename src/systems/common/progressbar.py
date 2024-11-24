import sys
from tqdm.auto import tqdm
from lightning.pytorch.callbacks import Callback


class GlobalProgressBar(Callback):
    def __init__(self, global_progress: bool = True, leave_global_progress: bool = True, process_position=0):
        super().__init__()

        self.global_progress = global_progress
        self.global_desc = "Steps: {steps}/{max_steps}"
        self.leave_global_progress = leave_global_progress
        self.global_pb = None
        self.process_position = process_position
        self.cur = -1

    def on_train_start(self, trainer, pl_module):
        if pl_module.local_rank == 0:
            if trainer.max_steps > 0:
                desc = self.global_desc.format(steps=pl_module.global_step + 1, max_steps=trainer.max_steps)
            else:
                desc=""

            self.global_pb = tqdm(
                desc=desc,
                dynamic_ncols=True,
                total=trainer.max_steps,
                initial=pl_module.global_step,
                leave=self.leave_global_progress,
                disable=not self.global_progress,
                position=self.process_position,
                file=sys.stdout,
            )

    def on_train_end(self, trainer, pl_module):
        if pl_module.local_rank == 0:
            self.global_pb.close()
            self.global_pb = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.local_rank == 0:

            # Set description
            if trainer.max_steps > 0:
                desc = self.global_desc.format(steps=pl_module.global_step + 1, max_steps=trainer.max_steps)
            else:
                desc=""
            self.global_pb.set_description(desc)

            # Update progress
            if pl_module.global_step + 1 != self.cur:
                self.global_pb.update(1)
                self.cur = pl_module.global_step + 1
