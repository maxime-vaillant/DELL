import torch


class PeakMemoryMixin:
    """Logs peak allocated and reserved CUDA memory per stage.

    Lightning runs validation inside the training epoch, so `on_validation_epoch_start` resets the
    CUDA peak counter before `on_train_epoch_end` can read it. Subclasses therefore call
    `_log_peak_memory('train')` from `on_validation_epoch_start` before resetting, and from
    `on_train_epoch_end` only when validation did not run.
    """

    def _reset_peak_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _log_peak_memory(self, stage: str) -> None:
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        else:
            allocated_mb = reserved_mb = 0.0
        self.log(f'{stage}/peak_memory_allocated_mb', allocated_mb, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{stage}/peak_memory_reserved_mb', reserved_mb, on_step=False, on_epoch=True, sync_dist=True)
