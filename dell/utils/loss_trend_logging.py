class LossTrendMixin:
    """Logs `{stage}/{key}_trend`, the relative loss drop over the last `loss_trend_window` epochs.

    A value near 0 means the loss has plateaued. Subclasses must set `loss_trend_window` and
    `_loss_history = {}` in `__init__` and call `_log_loss_trends(stage, keys)` from
    `on_train_epoch_end`, after the losses were logged with `on_epoch=True`.
    """

    def _log_loss_trends(self, stage: str, keys: list[str]) -> None:
        for key in keys:
            # train and val share keys, so the history is keyed by the full 'stage/key' name
            metric_name = f'{stage}/{key}'
            metric = self.trainer.callback_metrics.get(metric_name)
            if metric is None:
                continue

            history = self._loss_history.setdefault(metric_name, [])
            history.append(metric.item())

            if len(history) <= self.loss_trend_window:
                continue

            earlier, latest = history[-self.loss_trend_window - 1], history[-1]
            trend = (earlier - latest) / (earlier + 1e-8)
            self.log(f'{metric_name}_trend', trend, on_step=False, on_epoch=True, sync_dist=True)
