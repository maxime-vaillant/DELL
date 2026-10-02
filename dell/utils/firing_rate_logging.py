from lightning_fabric.utilities.exceptions import MisconfigurationException


class FiringRateMixin:
    """Logs firing rates for a module with a `unet` or `encoder`.

    Forward hooks on every neuron log `{stage}/{name}_firing_rate`, the mean neuron output.
    `_log_overall_firing_rate(stage)` logs `{stage}/firing_rate`, total spikes over total
    neuron-timesteps across all hooked layers.

    Subclasses call `_register_firing_rate_hooks()` once in `__init__`, after `unet` or `encoder`
    exists and before building any training-only heads. They set `self._current_stage` at the
    start of `_shared_step`, then call `_log_overall_firing_rate` at its end.
    """

    def _register_firing_rate_hooks(self) -> None:
        if hasattr(self, 'unet'):
            named_neurons = [(f'enc_layer_{i}', block.act) for i, block in enumerate(self.unet.encoder)]
            named_neurons += [
                ('bottleneck_act1', self.unet.bottleneck.act1),
                ('bottleneck_act2', self.unet.bottleneck.act2),
            ]
            named_neurons += [(f'dec_layer_{i}', block.fusion_act) for i, block in enumerate(self.unet.decoder)]
        else:
            named_neurons = [(f'enc_layer_{i}', block.act) for i, block in enumerate(self.encoder)]

        self._firing_rate_totals = []
        for name, neuron in named_neurons:
            neuron.register_forward_hook(self._make_firing_rate_hook(name))

    def _make_firing_rate_hook(self, name: str):
        def hook(module, input, output):
            spikes = output.detach()
            # Modules with several validation dataloaders set `_current_metrics_stage` so that each
            # dataloader logs under its own name; Lightning rejects one name from two dataloaders.
            stage = getattr(self, '_current_metrics_stage', self._current_stage)
            try:
                self.log(
                    f'{stage}/{name}_firing_rate', spikes.mean(),
                    on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False,
                )
            except MisconfigurationException:
                # Forward passes outside a Lightning loop (e.g. predict()) have nothing to log to.
                return
            self._firing_rate_totals.append((spikes.sum(), spikes.numel()))
        return hook

    def _log_overall_firing_rate(self, stage: str, metrics_stage: str | None = None) -> None:
        """Log the network-wide firing rate under `metrics_stage` (default: `stage`).

        The metric name differs from `stage` for modules with several validation dataloaders.
        """
        metrics_stage = metrics_stage or stage
        total_spikes = sum(total for total, _ in self._firing_rate_totals)
        total_neuron_steps = sum(count for _, count in self._firing_rate_totals)
        self.log(
            f'{metrics_stage}/firing_rate', total_spikes / total_neuron_steps,
            on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False,
        )
        self._firing_rate_totals = []
