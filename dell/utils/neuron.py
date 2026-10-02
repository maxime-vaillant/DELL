from spikingjelly.activation_based import neuron, surrogate

NEURON_KWARGS = {
    'tau': 3.2,
    'decay_input': False,
    'surrogate_function': surrogate.ATan(),
    'detach_reset': False,
    'backend': 'cupy',
}


def get_neuron() -> neuron.LIFNode:
    """The leaky integrate-and-fire neuron used throughout the network."""
    return neuron.LIFNode(**NEURON_KWARGS)
