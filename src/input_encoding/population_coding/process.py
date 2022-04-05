from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class PopulationCoding(AbstractProcess):
    """Abstract class for variables common to all neurons with leaky
    integrator dynamics."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_shape = kwargs.get("shape", (1,))
        n_neuron_per_dim = kwargs.get("n_neuron_per_dim", 2)

        self.a_in = InPort(shape=in_shape)
        self.s_out = OutPort(shape=in_shape+(n_neuron_per_dim,))
