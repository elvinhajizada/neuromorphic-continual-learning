import numpy as np
import unittest

from matplotlib import pyplot as plt


from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import RefPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.decorator import implements, requires
from lava.proc.monitor.process import Monitor
from lava.proc.lif.process import LIF
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.run_configs import Loihi1SimCfg
from input_encoding.population_coding.utils import gaussian


class PopulationCoding(unittest.TestCase):

    def test_gaussion(self):
        x = np.arange(0, 10, 0.1)
        mu = np.arange(0, 11)
        sig = 1.5

        for m in mu:
            a = gaussian(x, m, sig)
            plt.plot(x, a)

        plt.show()

    def test_get_activation_vals(self):
        in_shape = (1,)
        n_neuron_per_dim = 11
        vals = [0.5, 5, 9]
        x = np.arange(0, n_neuron_per_dim)
        mu = np.arange(0, n_neuron_per_dim)
        sig = 1.5
        neuron_shape = in_shape + (n_neuron_per_dim,)
        a_in = np.zeros(shape=neuron_shape)
        for ind, val in enumerate(vals):
            a = np.around(gaussian(val, mu, sig), 2)
            print(a)
            plt.plot(x, a)

        plt.show()


if __name__ == '__main__':
    unittest.main()