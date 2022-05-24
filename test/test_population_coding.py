import unittest

import numpy as np
from matplotlib import pyplot as plt
from input_encoding.population_coding.utils import gaussian, \
    gen_population_coding, gen_prototypes_from_data
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from som import SOM

from typing import List, Tuple
from lava.proc.monitor.process import Monitor
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.io.dataloader import StateDataloader
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.lib.dnf.utils.plotting import raster_plot


class TestRunConfig(RunConfig):
    """Run configuration selects appropriate ProcessModel based on tag:
    floating point precision or Loihi bit-accurate fixed point precision"""
    def __init__(self, select_tag: str = 'fixed_pt') -> None:
        super().__init__(custom_sync_domains=None)
        self.select_tag = select_tag

    def select(
        self,
        _: List[AbstractProcessModel],
        proc_models: List[PyLoihiProcessModel]
    ) -> PyLoihiProcessModel:
        # print(proc_models)
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError('No legal ProcessModel found.')


class PopulationCodedDataset:
    def __init__(self, data: np.ndarray, shape: tuple,) -> None:
        self.shape = shape
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, id: int) -> Tuple[np.ndarray, int]:
        data_item = self.data[id,...]
        label = id
        return data_item, label


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

    def test_input_injection(self):
        # Load Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        n_neuron_per_dim = 10
        sigma_ratio = 0.07
        x_min, x_max = np.min(X, 0), np.max(X, 0)

        # Create population coding of the Iris data
        u_data = np.zeros(shape=X.shape + (n_neuron_per_dim,))
        for i in range(X.shape[0]):
            x = X[i, :]
            u_data[i, :, :] = gen_population_coding(x, x_min, x_max,
                                                    n_neuron_per_dim,
                                                    sigma_ratio)

        u_data_flat = u_data.reshape(
                (u_data.shape[0], u_data.shape[1] * u_data.shape[2]))
        u_data = 100 * u_data
        print("\n", u_data[[0, 1, 2], :, :])

        iris_u_dataset = PopulationCodedDataset(data=u_data, shape=u_data.shape)
        u_data_loader = StateDataloader(dataset=iris_u_dataset, interval=100)

        num_steps = 100

        neurons = LIF(shape=(u_data.shape[-2], u_data.shape[-1]),
                      vth=120,
                      du=0.05,
                      dv=0.1)
        monitor = Monitor()

        monitor.probe(target=neurons.v, num_steps=num_steps)
        monitor.probe(target=neurons.s_out, num_steps=num_steps)
        u_data_loader.connect_var(neurons.u)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='floating_pt')
        neurons.run(condition=run_condition, run_cfg=run_config)
        data = monitor.get_data()
        neurons.stop()
        # print("probed u:\n", data[neurons.name][neurons.v.name])
        spike_data = data[neurons.name][neurons.s_out.name]
        print(spike_data.shape)
        spike_data = np.reshape(spike_data,
                                (spike_data.shape[0],
                                 spike_data.shape[1]*spike_data.shape[2]))
        plt.figure()
        raster_plot(spike_data)
        plt.show()

    def test_prototype_neuron(self):
        # Load Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        n_neuron_per_dim = 10
        sigma_ratio = 0.07
        x_min, x_max = np.min(X, 0), np.max(X, 0)

        # Create population coding of the Iris data
        u_data = np.zeros(shape=X.shape + (n_neuron_per_dim,))
        for i in range(X.shape[0]):
            x = X[i, :]
            u_data[i, :, :] = gen_population_coding(x, x_min, x_max,
                                                    n_neuron_per_dim,
                                                    sigma_ratio)

        u_data_flat = u_data.reshape(
                (u_data.shape[0], u_data.shape[1] * u_data.shape[2]))
        u_data_flat = 100 * u_data_flat
        print("\n", u_data[[0, 1, 2], :, :])

        interval = 100
        n_sample = 9
        small_dataset = u_data_flat[np.r_[0:3, 51:54, 101:104], :]

        iris_u_dataset = PopulationCodedDataset(data=small_dataset,
                                                shape=small_dataset.shape)

        u_data_loader = StateDataloader(dataset=iris_u_dataset,
                                        interval=interval)

        in_pop_shape = u_data_flat.shape[-1]
        n_protos = 3

        prototype_weights = gen_prototypes_from_data(n_protos=n_protos,
                                                     data=u_data_flat)
        learnt_protos = u_data_flat[[0, 52, 102], :]
        prototype_weights = learnt_protos
        num_steps = interval * n_sample

        conn_shape = (n_protos, in_pop_shape)

        input_neurons = LIF(shape=(in_pop_shape,),
                            vth=120,
                            du=0.05,
                            dv=0.1)

        conn = Dense(shape=conn_shape, weights=prototype_weights)

        proto_neurons = LIF(shape=(n_protos,),
                            vth=4000,
                            du=0.05,
                            dv=0.1)
        monitor = Monitor()

        input_neurons.s_out.connect(conn.s_in)
        conn.a_out.connect(proto_neurons.a_in)

        monitor.probe(target=input_neurons.v, num_steps=num_steps)
        monitor.probe(target=input_neurons.s_out, num_steps=num_steps)
        monitor.probe(target=proto_neurons.s_out, num_steps=num_steps)

        u_data_loader.connect_var(input_neurons.u)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='floating_pt')
        input_neurons.run(condition=run_condition, run_cfg=run_config)
        data = monitor.get_data()
        input_neurons.stop()
        # print("probed u:\n", data[neurons.name][neurons.v.name])
        in_spike_data = data[input_neurons.name][input_neurons.s_out.name]
        out_spike_data = data[proto_neurons.name][proto_neurons.s_out.name]
        plt.figure()
        raster_plot(in_spike_data)
        plt.figure()
        raster_plot(out_spike_data)
        plt.show()


if __name__ == '__main__':
    unittest.main()