import numpy as np


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gen_population_coding(x, min_x, max_x, n_neuron_per_dim, sigma):
    mu = np.linspace(min_x, max_x, n_neuron_per_dim)
    sig = sigma*(max_x-min_x)
    if isinstance(x, np.ndarray):
        x_shape = x.shape
    else:
        x_shape = (1,)
    a = np.zeros(shape=(x_shape+(n_neuron_per_dim,)))
    for ind, m in enumerate(mu):
        a[:, ind] = np.around(gaussian(x, m, sig), 2)
    return a


def gen_prototypes_from_data(n_protos, data):
    prototypes = np.around(np.random.normal(np.mean(data, axis=0),
                                            np.std(data, axis=0),
                                            size=(n_protos, len(data[0]))), 1)

    return prototypes
