import numpy as np
from typing import Tuple, Optional, Union, List
import warnings
import matplotlib.pyplot as plt


class Kernel:
    """An class for kernels"""
    def __init__(self, weights: np.ndarray, padding_value: float = 0.0):
        """Constructor of the Kernel class."""
        self.weights = self._validate_weights(weights)
        self.padding_value = padding_value

    def _validate_weights(self, weights):
        assert isinstance(weights, np.ndarray), "<weights> must be of type " \
                                                 "numpy.ndarray"
        return weights

    def plot(self):
        """Plots the weights of the kernel."""
        if self.weights is not None:
            if self.weights.ndim < 3:
                if self.weights.ndim == 2:
                    plt.imshow(self.weights, cmap='gray')
                    plt.xlabel(r'$\Delta$ x')
                    plt.ylabel(r'$\Delta$ y')
                    plt.colorbar()
                    plt.show()
                elif self.weights.ndim < 2:
                    plt.plot(self.weights)
                    plt.xlabel(r'$\Delta x')
                    plt.show()
            else:
                raise NotImplementedError("not yet implemented for "
                                          "dimensionality > 2")
        else:
            warnings.warn("Weights of the kernel have not been computed yet. "
                          "Create a connection that uses the kernel and only "
                          "plot the weights afterward.")


class GaborKernel(Kernel):
    """
    A kernel that generates Gabor filters with given params.
    https://en.wikipedia.org/wiki/Gabor_filter
    :param width: (Optional) Width of the square-shaped Gabor kernel
    :type width: int
    :param sigma: the sigma/standard deviation of the Gaussian envelope
    along x-axis
    :type sigma: float, int
    :param theta: Orientation of the Gabor kernel
    :type theta: float, int
    :param lamda: the wavelength of the sinusoidal factor
    :type lamda: float, int
    :param gamma: spatial aspect ratio, and specifies the ellipticity of
    the support of the Gabor kernel
    :type gamma: float, int
    :param psi: the phase offset
    :type psi: float, int
    :param n_stds: The linear size of the kernel is n_stds (3 by default)
    standard deviations
    :type n_stds: float, int
    :param amplitude:
    :type amplitude: float, int
    """
    def __init__(self,
                 width: Optional[int] = None,
                 sigma: Union[float, int] = 1,
                 theta: Union[float, int] = 0,
                 lamda: Union[float, int] = 1,
                 gamma: Union[float, int] = 1,
                 psi: Union[float, int] = 0,
                 n_stds: Union[float, int] = None,
                 amplitude: Union[float, int] = 1):

        self.n_stds_default_value = 3

        # If n_stds is specified by user (hence different than default value
        # and width is also specified raise ValueError, as both cannot be
        # specified at the same time)
        if n_stds is not None and width is not None:
            raise ValueError("Width and n_stds cannot be set at the same time.")

        # If n_stds is not specified by user then initialize n_stds to
        # n_stds_default_value
        if n_stds is None:
            n_stds = self.n_stds_default_value

        # Validate the type and ranges of the inputs
        self.sigma, self.lamda, self.theta, self.gamma, self.gamma, \
            self.n_stds, self.amplitude, self.width = self._validate_input(
                sigma, lamda, theta, psi, gamma, n_stds, amplitude, width)

        self.width = width
        self.sigma = sigma
        self.gamma = gamma
        self.lamda = lamda
        self.psi = psi
        self.theta = theta
        self.amplitude = amplitude
        self.n_stds = n_stds

        # Compute weigths based on specified kernel params
        weights = self.compute_weights()
        super().__init__(weights=weights, padding_value=0.0)

    @staticmethod
    def _validate_input(sigma, lamda, theta, psi, gamma, n_stds, amplitude,
                        width):
        """Validate user input"""

        if not isinstance(sigma, (float, int)):
            raise TypeError("sigma should be float or int")
        if not isinstance(lamda, (float, int)):
            raise TypeError("lamda should be float or int")
        if not isinstance(psi, (float, int)):
            raise TypeError("psi should be float or int")
        if not isinstance(theta, (float, int)):
            raise TypeError("theta should be float or int")
        if not isinstance(amplitude, (float, int)):
            raise TypeError("amplitude should be float or int")
        if not isinstance(n_stds, (float, int)):
            raise TypeError("n_stds should be float or int")

        if width is not None:
            if not isinstance(width, int):
                raise TypeError("width should be int")
            if width <= 0:
                raise ValueError("width should be positive")

        if sigma <= 0:
            raise ValueError("sigma should be positive")
        if gamma <= 0:
            raise ValueError("gamma should be positive")
        if lamda <= 0:
            raise ValueError("lamda should be positive")
        if n_stds <= 0:
            raise ValueError("n_stds should be positive")
        if amplitude == 0:
            raise ValueError("amplitude should be non-zero")

        return sigma, lamda, theta, psi, gamma, n_stds, amplitude, width

    def compute_weights(self):
        """Compute the weights for the Gabor filter instace. We define a
        meshgrid of the kernel, that is always square grid, i.e. width=length"""

        # Calculate sigma y axis based on gamma(aspect ratio) and sigma,
        # which is specified by user as sigma_x (sigma along x axis)
        sigma_x = self.sigma
        sigma_y = float(self.sigma) / self.gamma

        # If width is not specified by user create square meshgrid based on
        # sigma & gamma (i.e. sigma_x and sigma_y) and n_stds (number of
        # standard deviation user wants to have size)
        if self.width is None:
            xmax = np.ceil(max(abs(self.n_stds * sigma_x * np.cos(self.theta)),
                               abs(self.n_stds * sigma_y * np.sin(self.theta)),
                               1))
            ymax = np.ceil(max(abs(self.n_stds * sigma_x * np.sin(self.theta)),
                               abs(self.n_stds * sigma_y * np.cos(self.theta)),
                               1))
            xmax = max(xmax, ymax)
            ymax = xmax
            xmin = -xmax
            ymin = -ymax
            [x, y] = np.meshgrid(np.arange(xmin, xmax + 1),
                                 np.arange(ymin, ymax + 1))

        # If width is specified by user, then create square grid with the
        # shape of (width,width)
        else:
            [xmin, xmax] = (np.ceil(-self.width / 2).astype('int'),
                            np.ceil(self.width / 2).astype('int'))
            ymin = xmin
            ymax = xmax
            [x, y] = np.meshgrid(range(xmin, xmax), range(ymin, ymax))

        # Rotation by theta, to have orinetation of Gabor kernel
        x_theta = x * np.cos(self.theta) + y * np.sin(self.theta)
        y_theta = -x * np.sin(self.theta) + y * np.cos(self.theta)

        # Gabor kernel as multiplication of sinusoidal wave and Gaussian
        # envelope. source: https://en.wikipedia.org/wiki/Gabor_filter
        gb = np.exp(
            -.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
            * np.cos(2 * np.pi / self.lamda * x_theta + self.psi)

        return self.amplitude * gb
