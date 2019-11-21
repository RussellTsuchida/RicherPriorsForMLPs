import numpy as np
import GPy
import abc

from .nn_kernel import NNKernel

class NNKernelLinear(NNKernel):
    __metaclass__ = abc.ABCMeta
    def __init__(self, input_dim, variance_w = 1., mean_w = 0, 
            variance_b = 0, mean_b = 0, L=1, name='nn_lin', 
            standard_first_layer = False):
        """
        Equivalent kernel of a neural network with a single hidden layer and
        a linear activation function.

        input_dim (int): Dimensionality of input.
        variance_w (float): Variance of the weights.
        mean_w (float): Mean of the weights.
        variance_b (float): Variance of the biases.
        mean_b (float): Mean of the biases.
        L (int): The number of hidden layers.
        """
        super().__init__(input_dim, variance_w, mean_w, variance_b, mean_b, L,
                name)

    def _single_layer_K(self, x1norm, x2norm, x1sum, x2sum, cos_theta, 
            x2_none=False):
        """
        Kernel for a single layer
        """
        return (x1norm*x2norm*cos_theta+x1sum*x2sum)

    def _single_layer_M(self, x1norm, x1sum):
        return x1sum
