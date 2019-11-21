import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as norm
from scipy.special import erf
import GPy
import abc

from .nn_kernel import NNKernel
from .nn_kernel_linear import NNKernelLinear
from .nn_kernel_abs import NNKernelAbs

class NNKernelRelu(NNKernel):
    __metaclass__ = abc.ABCMeta
    def __init__(self, input_dim, variance_w = 1., mean_w = 0, 
            variance_b = 0, mean_b = 0, L=1, name='nn_relu',
            standard_first_layer= False):
        """
        Equivalent kernel of a neural network with a single hidden layer and
        a ReLU activation function.
        
        Args:
            input_dim (int): Dimensionality of input.
            variance_w (float): Variance of the weights.
            mean_w (float): Mean of the weights.
            variance_b (float): Variance of the biases.
            mean_b (float): Mean of the biases.
            L (int): The number of hidden layers.

        """
        super().__init__(input_dim, variance_w, mean_w, variance_b, mean_b, L,
                name, standard_first_layer)

        self.kernel_lin = NNKernelLinear(input_dim)
        self.kernel_abs = NNKernelAbs(input_dim)

    def _single_layer_K(self, x1norm, x2norm, x1sum, x2sum, cos_theta, 
            x2_none=False):
        """
        Kernel for a single layer
        """
        mu_tilde_1 = np.divide(x1sum, x1norm, out=np.zeros_like(x1sum),
                where=x1norm!=0)
        mu_tilde_2 = np.divide(x2sum, x2norm, out=np.zeros_like(x2sum),
                where=x2norm!=0)

        pdf1 = norm.pdf(-mu_tilde_1)
        pdf2 = norm.pdf(-mu_tilde_2)
        cdf1 = norm.cdf(-mu_tilde_1)
        cdf2 = norm.cdf(-mu_tilde_2)
        erf1 = erf(mu_tilde_1/np.sqrt(2))
        erf2 = erf(mu_tilde_2/np.sqrt(2))

        # cross1 is expected value of X_1 |X_2|
        term1 = cos_theta*((1+mu_tilde_2**2)*(1-2*cdf2)+2*mu_tilde_2*pdf2)
        term2 = (mu_tilde_2*erf2+2*pdf2)*(mu_tilde_1-cos_theta*mu_tilde_2)
        cross1 = (term1 + term2)*x1norm*x2norm
        # cross2 is expected value of X_2 |X_1|
        term1 = cos_theta*((1+mu_tilde_1**2)*(1-2*cdf1)+2*mu_tilde_1*pdf1)
        term2 = (mu_tilde_1*erf1+2*pdf1)*(mu_tilde_2-cos_theta*mu_tilde_1)
        cross2 = (term1 + term2)*x1norm*x2norm

        k_abs = self.kernel_abs._single_layer_K(x1norm, x2norm, x1sum, x2sum, 
                cos_theta, x2_none)
        k_lin = self.kernel_lin._single_layer_K(x1norm, x2norm, x1sum, x2sum, 
                cos_theta, x2_none)

        return (k_abs+k_lin+cross1+cross2)/4.

    def _single_layer_M(self, x1norm, x1sum):
        return \
        (self.kernel_lin._single_layer_M(x1norm,x1sum)+\
         self.kernel_abs._single_layer_M(x1norm,x1sum))\
         /2.



