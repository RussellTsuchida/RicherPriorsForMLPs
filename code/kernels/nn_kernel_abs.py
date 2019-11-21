import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as norm
from scipy.special import erf
import GPy
import abc

from .nn_kernel import NNKernel


class NNKernelAbs(NNKernel):
    __metaclass__ = abc.ABCMeta
    def __init__(self, input_dim, variance_w = 1., mean_w = 0, 
            variance_b = 0, mean_b = 0, L=1, name='nn_abs', 
            standard_first_layer = False):
        """
        Equivalent kernel of a neural network with a single hidden layer and
        an absolute value activation function.

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
            x2_none = False):
        """
        Kernel for a single layer
        """
        mu_tilde_1 = np.divide(x1sum, x1norm, out=np.zeros_like(x1sum), 
                where=x1norm!=0)
        mu_tilde_2 = np.divide(x2sum, x2norm, out=np.zeros_like(x2sum), 
                where=x2norm!=0)

        bvn_cdf, bvn_pdf = self._vectorised_bvn_cdf_pdf(mu_tilde_1, mu_tilde_2,
                cos_theta, vectorised=True, x2_none=x2_none)
    
        pdf1 = norm.pdf(mu_tilde_1)
        pdf2 = norm.pdf(mu_tilde_2)
        cdf1 = norm.cdf(mu_tilde_1)
        cdf2 = norm.cdf(mu_tilde_2)

        sin_theta = np.clip(np.sqrt(np.clip(1-cos_theta**2, 0, 1)), 0, 1)
        #sin_theta_0 = np.where( np.abs(sin_theta) <= self.TOL)
        #sin_theta = np.clip(sin_theta, self.TOL, None)

        term1 = (x1norm*x2norm*cos_theta+x1sum*x2sum)*(4*bvn_cdf-2*cdf1-2*cdf2+1)
        
        # Handle division by zero carefully
        if isinstance(cos_theta, np.ndarray):
            term2 = 2*x1sum*x2norm*pdf2*erf(\
                    np.divide(mu_tilde_1-cos_theta*mu_tilde_2,
                    np.sqrt(2)*sin_theta, out=np.zeros_like(sin_theta),
                    where=sin_theta!=0))
            term3 = 2*x2sum*x1norm*pdf1*erf(\
                    np.divide(mu_tilde_2-cos_theta*mu_tilde_1,
                    np.sqrt(2)*sin_theta, out=np.zeros_like(sin_theta),
                    where=sin_theta!=0))
        elif sin_theta == 0:
            term2 = 0
            term3 = 0
        else:
            term2 = 2*x1sum*x2norm*pdf2*erf(\
                    (mu_tilde_1-cos_theta*mu_tilde_2)/\
                    (np.sqrt(2)*sin_theta))
            term3 = 2*x2sum*x1norm*pdf1*erf(\
                    (mu_tilde_2-cos_theta*mu_tilde_1)/\
                    (np.sqrt(2)*sin_theta))
        term4 = 4*x1norm*x2norm*sin_theta**2*bvn_pdf
        
        ret = term1 + term2 + term3 + term4
        return ret

    def _single_layer_M(self, x1norm, x1sum):
        #mu_tilde = x1sum/x1norm
        mu_tilde = np.divide(x1sum, x1norm, out=np.zeros_like(x1sum), 
                where=x1norm!=0)
        pdf = norm.pdf(mu_tilde, loc=0)

        return x1sum*erf(mu_tilde/np.sqrt(2))+2*x1norm*pdf


