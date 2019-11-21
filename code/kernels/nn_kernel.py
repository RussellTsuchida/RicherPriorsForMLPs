import numpy as np
import GPy
import abc
import warnings
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.special import erf

from .TVTST3 import vec_bvnd

class NNKernel(GPy.kern.Kern):
    __metaclass__ = abc.ABCMeta
    TOL = 0#10e-8
    def __init__(self, input_dim, variance_w = 1., mean_w = 0, 
            variance_b = 0, mean_b = 0, L=1, name='nn_generic',
            standard_first_layer = False):
        """
        Equivalent kernel of a neural network with a single hidden layer and
        a particular activation function.

        Args:
            input_dim (int): Dimensionality of input.
            variance_w (float or list(float)): Variance of the weights. If a
                single float, the weights in each layer will share the same
                variance. Otherwise if a list, the length of the list must be
                equal to L. If standard_first_layer==True, the first element
                of the list will be ignored.
            mean_w (float or list(float)): Mean of the weights. See variance_w
                for list behaviour.
            variance_b (float): Variance of the biases. See variance_w
                for list behaviour.
            mean_b (float): Mean of the biases.  See variance_w
                for list behaviour.
            L (int): The number of hidden layers.
            standard_first_layer (Bool): True to use 0 mean var 1 weights
                in first layer. Otherwise use mean_w and variance_w.
        """
        super().__init__(input_dim=input_dim, active_dims=range(input_dim),
                name=name)

        self._check_specifications(input_dim, variance_w, mean_w,
                variance_b, mean_b, L, standard_first_layer)
    
    def _scalar_to_list(self, scalar):
        """
        Check if scalar is a scalar, and if it is convert it to a list of length
        self.L

        Args:
            scalar(scalar numeric or list or nparray)
        """
        if np.isscalar(scalar):
            scalar = [scalar]*self.L
        else:
            assert len(list(scalar)) == self.L
        return scalar
 
    def _set_first_layer_standard_params(self):
        """
        If standard_first_layer is true, the parameters in the first layer 
        should be standardised.
        """
        if self.standard_first_layer:
            self.w_variance[0] = 1; self.w_mean[0] = 0; 
            self.b_variance[0] = 0; self.b_mean[0] = 0

    def _check_specifications(self, input_dim, variance_w, mean_w,
            variance_b, mean_b, L, standard_first_layer):
        """
        Check the input parameters and set attributes.
        """
        self.input_dim  = input_dim
        self.L = L
        self.standard_first_layer = standard_first_layer

        self.w_variance, self.b_variance, self.w_mean, self.b_mean = \
                [self._scalar_to_list(param)\
                for param in [variance_w, variance_b, mean_w, mean_b]]

        self._set_first_layer_standard_params()

        self.w_variance=GPy.core.parameterization.Param('weight_variance', 
                self.w_variance)
        self.b_variance=GPy.core.parameterization.Param('bias_variance', 
                self.b_variance)
        self.w_mean=GPy.core.parameterization.Param('weight_mean', 
                self.w_mean)
        self.b_mean=GPy.core.parameterization.Param('bias_mean', 
                self.b_mean)

    def input_norms_sum_cos_theta(self, x1, x2):
        """
        Find the norms of the inputs (after appending the biases), 
        the sums of the inputs (after appending biases),  and the
        cosine angle between the inputs (after appending the biases).

        Args:
            x1 (nparray):
            x2 (nparray):
        Returns:
            x1norm (nparray)
            x2norm (nparray)
            cos_theta (nparray)
        """
        if x2 is None:
            x2 = x1
        #if self.standard_first_layer:
        #    w_mu = 0.; w_var = 1.; b_mu = 0.; b_var = 1.
        #else:
        w_mu = self.w_mean[0]; w_var = self.w_variance[0]; 
        b_mu = self.b_mean[0]; b_var = self.b_variance[0]

        x1sum = np.reshape(np.sum(x1, axis=1), (-1,1))*w_mu + b_mu
        x2sum = np.reshape(np.sum(x2, axis=1), (1,-1))*w_mu + b_mu
        
        x1 = np.hstack((x1*np.sqrt(w_var), 
            np.ones((x1.shape[0],1))*np.sqrt(b_var)))
        x2 = np.hstack((x2*np.sqrt(w_var), 
            np.ones((x2.shape[0],1))*np.sqrt(b_var)))

        x1norm = np.reshape(np.linalg.norm(x1, axis=1), (-1, 1))
        x2norm = np.reshape(np.linalg.norm(x2, axis=1), (1, -1))

        dot_prod = np.matmul(x1, np.transpose(x2))
        bothnorms = x1norm*x2norm
        cos_theta = np.divide(dot_prod, bothnorms, out=np.zeros_like(dot_prod),
                where=bothnorms!=0)
        cos_theta = np.maximum(np.minimum(cos_theta, 1.), -1.)

        return [x1norm, x2norm, x1sum, x2sum, cos_theta]

    def K(self, x1, x2=None):
        """
        Implementation of the kernel, required for GPy.
        """
        x2_none = (x2 is None)
        #x2_none = False
        x1norm, x2norm, x1sum, x2sum, cos_theta = \
            self.input_norms_sum_cos_theta(x1, x2)

        for l in range(0, self.L):
            x1norm_old = x1norm
            x2norm_old = x2norm

            k = self._single_layer_K(x1norm,x2norm,x1sum,x2sum,cos_theta, 
                x2_none)
            x1norm=np.sqrt(np.clip(self._single_layer_K(x1norm,x1norm,x1sum,
                x1sum,1,x2_none), 0, None))
            x2norm=np.sqrt(np.clip(self._single_layer_K(x2norm,x2norm,x2sum,
                x2sum,1,x2_none), 0, None))

            # Affine transform
            x1sum = self.w_mean[l]*self._single_layer_M(x1norm_old, x1sum) + \
                    self.b_mean[l]
            x2sum = self.w_mean[l]*self._single_layer_M(x2norm_old, x2sum) + \
                    self.b_mean[l]
            k = self.w_variance[l]*k + self.b_variance[l]

            x1norm = np.sqrt(self.w_variance[l]*x1norm**2+self.b_variance[l])
            x2norm = np.sqrt(self.w_variance[l]*x2norm**2+self.b_variance[l])
            bothnorms = x1norm*x2norm
            cos_theta = np.divide(k, bothnorms, out=np.zeros_like(k),
                    where=bothnorms!=0)
            cos_theta = np.maximum(np.minimum(cos_theta, 1.), -1.)

            # k Could be not symmetric due to the fact that it involves 
            # numerical evaluation of the univariate and bivariate CDFs.
            # This calculation is not deterministic. The error may magnify
            # especially when L is large. To force the matrix 
            # to be symmetric, we can average itself and its transpose.
            # We should warn the user when this happens.
            k = cos_theta*bothnorms
            if x2_none and not (np.allclose(k, k.T)):
                warnings.warn('Kernel matrix not symmetric for ' + \
                        str(l) + ' layer GP. Forcing symmetric...')
            if x2_none:
                k = (k + k.T)/2.
                cos_theta = (cos_theta + cos_theta.T)/2.
                x1sum = (x1sum + x2sum.T)/2.
                x2sum = x1sum.T
                x1norm = (x1norm + x2norm.T)/2.
                x2norm = x1norm.T

        return k

    @abc.abstractmethod
    def _single_layer_K(self, x1norm, x2norm, x1sum, x2sum, cos_theta, 
            x2_none = False):
        """
        Kernel for a single layer
        """
        pass

    @abc.abstractmethod
    def _single_layer_M(self, x1norm, x1sum):
        """
        First order "M" term for a single layer
        """
        pass

    @classmethod
    def _vectorised_bvn_cdf_pdf(cls, mu_tilde_1, mu_tilde_2, cos_theta, 
            vectorised = True, x2_none=False):
        # Keep this for verification purposes
        if (not vectorised):
            return cls._unvectorised_bvn_cdf_pdf(mu_tilde_1, mu_tilde_2,
                    cos_theta)

        # Interleave to make a vector of each mu combination
        mu_1_mat = np.tile(mu_tilde_1, [1, mu_tilde_2.shape[1]])
        mu_2_mat = np.tile(mu_tilde_2, [mu_tilde_1.shape[0], 1])

        # Handle K(x,x)
        if (type(cos_theta) is int):
            if cos_theta == 1: # X1 = X2 a.s, so the pdf/cdf is 1D
                cdfs = norm.cdf(mu_tilde_1)
                pdfs = norm.pdf(mu_tilde_1)
            elif cos_theta == -1: # X1=-X2 a.s, so cdf/pdf is 0
                cdfs = np.zeros_like(mu_tilde_1)
                pdfs = np.zeros_like(mu_tilde_1)
            return [cdfs, pdfs]

        # We only need to evaluate the upper diagonals
        if x2_none:
            idx = np.triu_indices(cos_theta.shape[0])
            mu_1_mat = mu_1_mat[idx]
            mu_2_mat = mu_2_mat[idx]
            cos_theta = cos_theta[idx]

        mu_1_flat = mu_1_mat.flatten()
        mu_2_flat = mu_2_mat.flatten()
        flat_cos_theta = cos_theta.flatten()
        flat_sin_theta = np.clip(np.sqrt(np.clip(1-flat_cos_theta**2, 0, 1)),
                0, 1)
        n = flat_cos_theta.shape[0]

        # Handle zero and pi angles
        angle_zero = np.where(np.abs(flat_cos_theta - 1) <= cls.TOL)
        angle_pi = np.where(np.abs(flat_cos_theta + 1) <= cls.TOL)

        # Calculate pdf of standard normals, lots of stuff to avoid div 0
        Z = 2*np.pi*flat_sin_theta
        exponent = np.divide(\
                mu_1_flat**2-2*mu_1_flat*mu_2_flat*flat_cos_theta+mu_2_flat**2,
                2*flat_sin_theta**2, out=np.inf*np.ones_like(flat_sin_theta),
                where=flat_sin_theta!=0)
        pdfs = np.divide(np.exp(-exponent), Z, out=np.zeros_like(Z), 
                where = Z!=0)
        pdfs[angle_zero] = norm.pdf(mu_1_flat[angle_zero])
        pdfs[angle_pi] = np.zeros_like(mu_1_flat[angle_pi])

        # Calculate the cdf 
        cdfs = np.empty((n,))
        vec_bvnd(-mu_1_flat, -mu_2_flat, flat_cos_theta, cdfs, n)

        # Make symmetric matrix or reshape
        if x2_none:
            pdfs_out = np.zeros((mu_tilde_1.shape[0], mu_tilde_2.shape[1]))
            cdfs_out = np.zeros((mu_tilde_1.shape[0], mu_tilde_2.shape[1]))
            pdfs_out[idx] = pdfs
            cdfs_out[idx] = cdfs
            pdfs_out = pdfs_out + pdfs_out.T - np.diag(pdfs_out.diagonal())
            cdfs_out = cdfs_out + cdfs_out.T - np.diag(cdfs_out.diagonal())
        else:
            pdfs_out = np.reshape(pdfs, (cos_theta.shape[0], cos_theta.shape[1]))
            cdfs_out = np.reshape(cdfs, (cos_theta.shape[0], cos_theta.shape[1]))

        return [cdfs_out, pdfs_out]

    @classmethod
    def _unvectorised_bvn_cdf_pdf(cls, mu_tilde_1, mu_tilde_2, cos_theta):
        '''
        TODO: This inefficient code loops through all combinations of mu1, mu2
        and cos_theta. Ideally we would have a "vectorised" implementation.
        
        Args:
            mu_tilde_1 (nparray): (n x 1) array of means in first dimension.
            mu_tilde_2 (nparray): (1 x m) array of means in second dimension.
            cos_theta (np_array): (n x m) array of correlation coefficients.
        Returns:
            List containing two (n x m) arrays: the first represents the cdfs
            and the second represents the pdfs.
        '''
        n = mu_tilde_1.shape[0]
        m = mu_tilde_2.shape[1]

        cdf_array = np.empty((n, m))
        pdf_array = np.empty((n, m))

        for i in range(n):
            mu1 = mu_tilde_1[i,0]
            for j in range(m):
                if (type(cos_theta) is int) and (cos_theta == 1):
                    rho = 1
                else:
                    rho = cos_theta[i,j]

                mu2 = mu_tilde_2[0,j]
                
                mean = np.array([mu1, mu2])
               # Handle degenerate distributions
                if abs(rho - 1.) <= cls.TOL: # X1 = X2 a.s, so the pdf/cdf is 1D
                    bvn_cdf = norm.cdf(mu1)
                    bvn_pdf = norm.pdf(mu1)
                elif abs(rho + 1.) <= cls.TOL: # X1=-X2 a.s, so cdf/pdf is 0
                    bvn_cdf = 0
                    bvn_pdf = 0
                else:
                    cov = np.array([[1, rho],[rho, 1]])
                    bvn_dist = mvn(mean=np.array([0,0]), cov=cov)
                    bvn_cdf = bvn_dist.cdf(mean)
                    bvn_pdf = bvn_dist.pdf(mean)

                cdf_array[i,j] = bvn_cdf
                pdf_array[i,j] = bvn_pdf
        
        return [cdf_array, pdf_array]

    def Kdiag(self, X):
        k = self.K(X)

        return np.reshape(np.diag(k), (-1,))

    def update_gradients_full(self, dl_dK, X, X2):
        pass

    def normalised_kernel_given_data(self, data, layers=[1,2,4,8,16,32]):
        """
        Evaluate the normalised kernel empirically for each specified layer.

        Args:
            data (np.array(float)): size (n, 2*self.layer_widths[0])
                representing a set of n pairs of datapoints.
            layers (list(int)): list of layer indices where the normalised 
                kernel is desired. If some layers don't exist, they will
                be ignored.
        Returns:
            list of nparrays. Each element in the array represents a layer
            as specified by layers.
        """
        # Remove layers that don't exist
        layers = [l for l in layers if l<=self.L]

        # Feed data through the layers
        outputs = np.empty((len(layers), data.shape[0]))

        x       = data[:,:self.input_dim]
        xdash   = data[:,self.input_dim:]

        for pair in range(data.shape[0]):
            x1 = np.atleast_2d(x[pair,:])
            x2 = np.atleast_2d(xdash[pair, :])

            x1norm, x2norm, x1sum, x2sum, cos_theta = \
                    self.input_norms_sum_cos_theta(x1, x2)
            for layer in range(1, layers[-1]+1):
                x1norm_old = x1norm
                x2norm_old = x2norm

                k = self._single_layer_K(x1norm,x2norm,x1sum,x2sum,cos_theta)
                x1norm=np.sqrt(self._single_layer_K(x1norm,x1norm,x1sum,x1sum,1))
                x2norm=np.sqrt(self._single_layer_K(x2norm,x2norm,x2sum,x2sum,1))

                cos_theta = k/(x1norm*x2norm)
                cos_theta = np.maximum(np.minimum(cos_theta, 1.), -1.)
                
                if layer in layers:
                    outputs[layers.index(layer),pair] = cos_theta

                # Affine transform
                x1sum = self.w_mean[layer-1]*self._single_layer_M(x1norm_old, x1sum) + \
                        self.b_mean[layer-1]
                x2sum = self.w_mean[layer-1]*self._single_layer_M(x2norm_old, x2sum) + \
                        self.b_mean[layer-1]

                k = self.w_variance[layer-1]*k + self.b_variance[layer-1]
                x1norm = np.sqrt(self.w_variance[layer-1]*x1norm**2+\
                        self.b_variance[layer-1])
                x2norm = np.sqrt(self.w_variance[layer-1]*x2norm**2+\
                        self.b_variance[layer-1])
                cos_theta = k/(x1norm*x2norm)
                cos_theta = np.maximum(np.minimum(cos_theta, 1.), -1.)

        return outputs

    def sample_prior(self, data, num_samples=5):
        return np.random.multivariate_normal(data.shape[0]*[0], self.K(data), 
                size=num_samples)
