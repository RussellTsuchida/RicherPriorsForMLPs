import numpy as np
from scipy import special
from scipy.stats.mstats import mquantiles

class DiagMVNMixture(object):
    """
    Diagonal multivariate normal mixture model
    """
    def __init__(self, means, diag_var, coefficients=None):
        """
        n dimensional mixture of m MVNs.

        Args:
            means (nparray): (n, m) means for each m component.
            diag_var (nparray): (n, m) variances for each m component.
            coefficients (Nonetype or nparray): (m, 1) mixture component 
                coefficients. If none, use a coefficient of 1/m for each.
        """
        self.n = means.shape[0]
        self.m = means.shape[1]
        assert self.n == diag_var.shape[0]
        assert self.m == diag_var.shape[1]

        if coefficients is None:
            coefficients = np.ones((self.m,1))/self.m

        self.means          = means
        self.diag_var       = diag_var
        self.coefficients   = coefficients

    def quantiles(self, prob, num_samples=10000):
        """
        Get the quantiles empirically by sampling.

        Args:
            prob (list(float)): The quantile probabilities.
            num_samples (int): Number of samples to use in calculation.
        """
        if self.m == 1:
            return self.multivariate_normal_quantile(\
                    self.means.flatten(), self.diag_var.flatten(), prob)

        samples = np.empty((0, self.n))
        for component in range(0, self.m):
            coeff = self.coefficients[component]
            mu_component = self.means[:,component]
            sigma_component = self.diag_var[:,component]
            z = np.random.normal(0, 1, (int(num_samples/self.m), 
                self.n))
            z = sigma_component*z + mu_component
            samples = np.vstack((samples, z))

        quantiles = mquantiles(samples, axis=0,
            prob=prob)
        return quantiles
    
    @staticmethod
    def multivariate_normal_quantile(mus, sigmas, ps):
        """
        Multivariate normal quantile for when there is only 1 mixture component.

        Args:
            mus (nparray): (n,) array of means of normals
            sigmas (nparray): (n,) array of std of normals
            ps (nparray): (m,) array of probabilities
        Returns:
            quantiles (nparray): (n,m) array of quantiles for each mu and sigma
        """
        ps = np.asarray(ps)
        standard_qs =  np.sqrt(2)*special.erfinv(2*ps-1)
        standard_qs = np.reshape(standard_qs, (-1,1))
        return mus + sigmas*standard_qs

    def mean(self):
        """
        The mean of the mixture distribution is just the average of the means of
        the components.
        """
        return np.mean(self.means, axis=1)

    def error_mse_to_mean(self, y):
        """
        Find the mse between the targets y and the mean of the mixture 
        distribution.

        Args:
            y (nparray): (n, 1) targets, to compare with the means.
        Returns:
            float representing MSE.
        """
        mean = np.reshape(self.mean(), (-1,))
        y = np.reshape(y, (-1,))
        assert y.shape[0] == self.n
        return np.sum( (y - mean)**2)/self.n

    def error_num_outside_quantile(self, y, prob, num_samples=10000):
        """
        Count the number of occurances where y is outside the quantile.

        Args:
            y (nparray): (n, 1) targets to check if inside quantiles.
            prob (list): Defines quantiles. Length must be an even number
            num_samples (int): number of samples to use for quantile sample.

        Returns:
            list of ints of number of times y is outside specified quantiles.
        """
        assert len(prob)%2 == 0
        quantile = self.quantiles(prob, num_samples)
        y = np.reshape(y, (-1,))
        num_outside_list = []
        for q in range(0, int(quantile.shape[0]/2)):
            maxx = quantile[-(q+1),:]
            minn = quantile[q,:]

            idx1 = (y >= maxx)
            idx2 = (y <= minn)

            invalid_idx = np.logical_or(idx1, idx2)
            num_outside = np.sum(invalid_idx)
            
            num_outside_list.append(num_outside)

        return num_outside_list

