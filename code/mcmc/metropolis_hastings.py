import numpy as np
import scipy.stats

class MetropolisHastings(object):
    def __init__(self, target_dist, d, use_log=False, g_var = None, cor = 0):
        """
        Rough Metropolis-Hastings with hard-coded proposal distribution.

        Args:
            target_dist (function): proportal to the desried probability 
                distribution. Takes (n, d) input and returns (n, 1) output.
                d (int): Dimensionality of samples
                use_log (Bool): True if target_dist is a log density
                cor (float): correlation for bivariate distribution.
        """
        self.d = d
        self.f = target_dist
        self.use_log = use_log

        # Default variance for proposal steps
        if g_var is None:
            g_var = [2.38, 3]
        self.g_var = g_var
        self.g_var = \
            np.asarray([[g_var[0], cor*np.sqrt(g_var[0]*g_var[1])], 
                        [cor*np.sqrt(g_var[0]*g_var[1]), g_var[1]]])

        assert len(self.g_var) == self.d

    def get_proposal(self):
        """
        Plot the proposal distribution as a sanity check.

        Args:
            fname (str): file name to save figure.
        """
        centre = np.asarray([0, 0])
        pdf = np.empty((30, 30))
        for i, x1 in enumerate(np.linspace(-2, 2, 30)):
            for j, x2 in enumerate(np.linspace(-5, 5, 30)):
                pdf[i, j] = self.g_pdf(np.asarray([x1, x2]), centre, self.g_var)
        return pdf            

    def g_sampler(self, x, var):
        """
        Sample from the proposal distribution.

        Args:
            x (nparray): (self.d,) array containing current sample.
            var (float): variance of proposal distribution.
        Returns:
            nparray of size (self.d,)
        """
        return scipy.stats.multivariate_normal.rvs([x[0], x[1]], var)
        #return  np.random.normal([x[0], x[1]], var)

    def g_pdf(self, x, x_new, var):
        """
        PDF of the proposal distribution.

        Args:
            x (nparray): (self.d,) array containing current sample.
            x_new (nparray): (self.d,) array containing new proposed sample.
            var (float): variance of proposal distribution.
        Returns:
            float
        """
        #pdf1 = scipy.stats.norm.pdf(x[0], x_new[0], var[0])*\
        #       scipy.stats.norm.pdf(x[1], x_new[1], var[1])
        #return pdf1
        return scipy.stats.multivariate_normal.pdf(x, x_new, var)

    def g_logpdf(self, x, x_new, var):
        """
        Log PDF of the proposal distribution.

        Args:
            x (nparray): (self.d,) array containing current sample.
            x_new (nparray): (self.d,) array containing new proposed sample.
            var (float): variance of proposal distribution.
        Returns:
            float
        """
        #pdf1 = scipy.stats.norm.logpdf(x[0], x_new[0], var[0]) + \
        #       scipy.stats.norm.logpdf(x[1], x_new[1], var[1])
        #return pdf1
        return scipy.stats.multivariate_normal.logpdf(x, x_new, var)

    def _acceptance_ratio(self, new_sample, sample):
        """
        Calculate the probability of acceptance.

        Args: 
            new_sample (nparray): (self.d,) array containing sample to be 
                accepted or rejected.
            sample (nparray): (self.d,) array containing old sample.
        Returns:
            float. If a value greater than 1, this should be interpreted as 
                100% probability.
        """
        # automatically reject samples with negative variance
        # TODO: this is programmatically hacky but mathematically valid.
        # We require this atm because otherwise we will attempt to evaluate
        # the likelihood of a kernel with negative variance
        if new_sample[1] < 0:
            acceptance_ratio = 0
        else:
            acceptance_ratio = \
            self.f(new_sample)*self.g_pdf(sample, new_sample, self.g_var)\
            /(self.f(sample)*self.g_pdf(new_sample, sample, self.g_var))
        return acceptance_ratio

    def _log_acceptance_ratio(self, new_sample, sample):
        """
        Calculate the logarithm of the probability of acceptance.

        Args: 
            new_sample (nparray): (self.d,) array containing sample to be 
                accepted or rejected.
            sample (nparray): (self.d,) array containing old sample.
        Returns:
            float. If a value greater than 0, this should be interpreted as
                100% probability.
        """
        if new_sample[1] < 0:
            log_acceptance_ratio = -np.inf
        else:
            log_acceptance_ratio = \
            self.f(new_sample) + self.g_logpdf(sample, new_sample, self.g_var) -\
            self.f(sample) - self.g_logpdf(new_sample, sample, self.g_var)
        return log_acceptance_ratio

    def _accept_or_reject(self, new_sample, sample):
        """
        Accept or reject a sample based on acceptance ratio.

        Args: 
            new_sample (nparray): (self.d,) array containing sample to be 
                accepted or rejected.
            sample (nparray): (self.d,) array containing old sample.
        Returns:
            Bool (True to accept, False to reject).
        """
        if not self.use_log:
            ar = self._acceptance_ratio(new_sample, sample)
            return (ar > np.random.uniform())
        else:
            log_ar = self._log_acceptance_ratio(new_sample, sample)
            return (log_ar > np.log(np.random.uniform()))

    def sample(self, initial_sample, num_samples=1000, burn_in = 1000,
            filter_length=100):
        """
        Sample from the target distribution. Includes rudimentary tuning of 
        proposal variance during burn in period.

        Args:
            initial_sample (nparray): (self.d,) representing start point for
                sampler.
            num_samples (int): number of samples required.
            burn_in (int): burn in period.
            filter_length (int): number of points to throw away between samples
                to avoid correlation.
        """
        d = self.d
        samples = np.empty((num_samples, d))
        sample = initial_sample
        c1 = 0  # Number of total samples
        c2 = 0  # Number of accepted samples
        c3 = 0  # Number of stored samples
        while c3 < num_samples:
            # Generate sample
            new_sample = self.g_sampler(sample, self.g_var)
            # Accept or reject
            if self._accept_or_reject(new_sample, sample):
                sample = new_sample
                c2 = c2 + 1
            # Store the sample
            if (c1 > burn_in) and (c1 % filter_length == 0):
                samples[c3,:] = sample
                c3 = c3 + 1
            # Increment
            c1 = c1 + 1
            
            # Acceptance rate and very rough variance tuning
            ar = c2/c1
         
        return samples

