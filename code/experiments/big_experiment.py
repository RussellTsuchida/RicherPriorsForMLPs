# External imports
import numpy as np
import GPy
import sys
import time
from scipy.stats import norm, gamma, invgamma

# Internal imports
from ..kernels.nn_kernel_relu import NNKernelRelu
from .data.other_datasets import load_or_generate_snelson
from .data.other_datasets import load_or_generate_smooth_xor
from .data.other_datasets import load_or_generate_1d_regression
from .plotters import plot_likelihood
from .plotters import plot_posterior_samples
from .plotters import plot_samples
from .plotters import plot_gp
from .plotters import plot_gp_errors
from .plotters import plot_proposal
from .dynamic_grid import DynamicGrid
from .experiment_array import ExperimentArray
from .data.datasets import Ghcn
from .data.datasets import Mauna
from .data.datasets import Yacht
from .data.datasets import Normaliser
from ..mcmc.metropolis_hastings import MetropolisHastings
from ..mvn_mixture.diag_mvn_mixture import DiagMVNMixture

############################################
########## EXPERIMENT PARAMETERS ###########
############################################
L           = int(sys.argv[1])
USE_LOG     = True if sys.argv[2] == 'log' else False
DATASET     = sys.argv[3]

GRID_SIZE   = 100
NOISE_VAR   = 0.1 # If this setting is changed, FORCE_NEW should be set to true
FORCE_NEW   = False # Generate a new dataset or use an old one
REPARAM     = False # True to use the reparameterisation
MH_NUM_SAMPLES  = 100
MH_BURN_IN      = 20
MH_FILTER       = 20
OUT_DIR         = 'code/experiments/outputs/' + DATASET + '/'
############################################
############################################
if DATASET == 'snelson':
    X, Y, X_test, Y_test = load_or_generate_snelson()
elif DATASET == '1d_regression':
    X, Y, X_test, Y_test = load_or_generate_1d_regression(force_new = FORCE_NEW,
        noise_var = NOISE_VAR)
elif DATASET == 'smooth_xor':
    X, Y, X_test, Y_test = load_or_generate_smooth_xor(force_new = FORCE_NEW,
        noise_var = NOISE_VAR)
elif DATASET == 'mauna':
    data = Mauna(OUT_DIR)
    X, Y, X_test, Y_test = data.load_or_generate_data(FORCE_NEW)
    normaliser = Normaliser(X, Y, mode='normalise')
    X, _            = normaliser.normalised_XY(X, Y)
    X_test, _  = normaliser.normalised_XY(X_test, Y_test)
    normaliser = Normaliser(X, Y, mode='standardise')
    _, Y            = normaliser.normalised_XY(X, Y)
    _, Y_test  = normaliser.normalised_XY(X_test, Y_test)
elif DATASET == 'yacht':
    data = Yacht(OUT_DIR, train_size=100, test_size=208)
    X, Y, X_test, Y_test = data.load_or_generate_data(FORCE_NEW)
    normaliser = Normaliser(X, Y, mode='normalise')
    X, _            = normaliser.normalised_XY(X, Y)
    X_test, _  = normaliser.normalised_XY(X_test, Y_test)
    normaliser = Normaliser(X, Y, mode='standardise')
    _, Y            = normaliser.normalised_XY(X, Y)
    _, Y_test  = normaliser.normalised_XY(X_test, Y_test)

# Prior for w_centre and w_var
prior_w_centre_mu_0 = -1
prior_w_centre_sigma = 2
# Mode = scale/(shape+1), Mean = scale/(shape-1), shape > 1
# Variance scale^2/( (shape-1)^2 (shape-2) ), shape > 2
prior_w_var_shape = 2.5
prior_w_var_scale_0 = 6

# Variance for proposal
prop_var_0 = [2.38, 2.38*2]

sigma_left      = 1.
mu_bot_0        = -4
mu_top          = 0.1
# Starting grid points are different for different datasets
if (DATASET == 'yacht') or (DATASET == 'mauna'):
    dynamic_grid = DynamicGrid([1., 1.], [3.5, -0.5])
else:
    dynamic_grid = DynamicGrid([1., 0.1], [10, -4])

# Quantile p values for the # outside graph
quantile = [0.125, 0.875]

# Depths 2 to 64
depth_list      = list(range(2,65))
mse_list        = ExperimentArray((5, 63), OUT_DIR + sys.argv[2] + '/mse/') 
num_outside_list= ExperimentArray((5, 63), OUT_DIR + sys.argv[2] + '/num/') 

# Scale the prior, proposal variance, and plot window with the depth
if REPARAM:
    prior_w_var_scale = prior_w_var_scale_0
    prior_w_centre_mu = -0.5
    prop_var = prop_var_0
else:
    prior_w_var_scale = prior_w_var_scale_0
    prior_w_centre_mu = prior_w_centre_mu_0
    prop_var = prop_var_0

# Set up the model and likelihood
if REPARAM:
    kern = lambda w_centre, w_var: \
        NNKernelRelu(X.shape[1], variance_w=w_var**2, L=L, 
        mean_w=w_centre*(w_var), 
        variance_b=w_var**2, 
        mean_b = w_centre*(w_var),
        standard_first_layer=False)
else:
    kern = lambda w_centre, w_var: \
        NNKernelRelu(X.shape[1], variance_w=w_var, L=L, 
        mean_w=w_centre, 
        variance_b=w_var, mean_b = w_centre,
        standard_first_layer=False)
model = lambda w_centre, w_var: \
    GPy.models.GPRegression(X, Y, kern(w_centre, w_var),noise_var=NOISE_VAR)
if USE_LOG:
    ll = lambda w_centre, w_var: model(w_centre, w_var).log_likelihood()
else:
    ll = lambda w_centre, w_var: np.exp(\
            model(w_centre, w_var).log_likelihood())

# Set up the prior
invgamma_rv = invgamma(a=prior_w_var_shape)
normal_rv = norm(loc=prior_w_centre_mu, scale=prior_w_centre_sigma)
if USE_LOG:
    prior = lambda w_centre, w_var: \
        normal_rv.logpdf(w_centre) + \
        invgamma_rv.logpdf(w_var/prior_w_var_scale)-np.log(prior_w_var_scale)
else:
    prior = lambda w_centre, w_var: \
        normal_rv.pdf(w_centre) * \
        invgamma_rv.pdf(w_var/prior_w_var_scale)/prior_w_var_scale

# Evaluate the likelihood over a dynamic grid. First coarsely then finely
max_like = -np.inf
max_like_c0 = -np.inf

for coarse_fine in range(2):
    # Update the coarse grid 
    if coarse_fine == 1:
        dynamic_grid.update_corners(likelihood, USE_LOG, GRID_SIZE)
    sigma2_list, mu_list = dynamic_grid.one_dimensional_grids()
    SIGMA2, MU = dynamic_grid.two_dimensional_grid()

    likelihood = np.empty((len(mu_list), len(sigma2_list)))
    likelihood.fill(np.nan)
    for i, sigma2 in enumerate(sigma2_list):
        for j, mu in enumerate(mu_list):
            try:
                value = ll(mu, sigma2)
                likelihood[j, i] =  value
                if value > max_like:
                    max_like = value
                    MLE = np.asarray([mu, sigma2])
                if (mu == 0) and (value > max_like_c0) :
                    max_like_c0 = value
                    MLE_c0 = np.asarray([mu, sigma2])
            except:
                print("Bad hyperparameters " + str(mu) + '_' + str(sigma2))


# Plot the likelihood
plot_likelihood(likelihood, sigma=SIGMA2, mu=MU,
        extent=[min(sigma2_list), max(sigma2_list), min(mu_list), max(mu_list)],
        name=OUT_DIR + 'likelihood' + str(L) + '_'+ str(USE_LOG) + \
                '.pdf', reparam = REPARAM)
plot_posterior_samples(np.reshape(MLE_c0, (1,2)), 
        name=OUT_DIR + 'likelihood' + str(L) + '_'+ str(USE_LOG) + \
                '.pdf', close=False, marker='m*')
plot_posterior_samples(np.reshape(MLE, (1,2)), 
        name=OUT_DIR + 'likelihood' + str(L) + '_'+ str(USE_LOG) + \
                '.pdf', close=True, marker='r*')

# The target is the prior multiplied by the likelihood
if USE_LOG:
    # Subtract log of maximum likelihood to keep around order 0
    target = lambda x: ll(x[0], x[1]) + prior(x[0], x[1]) - max_like
else:
    # Divide by the maximum likelihood to keep likelihood of order 1
    target = lambda x: ll(x[0], x[1]) * prior(x[0], x[1]) / max_like 

# Plot the target distribution
if not USE_LOG:
    post = likelihood * prior(MU, SIGMA2)
else:
    post = likelihood + prior(MU, SIGMA2)

indx = np.unravel_index(np.argmax(post, axis=None), post.shape)
MAP = np.asarray([mu_list[indx[0]], sigma2_list[indx[1]]])
print('MAP:')
print(MAP)
post_c0 = post[np.where(mu_list == 0)[0][0], :]
indx = np.unravel_index(np.argmax(post_c0, axis=None), post_c0.shape)
MAP_c0 = np.asarray([0, sigma2_list[indx[0]]])

# Get samples from the posterior using MH, starting at the MAP
mh = MetropolisHastings(target, 2, USE_LOG, prop_var, 0 if REPARAM else -0.9)
#plot_proposal(mh, OUT_DIR + 'proposal' + str(L) + '.pdf',
#extent=[min(sigma2_list), max(sigma2_list), min(mu_list), max(mu_list)])
samples = mh.sample(MAP, num_samples=MH_NUM_SAMPLES, 
        burn_in = MH_BURN_IN, filter_length = MH_FILTER)

# Approximate the MAP as the "most likely" sample we obtained under the posterior
max_post = -np.inf
max_post_c0 = -np.inf
for sample_idx in range(samples.shape[0]):
    sample = samples[sample_idx, :]
    posterior_density = target(sample)
    if posterior_density > max_post:
        max_post = posterior_density
        MAP = sample
print(MLE)
print(MAP)

# Plot the hyper posterior
plot_likelihood(post, sigma=SIGMA2, mu=MU,
        extent=[min(sigma2_list), max(sigma2_list), min(mu_list), max(mu_list)],
        name=OUT_DIR + 'posterior' + str(L) + '_' + str(USE_LOG) + \
                '.pdf', reparam = REPARAM)
plot_posterior_samples(samples, name=OUT_DIR + 'posterior' + \
        str(L) +'_' + str(USE_LOG) + '.pdf')
plot_posterior_samples(np.reshape(MAP_c0, (1,2)), 
        name=OUT_DIR + 'posterior' + str(L) + '_' + str(USE_LOG) + \
                '.pdf', marker='m*')
plot_posterior_samples(np.reshape(MAP, (1,2)), 
        name=OUT_DIR + 'posterior' + str(L) + '_' + str(USE_LOG) + \
                '.pdf', close=True, marker='r*')

# Compute predictive posterior predictive using MAP
m = model(MAP[0], MAP[1])
mean, var = m.predict(X_test)
var = np.clip(var, 0, None)
if X.shape[1] == 1:
    plot_gp(mean, np.sqrt(var), X_test, Y_test, X, Y, name=OUT_DIR + 'gp_MAP'+ \
            str(L) + '_' + str(USE_LOG) + '.pdf')
mvn_mix = DiagMVNMixture(mean, np.sqrt(var))
mse = mvn_mix.error_mse_to_mean(Y_test)
num_outside = mvn_mix.error_num_outside_quantile(Y_test, quantile)
mse_list[3, L-2] = mse
num_outside_list[3, L-2] = num_outside[0]

# Compute predictive posterior predictive using MLE
m = model(MLE[0], MLE[1])
mean, var = m.predict(X_test)
var = np.clip(var, 0, None)
if X.shape[1] == 1:
    plot_gp(mean, np.sqrt(var), X_test, Y_test, X, Y, name=OUT_DIR + 'gp_MLE' +\
            str(L) + '_' + str(USE_LOG) + '.pdf')
mvn_mix = DiagMVNMixture(mean, np.sqrt(var))
mse = mvn_mix.error_mse_to_mean(Y_test)
num_outside = mvn_mix.error_num_outside_quantile(Y_test, quantile)
mse_list[2, L-2] = mse
num_outside_list[2, L-2] = num_outside[0]

# Compute predictive posterior predictive using MAP restricted to c=0
m = model(MAP_c0[0], MAP_c0[1])
mean, var = m.predict(X_test)
var = np.clip(var, 0, None)
if X.shape[1] == 1:
    plot_gp(mean, np.sqrt(var), X_test, Y_test, X, Y, name=OUT_DIR +'gp_MAPc0'+\
            str(L) + '_' + str(USE_LOG) + '.pdf')
mvn_mix = DiagMVNMixture(mean, np.sqrt(var))
mse = mvn_mix.error_mse_to_mean(Y_test)
num_outside = mvn_mix.error_num_outside_quantile(Y_test, quantile)
mse_list[1, L-2] = mse
num_outside_list[1, L-2] = num_outside[0]

# Compute predictive posterior predictive using MLE restricted to c=0
m = model(MLE_c0[0], MLE_c0[1])
mean, var = m.predict(X_test)
var = np.clip(var, 0, None)
if X.shape[1] == 1:
    plot_gp(mean, np.sqrt(var), X_test, Y_test, X, Y, name=OUT_DIR +'gp_MLEc0'+\
            str(L) + '_' + str(USE_LOG) + '.pdf')
mvn_mix = DiagMVNMixture(mean, np.sqrt(var))
mse = mvn_mix.error_mse_to_mean(Y_test)
num_outside = mvn_mix.error_num_outside_quantile(Y_test, quantile)
mse_list[0, L-2] = mse
num_outside_list[0, L-2] = num_outside[0]

# Compute predictive posterior using full hyperposterior
mus = np.empty((X_test.shape[0], 0))
sigmas = np.empty((X_test.shape[0], 0))
for s_idx in range(samples.shape[0]):
    sample = samples[s_idx,:]
    m = model(sample[0], sample[1])
    mean, var = m.predict(X_test)
    var = np.clip(var, 0, None)
    mus = np.hstack((mus, mean))
    sigmas = np.hstack((sigmas, np.sqrt(var)))
if X.shape[1] == 1:
    plot_gp(mus, sigmas, X_test, Y_test, X, Y, 
            name=OUT_DIR + 'gp_maginalised' + str(L) + '_' +str(USE_LOG)+'.pdf')
mvn_mix = DiagMVNMixture(mus, sigmas)
mse = mvn_mix.error_mse_to_mean(Y_test)
num_outside = mvn_mix.error_num_outside_quantile(Y_test, quantile)
mse_list[4, L-2] = mse
num_outside_list[4, L-2] = num_outside[0]

# Plot the MSE and Number outside quantile
plot_gp_errors(depth_list[0:L-1], mse_list.load_array()[:,0:L-1], 'MSE', 
        name=OUT_DIR + 'gp_mse' + str(L) + '_' + str(USE_LOG) + '.pdf')
plot_gp_errors(depth_list[0:L-1], num_outside_list.load_array()[:,0:L-1], 
        '# outside quantile', 
        name=OUT_DIR + 'gp_num' + str(L) + '_' + str(USE_LOG) + '.pdf')

