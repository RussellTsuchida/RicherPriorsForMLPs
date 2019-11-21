# External libraries
import numpy as np
import GPy
import sys
import time
import os

# Internal imports
from ..networks.np_mlp import Mlp
from ..networks.np_initialisers import create_initialiser
from ..kernels.nn_kernel_relu import NNKernelRelu
from .data.other_datasets import load_or_generate_x_xdash
from .data.other_datasets import load_or_generate_hypersphere
from .plotters import plot_samples

############################################
########## EXPERIMENT PARAMETERS ###########
############################################
L           = 64 # number of HIDDEN (not input or output) layers
NUM_X_POINTS= 100
X_DIMENSION = 10
ACTIVATIONS = lambda x: np.clip(x, 0, None)
WIDTH       = 3000
NUM_SAMPLES = 5 
OUT_DIR     = 'code/experiments/outputs/samples/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
############################################
############################################

############################################
# Different means and variances to try
b_scale = 0; b_centre = 0
# By absolute homogeneity of ReLU, it is the ratio of mean to std that
# changes the shape of the samples. 
ratios = [0, -0.2, -0.4, -0.6, -0.8]
w_scale_ = np.sqrt(2)
############################################

# Load data
data = load_or_generate_hypersphere(NUM_X_POINTS, X_DIMENSION)
theta_list = np.linspace(10e-3, 2*np.pi-10e-3, NUM_X_POINTS)

for i in range(len(ratios)):
    # factor of 10**(i/L) is just a rough scaling to keep functions of order 1
    w_centre = round(ratios[i]*w_scale_*10**(i/32), 2)
    w_scale = round(w_scale_*10**(i/32), 2)
    print(w_centre)
    print(w_scale)
    w_init = create_initialiser(lambda A, B, C, D: w_scale*D + w_centre,
            lambda A, B, C: w_centre)
    b_init = create_initialiser(lambda A, B, C, D: b_scale*D + b_centre,
            lambda A, B, C: b_centre)

    # Neural network
    mlp = Mlp([X_DIMENSION]+L*[WIDTH]+[1],  
            L*[ACTIVATIONS]+[lambda x: x],
            w_init, b_init, standard_first=True)

    samples_mlp = mlp.sample_functions(data.T, NUM_SAMPLES)
    plot_samples(\
            theta_list, samples_mlp, OUT_DIR+'mlp_samples_'+str(i)+'_.pdf', 
            plot_legend=False, xlabel=r'$\theta^{(0)}$', labels=list(range(10)),
    ylabel=r'$f(\mathbf{e}_1\cos\theta^{(0)}+\mathbf{e}_2\sin\theta^{(0)})$',
            linewidth=4, markersize=0, default_colours = True,
            y_tick_labels=False)

    # Gaussian process
    t1 = time.time()
    kern = NNKernelRelu(X_DIMENSION, variance_w=w_scale**2, L=L, 
            mean_w = w_centre, variance_b=b_scale**2, mean_b = b_centre,
            standard_first_layer = True)

    samples_gp = kern.sample_prior(data, NUM_SAMPLES)
    print('Sampled GP in ' + str(time.time()-t1) + " seconds.")
    plot_samples(theta_list, samples_gp, OUT_DIR+'GP_samples_'+str(i)+'_.pdf', 
            plot_legend=False, xlabel=r'$\theta^{(0)}$', labels=list(range(10)),
    ylabel=r'$f(\mathbf{e}_1\cos\theta^{(0)}+\mathbf{e}_2\sin\theta^{(0)})$',
            linewidth=4, markersize=0, default_colours = True, 
            y_tick_labels=False)


