# External linspaces
import numpy as np
import GPy
import sys
import time
import os

# Internal imports
from ..networks.np_mlp import Mlp
from ..kernels.nn_kernel_relu import NNKernelRelu
from .plotters import plot_2d_samples


############################################
########## EXPERIMENT PARAMETERS ###########
############################################
L           = 64 # number of HIDDEN (not input or output) layers
NUM_X_POINTS= 30 #this is actually the square root of the number of points
ACTIVATIONS = tf.nn.relu
WIDTH       = 3000
OUT_DIR     = 'code/experiments/outputs/samples/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
############################################
############################################

b_scale = 0; b_centre = 0
# By absolute homogeneity of ReLU, it is the ratio of mean to std that
# changes the shape of the samples. 
ratios = [0, -0.2, -0.4, -0.6, -0.8]
w_scale_ = np.sqrt(2)

# Data
x = np.linspace(-2, 2, NUM_X_POINTS)
y = np.linspace(-2, 2, NUM_X_POINTS)
X, Y = np.meshgrid(x, y)
data = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

# Neural network
w_init = tf.keras.initializers.RandomNormal
b_init = tf.keras.initializers.RandomNormal

for i in range(len(ratios)):
    # factor of 10**(i/L) is just a rough scaling to keep functions of order 1
    w_centre = round(ratios[i]*w_scale_*10**(i/32), 2)
    w_scale = round(w_scale_*10**(i/32), 2)
    print(w_centre)
    print(w_scale)

    # Neural network
    mlp = Mlp([2]+L*[WIDTH]+[1],  
            L*[ACTIVATIONS]+[tf.keras.activations.linear],
            w_init, b_init, w_scale, w_centre, b_scale, b_centre)

    sample_mlp = mlp.sample_functions(data, 1)
    plot_2d_samples(data[:,0], data[:,1], sample_mlp[0,:], 
            OUT_DIR + 'samples_2d_mlp_' + str(i) + '.pdf')
    # Also plot with reversed colour map
    plot_2d_samples(data[:,0], data[:,1], -sample_mlp[0,:], 
            OUT_DIR + 'samples_2d_mlp_' + str(i) + '_r.pdf')

    # Gaussian process
    t1 = time.time()
    kern = NNKernelRelu(2, variance_w=w_scale**2, L=L, 
            mean_w = w_centre, variance_b=b_scale**2, mean_b = b_centre)

    sample_gp = kern.sample_prior(data, 1)
    print('Sampled GP in ' + str(time.time()-t1) + " seconds.")
    plot_2d_samples(data[:,0], data[:,1], sample_gp[0,:], 
            OUT_DIR + 'samples_2d_gp_' + str(i) + '.pdf')
    plot_2d_samples(data[:,0], data[:,1], -sample_gp[0,:], 
            OUT_DIR + 'samples_2d_gp_' + str(i) + '_r.pdf')




    

