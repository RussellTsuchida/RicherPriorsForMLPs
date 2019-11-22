# External imports
import numpy as np
import GPy
import sys
import time
import gc
import psutil
import os

# Internal imports
from .plotters import plot_samples
from .plotters import plot_sequence
from ..mmd.mmd import mmd
from ..kernels.nn_kernel_relu import NNKernelRelu
from ..networks.np_mlp import Mlp
from ..networks.np_initialisers import create_initialiser

############################################
########## EXPERIMENT PARAMETERS ###########
############################################
L           = int(sys.argv[1]) # number of HIDDEN (not input or output) layers
NUM_X_POINTS= 10
NUM_SAMPLES = 2000 # for calculation of MMD
X_DIMENSION = 4
ACTIVATIONS = lambda x: np.clip(x, 0, None)
WIDTH       = 200 # Maximum width of networks to compare
STEP        = 1 # Reduce WIDTH in this STEP every iteration of loop
OUT_DIR     = 'code/experiments/outputs/mmd/'

# Reduced memory mode: The user may optionally pas a second argument, i. This 
# indicates that the MMD is measured only on the specified Fi. Use this if 
# the computing system does not have much memory, and running all the F's would
# fail. The user may also pass -1, in which case the data from previous runs
# is put together into a single plot.
try:
    F_INDEX = int(sys.argv[2])
except:
    F_INDEX = 0
############################################
############################################
#### Different initialisation schemes #####
# Each set of functions needs to be consistent (not checked programmatically)
## NOTE: These functions assume A,B,C,D are scaled to have 0 mean and unit 
## variance (i.e, uniform on [-sqrt(3), sqrt(3)], not uniform on [0,1])
# First scheme is just iid uniform mean 0 var 1.
F1 = lambda A, B, C, D: np.sqrt(2)*D    # This is F
EF1 = lambda A, B, C: 0                 # Expectation of F over D
EEF1 = lambda A: 0                      # Expectation of F over C and D
EvarF1 = lambda A: 2                    # Expectation(variance of F over D) over C

# Second scheme is iid uniform mean -1.05 std 1.63
F2 = lambda A, B, C, D: 2*np.sqrt(2)*D - 0.5
EF2 = lambda A, B, C: -0.5                
EEF2 = lambda A: -0.5                   
EvarF2 = lambda A: 8                 

# Third scheme is indep. uniform conditional on A,C
F3 = lambda A, B, C, D: np.sqrt(2)*D - 1.5*A*C  
EF3 = lambda A, B, C: -1.5*A*C                
EEF3 = lambda A: 0                    
EvarF3 = lambda A: 2                 

# Fourth scheme involves 'complicated' mix of A, B, C and D
F4 = lambda A, B, C, D: np.sqrt(2)*D*(A+np.sqrt(3)) - 0.1*A**2*C**2 - 0.4
EF4 = lambda A, B, C: -0.1*A**2*C**2 - 0.4
EEF4 = lambda A:  -0.1*A**2 - 0.4
EvarF4 = lambda A: 2*(A+np.sqrt(3))**2

F_list      = [F1, F2, F3, F4]
EF_list     = [EF1, EF2, EF3, EF4]
EEF_list    = [EEF1, EEF2, EEF3, EEF4]
EvarF1_list = [EvarF1, EvarF2, EvarF3, EvarF4]

# Running in reduced memory mode
if F_INDEX > 0:
    F_list = [F_list[F_INDEX-1]]
    EF_list = [EF_list[F_INDEX-1]]
    EEF_list = [EEF_list[F_INDEX-1]]
    EvarF1_list = [EvarF1_list[F_INDEX-1]]

############################################
############################################
fig_biased = None
fig_unbiased = None

if F_INDEX == -1:
    for i in range(len(F_list)):
        biased_list = np.load(OUT_DIR + 'biased_' + str(L) + '_' + str(i+1) + '.npy')
        unbiased_list = np.load(OUT_DIR + 'unbiased_' + str(L) + '_' + str(i+1) + '.npy')
        widths = np.load(OUT_DIR + 'widths_' + str(L) + '_' + str(i+1) + '.npy')

        fig_biased = plot_sequence(widths, biased_list, OUT_DIR + str(L) + 'mmd_biased_'+\
                str(L) + '_' + str(i+1) + '.pdf', fig_biased, label = '$F' + str(i+1) + '$')
        fig_unbiased = plot_sequence(widths, unbiased_list, OUT_DIR + str(L) + 'mmd_unbiased_'+\
                str(L) + '_' + str(i+1) + '.pdf', fig_unbiased, label = '$F' + str(i+1) + '$')
    sys.exit()

# Test points
x = np.random.normal(size=(NUM_X_POINTS, X_DIMENSION))

b_centre = 0
b_scale  = 0

# Iterate over all the initialisers
for i in range(len(F_list)):
    F       = F_list[i]
    mu      = EF_list[i]
    Emu     = EEF_list[i]
    Esigma2 = EvarF1_list[i]

    # Gaussian process
    t1 = time.time()
    y2 = np.zeros((NUM_SAMPLES, NUM_X_POINTS))
    for s in range(NUM_SAMPLES):
        # Note: A is different in every layer
        A = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(L,))
        #A = np.random.normal(0, 1, size=(L,))

        kern = NNKernelRelu(X_DIMENSION, variance_w=Esigma2(A), L=L, 
                mean_w = Emu(A), variance_b=b_scale**2, mean_b = b_centre,
                standard_first_layer=True)
        y2[s,:] = kern.sample_prior(x, 1)
    print("Sampling GP took " + str(time.time() - t1) + " seconds.")
    
    # Neural network
    w_init = create_initialiser(F, mu, use_normal_ABCD=False)
    b_init = lambda x: np.zeros((x))

    # Find the biased and unbiased MMDs over different MLP widths
    biased_list = []
    unbiased_list = []
    widths = []
    width = WIDTH
    t1 = time.time()
    while width >= STEP:
        y1 = np.zeros((NUM_SAMPLES, NUM_X_POINTS))
        for s in range(NUM_SAMPLES):
            mlp = Mlp([X_DIMENSION]+L*[width]+[1], 
                L*[ACTIVATIONS]+[lambda x: x], 
                weight_init = w_init, bias_init = b_init,
                standard_first=True)

            y1[s,:] = np.reshape(mlp.layer_fun(x.T, all_layers=False), (-1,))

        biased, unbiased = mmd(y1, y2)
        biased_list.append(biased)
        unbiased_list.append(unbiased)
        widths.append(width)
        width = width - STEP
    print("Sampling all MLPs took " + str(time.time() - t1) + " seconds.")
   
    if F_INDEX > 0:
        idx = str(F_INDEX)
    else:
        idx = str(i+1)

    # Save the raw MMDs for plotting later in another program
    np.save(OUT_DIR + 'biased_' + str(L) + '_' + idx + '.npy',
            biased_list)
    np.save(OUT_DIR + 'unbiased_' + str(L) + '_' + idx + '.npy',
            unbiased_list)
    np.save(OUT_DIR + 'widths_' + str(L) + '_' + idx + '.npy',
            widths)

    fig_biased = plot_sequence(widths, biased_list, OUT_DIR + str(L) + 'mmd_biased_'+\
            str(L) + '_' + idx + '.pdf', fig_biased, label = '$F' + idx + '$')
    fig_unbiased = plot_sequence(widths, unbiased_list, OUT_DIR + str(L) + 'mmd_unbiased_'+\
            str(L) + '_' + idx + '.pdf', fig_unbiased, label = '$F' + idx + '$')

    # Attempt to clean up old tf objects (i.e. list of Mlps) and print current
    # memory useage. Unfortunately this doesn't seem to work.
    gc.collect()

    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0]/2.**30
    print('Memory Use: (GB)', memory_use)



