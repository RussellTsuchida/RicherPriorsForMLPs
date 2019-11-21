import scipy.spatial as sp
import numpy as np

def gaussian_kernel(x1,x2):
    dist = sp.distance.cdist(x1,x2)
    return np.exp(-dist**2)

def mmd(x1, x2):
    m = x1.shape[0]
    n = x2.shape[0]
                
    k11 = gaussian_kernel(x1,x1)
    k22 = gaussian_kernel(x2,x2)
    k12 = gaussian_kernel(x1,x2)

    biased = 1./m**2*np.sum(k11) + 1./n**2*np.sum(k22) - 2./(m*n)*np.sum(k12)

    np.fill_diagonal(k11, 0)
    np.fill_diagonal(k22, 0)
    unbiased =  1./(m*(m-1))*np.sum(k11) + 1./(n*(n-1))*np.sum(k22) -\
                2./(m*n)*np.sum(k12)

    return [biased, unbiased]


