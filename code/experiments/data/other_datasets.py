import numpy as np
import itertools

def load_or_generate_x_xdash(theta_list, input_dim, 
        fname='code/experiments/data/theta.npy', force_new=True):
    """
    Sample random points, x and x' from mutual random orthogonal 
    transformations of (1,0) and (cos(theta), sin(theta)), where theta
    is the angle between x and xdash.
   
    Note: The dimensions of the output might not be what you expect from a
    "typical" dataset.

    Args:
        theta (nparray(float)): Angle required between x and xdash
        input_dim (int): Dimensionality of x and x'
        force_new (bool): Force generation of a new dataset
    Returns:
        (theta_list.shape[0], 2*input_dim) np array. The first half of columns
        represent the x points and the second half represent the xdash points.
    """
    try:
        if force_new:
            raise Exception
        out = np.load(fname)
    except:
        out = np.empty((theta_list.shape[0], 2*input_dim))
        u = np.asarray([[1], [0]])

        theta_list = list(theta_list)

        for theta in theta_list:
            v = np.asarray([[np.cos(theta)], [np.sin(theta)]])

            A = np.random.uniform(size=(input_dim, 2))
            q, _ = np.linalg.qr(A)

            x       = np.transpose(np.matmul(q,u))
            xdash   = np.transpose(np.matmul(q,v))

            out[theta_list.index(theta), :input_dim] = x
            out[theta_list.index(theta), input_dim:] = xdash
        np.save(fname, out)
    return out


def load_or_generate_hypersphere(num_points, input_dim, 
        fname='code/experiments/data/hypersphere.npy', force_new=True):
    """
    Choose two orthogonal vectors in input_dim space. Then, travel around the
    hypersphere in input_dim space on the circle defined by the two vectors, 
    returning num_points evenly spaced points on this circle.

    Args:
        num_points (int): number of uniformly spaced points on the hypersphere.
        input_dim (int): Dimensionality of x and x'
        force_new (bool): Force generation of a new dataset
    Returns:
        (num_points, input_dim) nparray
    """
    try:
        if force_new:
            raise Exception
        out = np.load(fname)
    except:

        A = np.random.normal(size=(input_dim, 2))
        # q contains the two orthogonal vectors
        q, _ = np.linalg.qr(A)
 
        theta_list = np.linspace(10e-3, 2*np.pi-10e-3, num_points)
        out = np.empty((num_points, input_dim))
        theta_list = list(theta_list)

        for theta in theta_list:
            v = np.asarray([[np.cos(theta)], [np.sin(theta)]])
            x       = np.transpose(np.matmul(q,v))
            out[theta_list.index(theta), :] = x
        np.save(fname, out)
    return out

def load_or_generate_xor(input_dim=2, num_points = None,
        fname='code/experiments/data/xor', force_new=True):
    """
    Generate every permutation of (+/-1, ... +/-1) of size input_dim then
    XOR problem: take the XOR of every element of x as the target.

    Args:
        input_dim (int): Dimensionality of x
        force_new (bool): Force generation of a new dataset
        num_points (int or None): Number of datapoints to use. If None,
            use all 2**input_dim possible points.
    Returns:
        X - (num_points, input_dim) nparray
        Y - (num_points, 1) nparray
    """
    try:
        if force_new:
            raise Exception
        X = np.load(fname + '_X.npy')
        Y = np.load(fname + '_Y.npy')
    except:
        X = np.asarray(list(itertools.product([1, -1], repeat=input_dim)))
        Y = np.reshape(-X[:,0]*X[:,1], (2**input_dim, 1))
        if num_points is None:
            num_points = 2**input_dim
        X = X[:num_points, :]
        Y = Y[:num_points, :]

    np.save(fname + '_X.npy', X)
    np.save(fname + '_Y.npy', Y)
    return [X, Y]

def load_or_generate_smooth_xor(train_size=4, test_size=100,
        fname='code/experiments/data/smooth_xor', force_new=True,
        noise_var = 0.01):
    """
    Generate every permutation of (+/-1, ... +/-1) of size input_dim then
    XOR problem: take the XOR of every element of x as the target.

    Args:
        train_size (int): number of training examples
        test_size (int): number of testing examples
    Returns:
        X_train (nparray): (train_size, 1)
        Y_train (nparray): (train_size, 1)
        X_test (nparray): (test_size, 1)
        Y_test (nparray): (test_size, 1)
    """
    assert train_size == 4
    try:
        if force_new:
            raise Exception
        X_train = np.load(fname + '_X.npy')
        Y_train = np.load(fname + '_Y.npy')
        X_test = np.load(fname + '_Xtest.npy')
        Y_test = np.load(fname + '_Ytest.npy')
    except:
        f = lambda x: -x[:,0] * x[:,1] * np.exp(2-x[:,0]**2-x[:,1]**2) + \
                np.sqrt(noise_var)*np.random.normal(size=(x.shape[0],))
        X_train = np.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        Y_train = np.reshape(f(X_train), (train_size, 1))
        X_test = np.random.uniform(-2, 2, (test_size, 2))
        Y_test = np.reshape(f(X_test), (test_size, 1))

    np.save(fname + '_X.npy', X_train)
    np.save(fname + '_Y.npy', Y_train)
    np.save(fname + '_Xtest.npy', X_test)
    np.save(fname + '_Ytest.npy', Y_test)

    return [X_train, Y_train, X_test, Y_test]

def load_or_generate_snelson(fname='code/experiments/data/'):
    """
    Snelson dataset from http://www.gatsby.ucl.ac.uk/~snelson/SPGP_dist.zip

    Args:

    Returns:
        X       - (num_points, 1) nparray
        Y       - (num_points, 1) nparray
        X_star  - (num_test_points, 1) nparray
    """
    X = np.genfromtxt(fname + 'snelson_train_inputs')
    Y = np.genfromtxt(fname + 'snelson_train_outputs')
    #X_star = np.fromfile(fname + 'snelson_test_inputs')

    idx = np.argsort(X, axis=0)
    X = np.reshape(X[idx], (-1, 1))
    Y = np.reshape(Y[idx], (-1, 1))

    try:
        if force_new:
            raise Exception
        X_train = np.load(fname + 'snelson_X.npy')
        Y_train = np.load(fname + 'snelson_Y.npy')
        X_test = np.load(fname + 'snelson_Xtest.npy')
        Y_test = np.load(fname + 'snelson_Ytest.npy')
    except:
        train_idx = np.linspace(0, X.shape[0], 11, endpoint=False).astype(int)
        train_idx = train_idx[1:]
        X_train = X[train_idx,:]
        Y_train = Y[train_idx,:]
        X_test = X
        Y_test = Y

        meanx = np.mean(X_train); meany = np.mean(Y_train);
        stdx = np.std(X_train); stdy = np.std(Y_train)

        X_train = (X_train - meanx)/stdx
        Y_train = (Y_train - meany)/stdy
        X_test = (X_test - meanx)/stdx
        Y_test = (Y_test - meany)/stdy

        idx = np.argsort(X_test, axis=0)
        X_test = np.reshape(X_test[idx], (-1, 1))
        Y_test = np.reshape(Y_test[idx], (-1, 1))


    np.save(fname + 'snelson_X.npy', X_train)
    np.save(fname + 'snelson_Y.npy', Y_train)
    np.save(fname + 'snelson_Xtest.npy', X_test)
    np.save(fname + 'snelson_Ytest.npy', Y_test)
    #X_star = np.reshape(X_star, (-1, 1))

    return [X_train, Y_train, X_test, Y_test]

def load_or_generate_1d_regression(train_size=10, test_size=100,
        fname='code/experiments/data/', force_new=True, noise_var=0.1):
    """
    Toy 1D regression dataset.

    Args:
        train_size (int): number of training examples
        test_size (int): number of testing examples
    Returns:
        X_train (nparray): (train_size, 1)
        Y_train (nparray): (train_size, 1)
        X_test (nparray): (test_size, 1)
        Y_test (nparray): (test_size, 1)
    """
    try:
        if force_new:
            raise Exception
        X_train = np.load(fname + 'regression_train_x.npy')
        Y_train = np.load(fname + 'regression_train_y.npy')
        X_test = np.load(fname + 'regression_test_x.npy')
        Y_test = np.load(fname + 'regression_test_y.npy')
    except:
        f = lambda x: np.sin(2*np.pi*x/np.sqrt(3)) + \
                np.sqrt(noise_var)*np.random.normal(size=x.shape)
        X_train = np.reshape(np.linspace(-np.sqrt(3), np.sqrt(3), 
            train_size), (train_size, 1))
        Y_train = f(X_train)
        X_test = np.random.uniform(-np.sqrt(3), np.sqrt(3), test_size)
        X_test = np.reshape(np.sort(X_test, None), (test_size, 1))
        Y_test = f(X_test)
    
    np.save(fname + 'regression_train_x.npy', X_train)
    np.save(fname + 'regression_train_y.npy', Y_train)
    np.save(fname + 'regression_test_x.npy', X_test)
    np.save(fname + 'regression_test_y.npy', Y_test)

    return [X_train, Y_train, X_test, Y_test]


    


