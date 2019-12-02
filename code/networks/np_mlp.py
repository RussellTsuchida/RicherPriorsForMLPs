import numpy as np
from .np_initialisers import create_initialiser

class Mlp(object):
    def __init__(self, layer_widths, layer_activations, weight_init=None, 
            bias_init=None, standard_first = False):
        """
        Args:
            layer_widths (list(int)):[input_size,width1,...,widthn,output_size]
            layer_activations (list(tf.nn.activation)):List of activations,
                of length 1 less than layer_widths
            weight_init (tf.keras.initializers.Initializer): Initializer to be 
                used for the weights. If None, use a random normal with variance
                2/n_in.
            bias_init (tf.keras.initializers.Initializer): Initializer to be 
                used for the biases. If None, use a random normal with unit 
                variance.
        """
        self._check_specifications(layer_widths, layer_activations, 
                weight_init, bias_init, standard_first)
        self._make_layers()

    def _check_specifications(self, layer_widths, layer_activations, 
            weight_init, bias_init, standard_first):
        """
        Check constructor inputs and set attributes.
        """
        assert len(layer_widths)-1 == len(layer_activations) 

        self.layer_widths       = layer_widths
        self.layer_activations  = layer_activations

        if weight_init is None:
            weight_init = create_initialiser(lambda A, B, C, D: np.sqrt(2)*D, 
                    lambda A, B, C: 0)
        if bias_init is None:
            bias_init = create_initialiser(lambda A, B, C, D: np.sqrt(2)*D, 
                    lambda A, B, C: 0)
        self.weight_init        = weight_init
        self.bias_init          = bias_init
        self.standard_first     = standard_first

    def _make_layers(self):
        """
        Make each layer in the MLP and add it to self.model.
        """
        self.layer_weights = [0]*len(self.layer_widths[1:])
        self.layer_biases = [0]*len(self.layer_widths[1:])
        l = 1
        for width in self.layer_widths[1:]:
            if (l == 1):
                if (self.standard_first):
                    w_init = lambda x: np.random.normal(0, 1, x)
                    b_init = lambda x: np.zeros((x))
                else:
                    w_init = create_initialiser(self.weight_init._F, 
                            self.weight_init._expected_over_D_of_F,
                            apply_scale=False, 
                            use_normal_ABCD = self.weight_init.use_normal_ABCD)
                    b_init = create_initialiser(self.bias_init._F, 
                            self.bias_init._expected_over_D_of_F,
                            apply_scale=False, 
                            use_normal_ABCD = self.bias_init.use_normal_ABCD)
            else:
                w_init = self.weight_init
                b_init = self.bias_init
            if (l == len(self.layer_widths)):
                scale_w = 1./np.sqrt(self.layer_widths[l-1])
                w_init = lambda x: np.random.normal(0, scale_w, x)
                b_init = lambda x: np.zeros((x))

            input_shape = (self.layer_widths[l-1],)
            activation = self.layer_activations[l-1]

            self.layer_weights[l-1] = \
            w_init((self.layer_widths[l],self.layer_widths[l-1]))
            self.layer_biases[l-1] = \
            b_init((self.layer_widths[l], 1))

            l += 1

    def layer_fun(self, layer_input, all_layers=True):
        """
        Args:
            layer_input
            all_layers (Bool): True to output all layers as a list, False to 
                only output the last layer as a nparray.
        Returns:
            list of output signals at each hidden layer
        """
        if all_layers:
            outputs = [0]*len(self.layer_widths[1:])
        for l in range(len(self.layer_widths[1:])):
            weights = self.layer_weights[l]
            biases = self.layer_biases[l]
            
            print(weights.shape)
            print(biases.shape)
            print(layer_input.shape)

            layer_input = \
            self.layer_activations[l](np.matmul(weights, layer_input) + biases)

            if all_layers:
                outputs[l] = layer_input
        
        if not all_layers:
            outputs = layer_input

        return outputs


    def sample_functions(self, x, n=10):
        """
        Sample n random function evaluations at x from the network. 
        """
        outputs = np.empty((n, x.shape[1]))
        for iteration in range(n):
            if iteration != 0:
                self._make_layers()
            fx = self.layer_fun(x, all_layers=False)
            outputs[iteration, :] = np.reshape(fx, (-1, ))

        return outputs

    def _sample_x_xdash(self, theta):
        """
        Sample two random points, x and x' from mutual random orthogonal 
        transformations of (1,0) and (cos(theta), sin(theta)), where theta
        is the angle between x and xdash.
        
        Args:
            theta (float): Angle required between x and xdash
        Returns:
            x (np.array(float)): first point of size (self.layer_widths[0], 1)
            xdash (np.array(float)): second point (self.layer_widths[0], 1)
        """
        u = np.asarray([[1], [0]])
        v = np.asarray([[np.cos(theta)], [np.sin(theta)]])

        A = np.random.uniform(size=(self.layer_widths[0], 2))
        q, _ = np.linalg.qr(A)

        x       = np.transpose(np.matmul(q,u))
        xdash   = np.transpose(np.matmul(q,v))

        return x, xdash

    def empirical_normalised_kernel_given_data(self, data, 
            layers=[1,2,4,8,16,32]):
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
        layers = [l for l in layers if l<=len(self.layer_activations)-1]

        # Feed data through the network
        outputs = np.empty((len(layers), data.shape[0]))

        x       = data[:,:self.layer_widths[0]]
        xdash   = data[:,self.layer_widths[0]:]

        layer_outs_x        = self.layer_fun(x.T)
        layer_outs_xdash    = self.layer_fun(xdash.T)
        
        layer = 1
        for fx, fx_dash in zip(layer_outs_x, layer_outs_xdash):
            if layer in layers:
                dot = np.sum(fx*fx_dash, axis=0)
                
                cos_theta_hid = dot/\
                    (np.linalg.norm(fx, axis=0)*np.linalg.norm(fx_dash, axis=0))
                outputs[layers.index(layer),:]=\
                        cos_theta_hid
            layer += 1
        return outputs

    def reduce_effective_width(self, n_zero):
        """
        Reduce the effective width of the network.

        Args:
            n_zero (int): the number of units to remove form each layer.
        """
        num_in  = self.layer_widths[0]
        num_out = self.layer_widths[-1]

        self.layer_widths = [width - n_zero for width in self.layer_widths]

        self.layer_widths[0]    = num_in
        self.layer_widths[-1]   = num_out

        self._make_layers()
