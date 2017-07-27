###################
# A multilayer perceptron (neural network) implementation
# Good to use for understanding how neural nets works.
# Author: John Berroa
###################

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from testdata import Data
from scipy.special import expit as sigmoid
from numpy.random import multivariate_normal as multNorm

#TODO: Allow for custom activation function
#TODO: Error functions?
#TODO: Momentum?
#TODO: "Take the result with the best training or validation performance" so have an optioj to train it multiple times

class MultiLayerPerceptron:
    """
    Creates an MLP with customizable parameters.
    Stores weight types, epsilons (learning rates), and activation functions for each layer so that they can be
    different for experimentation
    """

    def __init__(self, global_activation, global_epsilon, global_weight_type, debug=False):
        if debug:
            np.random.seed(777)
        self.global_weight_type = self._init_weights(global_weight_type)
        self.global_epsilon = global_epsilon
        self.global_activation_func = self._init_activation(global_activation)

        # Description of the network
        self.layer_weights = []
        self.layer_weight_types = []
        self.layer_activation_funcs = []  # this is kept as a string for ease of reading
        self.layer_epsilons = []
        self.layer_sizes = []

        # Variables needed for training
        self.layer_sums = []
        self.layer_outputs = []

        # Recorded variables for analysis
        self.errors = []
        self.epsilon_over_time = []


    def __str__(self):
        """
        Gives a summary of the structure of the network
        """
        if len(self.layer_sizes) == 0:
            string = "Multilayer Perceptron not yet built, therefore unable to print structure.  Use 'create_layer'."
        else:
            string = "Details of the Multilayer Perceptron:\n" \
                     " Layers: {}\n" \
                     " Structural overview: {} (# neurons/layer)\n" \
                     " Input dimensionality: {}\n" \
                     " Global defaults:\n" \
                     "    Weight type: {}\n" \
                     "    Epsilon: {}\n" \
                     "    Activation function: {}\n" \
                     " Layer settings:\n".format(len(self.layer_sizes), self.layer_sizes, self.layer_weights[0].shape[0]-1,
                                                self.global_weight_type, self.global_epsilon, self.global_activation_func)
            for layer in range(len(self.layer_sizes)):
                layerstring = "   Layer: {}\n" \
                              "    Number of neurons: {}\n" \
                              "    Weight type: {}\n" \
                              "    Epsilon: {}\n" \
                              "    Activation function: {}\n".format(layer+1, self.layer_sizes[layer],
                                                                     self.layer_weight_types[layer], self.layer_epsilons[layer],
                                                                     self.layer_activation_funcs[layer])
                string = string + layerstring

            laststring = "Note: if the class is printed before the network is fully created, it will print " \
                         "whatever has been built thus far."
            string = string + laststring
        return string


    def _init_weights(self, w):
        """
        Checks if initialized global weight type is valid; creating weights is the _create_weights function
        :param w:
        :return desired weight initialization:
        """
        possible_weights = ['normal', 'trunc', 'ones', 'zeros', 'uniform']
        if w not in possible_weights:
            raise ValueError("Invalid global weight type.  "
                             "Input: '{}'; Required: 'normal','trunc','ones',zeros', or 'uniform'.".format(w))
        else:
            return w


    def _create_weights(self, w, dim_in, size):
        """
        Creates a weight matrix based on the input dimensionality and the number of neurons.  Adds bias dimension
        :param w:
        :param dim_in:
        :param size:
        :return weight matrix:
        """
        if w == 'default':  # if default, reenter function with default weight name type
            return self._create_weights(self.global_weight_type, dim_in, size)
        # +1 because of adding a bias
        elif w == 'normal':
            return np.random.normal(size=(dim_in + 1, size))
        elif w == 'trunc':
            return stats.truncnorm.rvs(-1,1,size=(dim_in + 1, size))
        elif w == 'ones':
            return np.ones((dim_in + 1, size))
        elif w == 'zeros':
            return np.zeros((dim_in + 1, size))
        elif w == 'uniform':
            return np.random.uniform(-1,1,(dim_in + 1, size))
        else:
            raise ValueError("Invalid weight initilization type.  "
                             "Input: '{}'; Required: 'normal','trunc','ones',zeros', or 'uniform'.".format(w))


    def _init_activation(self, a):
        """
        Declares the activation function and the derivative of it
        :param a:
        :return desired function:
        """
        possible_activations = ['sigmoid', 'tanh', 'linear', 'relu']
        if a not in possible_activations:
            raise ValueError("Invalid global activation function.  "
                             "Input: '{}'; Required: 'sigmoid','tanh','linear', or 'relu'.".format(a))
        else:
            return a


    def _get_activation_func(self, request):
        """
        Retrieves a specific activation function if desired, or returns the global one if not defined
        :param request:
        :return activation function:
        """
        if request == 'default':
            return self.global_activation_func
        elif request == 'sigmoid':
            return sigmoid
        elif request == 'tanh':
            return np.tanh
        elif request == 'linear':
            return lambda x: x
        elif request == 'relu':
            return lambda x: x if x > 0 else 0
        else:
            raise ValueError("Invalid activation function.  "
                             "Input: '{}'; Required: 'sigmoid','tanh','linear', or 'relu'.".format(request))


    def _get_derivative(self, func):
        """
        Returns the derivative of the sent in activation function, or the derivative of the error function
        :param func:
        :return func's derivative:
        """
        if func == 'sigmoid':
            return lambda out: out * (1 - out)
        elif func == 'tanh':
            return lambda out: 1 - out**2
        elif func == 'linear':
            return 1
        elif func == 'relu':
            return lambda out: 1 if out > 0 else 0
        elif func == 'error':
            return lambda target, output: target - output
        else:
            raise ValueError("Invalid activation function to generate derivative.  "
                             "Input: '{}'; Required: 'sigmoid','tanh','linear', or 'relu'.".format(func))


    def _get_epsilon(self, request):
        """
        Returns a different epsilon if desired
        :param request:
        :return epsilon:
        """
        if request == 'default':
            return self.global_epsilon
        else:
            return request  # because it will be a number in this case


    def _add_bias(self, v):
        """
        Takes in an input vector, adds a bias of 1 to the front of it, and then expands dimensions to avoid:
        (2,) vs. (2,1)
        :param v:
        :return vector with bias and proper dimensionality:
        """
        v = np.append(1, v)
        try:
            _ = v.shape[1]
        except:
            v = np.expand_dims(v, axis=1)
        return v


    def _epsilon_decay(self):
        raise NotImplementedError


    def create_layer(self, size, dim_in=None, activation='default', weight_type='default', epsilon='default'):
        epsilon = self._get_epsilon(epsilon)
        if len(self.layer_weights) == 0:  # generate the weight matrix; if first layer, make sure there is input dimensionality provided
            if dim_in == None:
                raise ValueError("You are creating the first layer of the network.  "
                                 "Please provide the input dimensionality with 'dim_in'")
            else:
                weights = self._create_weights(weight_type, dim_in, size)
        else:
            input_dimensionality = self.layer_sizes[-1]  # returns the number of outputs of the previous layer
            weights = self._create_weights(weight_type, input_dimensionality, size)

        # update layer information storage
        self.layer_weights.append(weights)
        self.layer_epsilons.append(epsilon)
        self.layer_sizes.append(size)
        if activation == 'default':
            self.layer_activation_funcs.append(self.global_activation_func)
        else:
            self.layer_activation_funcs.append(activation)
        if weight_type == 'default':
            self.layer_weight_types.append(self.global_weight_type)
        else:
            self.layer_weight_types.append(weight_type)


    def _forward_step(self, layer, input):
        """
        Calculates the given layer's output given an input
        :param layer:
        :param input:
        :return layer output:
        """
        print("THE INPUT:\n", self._add_bias(input))
        print("THE WEIGHTS:\n", self.layer_weights[layer])
        sums = np.dot(self._add_bias(input).T, self.layer_weights[layer])
        print("THE SUMS:\n", sums)
        print("THE SUMS DIMS:\n",sums.shape)
        self.layer_sums.append(sums)  # used for the backprop step   MAY NOT BE USED????
        activation_function = self._get_activation_func(self.layer_activation_funcs[layer])
        output = []
        for s in sums.T:
            print("S:\n",s)
            print("Sflt:\n", float(s))
            output.append(activation_function(float(s)))
            print("OUTPUT TO GO IN:\n", output[-1])
        # output = np.array([activation_function(s) for s in sums.T])
        output = np.array(output)
        output = np.expand_dims(output, axis=1)
        print("OUTPUT SHAPE:\n", output.shape)
        print("OUTPUT:\n", output, "\n\n\n")
        self.layer_outputs.append(output)
        return output


    def predict(self, input):
        """
        Propagates an input through the network to get the network's result, without doing the backprop step.
        :param input:
        :return prediction:
        """
        for layer in range(len(self.layer_sizes)):
            input = self._forward_step(layer, input)
        prediction = input  # just for clarification sake
        return prediction


    def _calculate_error(self, output, target):
        # so this should be a sum, where it sums over one item in the stochastic case, because it'll be a [1], but in the batch case
        # it should be a list of numbers (inputs that were sent through) that it iterates through
        # error = np.mean([.5 * (target - sample) ** 2 for sample in output])
        # shouldn't need to be! the whole batch should be input at once
        error = .5 * (target - output)**2
        self.errors.append(error)
        return error


    def _backpropagate(self, input, output, target):
        # if layer == len(self.layer_sums) - 1:  # because of 0 indexing
        #     delta = np.multiply(-(YYYY - TARGET), DERIVATIVE LAST LAYER(unactivated output of current layer))
        #     gradient = np.dot(activated output of previous layer.T, delta)
        # else:
        #     delta = np.dot(FIRST GRADIET, W2.T) * unacvitved output current layer layer )
        #     gradient = input.t, delta
        #
        error = self._calculate_error(output, target)
        temp_outputs = np.append(input, self.layer_outputs)
        for layer in reversed(range(len(self.layer_sizes))):
            print("BACKPROP LAYER:",layer)
            input = temp_outputs[layer - 1]
            output = temp_outputs[layer]
            activation_function = self._get_activation_func(self.layer_activation_funcs[layer])
            delta = np.array([activation_function(o) * error for o in output])
            delta = np.expand_dims(delta, axis=1)
            print("DELTA:\n",delta)
            print("DELTAshape:\n",delta.shape)
            error = np.dot(self.layer_weights[layer].T, delta)[1:]  # why this?[:1]
            self.descend_gradient(layer, delta, input)


    def descend_gradient(self, layer, delta, input):
        gradient = self.layer_epsilons[layer] * delta * np.append(1, input)  # WHY IS THIS A PLUS?
        print("GRADIENT:\n",gradient)
        print("GRADIENTshape:\n",gradient.shape)
        print(self.layer_weights[layer])
        print(self.layer_weights[layer].shape)
        self.layer_weights[layer] = self.layer_weights[layer] + gradient
        # self.layer_weights[layer]








if __name__ == "__main__":
    inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    outputs = np.array([[0],[1],[1],[0]])
    NeuralNet = MultiLayerPerceptron('relu',.001,'normal')
    NeuralNet.create_layer(3, 3)
    NeuralNet.create_layer(2)
    NeuralNet.create_layer(1)
    # print(NeuralNet)
    print("DEBUG")
    print(NeuralNet.layer_weights[0])
    input = inputs
    for l in range(len(NeuralNet.layer_weights)):
        print("Stepping forward for layer:",l+1)
        input = NeuralNet._forward_step(l, input)
    # print(NeuralNet.layer_outputs)

    # output = [1]
    NeuralNet._backpropagate(inputs, input, outputs)
    print(NeuralNet.layer_weights[0])

    # print(NeuralNet.layer_weight_types)
    # print(NeuralNet.global_activation_func)
    # print(NeuralNet.layer_activation_funcs)


'''
So I need to make it impervious to batch size.  It should always work out through the matrix math
https://iamtrask.github.io/2015/07/12/basic-python-network/
'''