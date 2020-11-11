import tensorflow as tf
import numpy as np
import time
from math import exp, log
import sys

class ninn2d:

    # Initialize the class
    def __init__(self, X, u, layers, lb, ub, options, weights, biases, from_scratch):

        # Options
        self.last_layer_act_func = options['last_layer_act_func']

        # Boundaries (in case of ODEs in time, initial and final time)
        self.lb = lb
        self.ub = ub

        # Experimental data data (positions and values)
        self.x1 = X[:,0:1]
        self.x2 = X[:,1:2]
        self.u = u

        # Number of layers
        self.layers = layers

        # Initialize NNs
        if from_scratch == True:
            self.weights, self.biases = self.initialize_NN(layers)
        else:
            self.weights, self.biases = weights, biases

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True) )

        # Define placeholders: experimental data (positions and values)
        self.x1_tf = tf.placeholder(tf.float32, shape=[None, self.x1.shape[1]])
        self.x2_tf = tf.placeholder(tf.float32, shape=[None, self.x2.shape[1]])
        self.u_tf  = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        # Neural networks: predict solution and residuals
        self.u_pred = self.net_u(self.x1_tf, self.x2_tf)

        # Define the loss function as sum of 2 contributions (prediction of real data and residual errors)
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))

        # Options
        options_LBFGSB = {  'maxiter': 10000,
                            'maxfun': 10000,
                            'maxcor': 50,
                            'maxls': 50,
                            'ftol' : 1.0 * np.finfo(float).eps,
                            'gtol': 1.e-9
                        }

        # Define the optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=None,
                                                                equalities=None,
                                                                inequalities=None,
                                                                var_to_bounds=None,
                                                                method = 'L-BFGS-B', options = options_LBFGSB)


        # Define Adam optimizer
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # Assembling
        init = tf.global_variables_initializer()
        self.sess.run(init)


    # Initialize the weights in the layers
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases


    # Xavier initialization of weights
    def xavier_init(self, size):
        in_dim = size[0]   # number of source neurons
        out_dim = size[1]  # number of destination neurons
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)


    # Neural net
    def neural_net(self, X, weights, biases):

        # Number of layers
        num_layers = len(weights) + 1

        # Normalize input values (-1:1)
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0

        # Calculations over the hidden layers
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        # Return output value (no activation function)
        # Return output value (no activation function)
        W = weights[-1]
        b = biases[-1]
        if (self.last_layer_act_func == 'Linear'):
            Y = tf.add(tf.matmul(H, W), b)
        elif (self.last_layer_act_func == 'Sigmoid'):
            Y = tf.sigmoid(tf.add(tf.matmul(H, W), b))
        return Y

    # Prediction of main variables
    def net_u(self, x1, x2):
        u = self.neural_net(tf.concat([x1, x2],1), self.weights, self.biases)
        return u

    # Callback function
    # This function is called every time the loss and gradients are computed
    def callback(self, loss):
        print("Loss: %e" % loss )

    # Training function
    # This is the main function called to solver the identification problem
    def train(self,nIter):

        # The feed dictionary to be passed to the optimizer
        tf_dict = {self.x1_tf: self.x1, self.x2_tf: self.x2, self.u_tf: self.u}

        # The Adam optimizer is called only in case nIter~=0
        start_time = time.time()
        for it in range(nIter):

            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss = self.sess.run(self.loss, tf_dict)

                print("Iteration: ", it, " Loss: ", loss)

                start_time = time.time()


        # Minimize
        # feed_dict: A feed dict to be passed to calls to session.run
        # fetches: A list of Tensors to fetch and supply to loss_callback as positional arguments.
        # loss_callback: A function to be called every time the loss and gradients are computed,
        #                with evaluated fetches supplied as positional arguments.

        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)


    # Predict
    def predict(self, X_star):

        tf_dict = {self.x1_tf: X_star[:,0:1], self.x2_tf: X_star[:,1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)

        return u_star
