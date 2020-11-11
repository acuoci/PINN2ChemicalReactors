import tensorflow as tf
import numpy as np
import time
from math import exp, log
import sys

class pinn3d:

    # Initialize the class
    # X,u = experimental data (position and values)
    # layers = number of layers
    # lb,ub = min/max values of independent variables

    def __init__(self, problem, X, u, layers, lb, ub, Ymin, Ymax, options, teta_guess, kappa_min, kappa_max,  weights, biases, from_scratch, reactor):

        # Options
        self.problem = problem
        self.constrained = options['constrained']
        self.last_layer_act_func = options['last_layer_act_func']

        # Boundaries (in case of ODEs in time, initial and final time)
        self.lb = lb
        self.ub = ub

        # Scaling factors
        self.Ymin = Ymin
        self.Ymax = Ymax
        self.scaling_output_variables = options['scaling_output_variables']
        self.scaling_residuals = options['scaling_residuals']

        # Experimental data data (positions and values)
        self.x1 = X[:,0:1]
        self.x2 = X[:,1:2]
        self.x3 = X[:,2:3]
        self.u = u

        # Number of layers
        self.layers = layers

        # [PROBLEM DATA] Constant physical data
        self.reactor = reactor

        # Weights
        self.weight_exp = options['weight_exp']
        self.weight_phy = options['weight_phy']

        # Range
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

        # Initialize NNs
        if from_scratch == True:
            self.weights, self.biases = self.initialize_NN(layers)
        else:
            self.weights, self.biases = weights, biases

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True) )

        # Initialize parameters
        self.teta = tf.Variable(teta_guess, dtype=tf.float32)

        # Define placeholders: experimental data (positions and values)
        self.x1_tf = tf.placeholder(tf.float32, shape=[None, self.x1.shape[1]])
        self.x2_tf = tf.placeholder(tf.float32, shape=[None, self.x2.shape[1]])
        self.x3_tf = tf.placeholder(tf.float32, shape=[None, self.x3.shape[1]])
        self.u_tf  = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        # Neural networks: predict solution and residuals
        self.u_pred = self.net_u(self.x1_tf, self.x2_tf, self.x3_tf)
        self.f_pred = self.net_f(self.x1_tf, self.x2_tf, self.x3_tf)

        # Define the loss function as sum of 2 contributions (prediction of real data and residual errors)
        self.loss = self.weight_exp*tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    self.weight_phy*self.f_pred

        # Define the optimizer
        if self.constrained == 'Linear':
        	teta_min = np.zeros(self.kappa_min.shape)
        	teta_max = np.ones(self.kappa_max.shape)
        else:
        	teta_min = self.kappa_min
        	teta_max = self.kappa_max

        # Options
        options_LBFGSB = {	'maxiter': 50000,
                            'maxfun': 50000,
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
                                                                var_to_bounds={self.teta: ( teta_min, teta_max)},
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
        W = weights[-1]
        b = biases[-1]
        if (self.last_layer_act_func == 'Linear'):
            Y = tf.add(tf.matmul(H, W), b)
        elif (self.last_layer_act_func == 'Sigmoid'):
            Y = tf.sigmoid(tf.add(tf.matmul(H, W), b))
        return Y


    # Prediction of main variables
    def net_u(self, x1, x2, x3):
        u = self.neural_net(tf.concat([x1,x2,x3],1), self.weights, self.biases)
        return u


    # Callback function
    # This function is called every time the loss and gradients are computed
    def callback(self, loss, teta):

        if self.constrained == 'Linear':
        	kappa = self.kappa_min + (self.kappa_max - self.kappa_min)*teta
        elif self.constrained == 'Exp':
        	kappa = np.exp(teta)
        elif self.constrained == 'None':
        	kappa = teta

        print("Loss: ", loss, " kappa: ", kappa )


    # Training function
    # This is the main function called to solver the identification problem
    def train(self,nIter):

        # The feed dictionary to be passed to the optimizer
        tf_dict = {self.x1_tf: self.x1, self.x2_tf: self.x2, self.x3_tf: self.x3, self.u_tf: self.u}

        # The Adam optimizer is called only in case nIter~=0
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:

                elapsed = time.time() - start_time

                loss = self.sess.run(self.loss, tf_dict)

                teta = self.sess.run(self.teta)
                if self.constrained == 'Linear':
                    kappa = self.kappa_min + (self.kappa_max - self.kappa_min)*teta
                elif self.constrained == 'Exp':
                    kappa = np.exp(teta)
                elif self.constrained == 'None':
                    kappa = teta

                print("It: ", it, " Loss: ", loss, " Time: ", time, " Kappa: ", kappa)
                start_time = time.time()


        # Minimize
        # feed_dict: A feed dict to be passed to calls to session.run
        # fetches: A list of Tensors to fetch and supply to loss_callback as positional arguments.
        # loss_callback: A function to be called every time the loss and gradients are computed,
        #                with evaluated fetches supplied as positional arguments.

        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.teta],
                                loss_callback = self.callback)


    # Predict
    def predict(self, X_star):

        tf_dict = {self.x1_tf: X_star[:,0:1], self.x2_tf: X_star[:,1:2], self.x3_tf: X_star[:,2:3]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)

        return u_star, f_star


    # Prediction of residuals (Physics-Guided Network)
    def net_f(self, x1, x2, x3):

        if (self.problem == 'AcetoAcetylation'):
            return self.net_f_AcetoAcetylation(x1, x2, x3)
        elif (self.problem == 'TolueneHydrogenation'):
            return self.net_f_TolueneHydrogenation(x1)
        elif (self.problem == 'AllylChlorination'):
            return self.net_f_AllylChlorination(x1, x2, x3)


    def net_f_AcetoAcetylation(self, t, Qin, Ccat):

        # Parameter
        if self.constrained == 'Linear':
        	kappa = self.kappa_min + (self.kappa_max - self.kappa_min)*self.teta
        elif self.constrained == 'Exp':
        	kappa = tf.exp(self.teta)
        elif self.constrained == 'None':
        	kappa = self.teta

        # Predicted variable (from the first neural network)
        utot = self.net_u(t,Qin,Ccat)
        if (self.scaling_output_variables == False):
            CA = utot[:,0:1]
            CB = utot[:,1:2]
            CC = utot[:,2:3]
            CD = utot[:,3:4]
            CE = utot[:,4:5]
        else:
            CA = utot[:,0:1]*(self.Ymax[0]-self.Ymin[0])+self.Ymin[0]
            CB = utot[:,1:2]*(self.Ymax[1]-self.Ymin[1])+self.Ymin[1]
            CC = utot[:,2:3]*(self.Ymax[2]-self.Ymin[2])+self.Ymin[2]
            CD = utot[:,3:4]*(self.Ymax[3]-self.Ymin[3])+self.Ymin[3]
            CE = utot[:,4:5]*(self.Ymax[4]-self.Ymin[4])+self.Ymin[4]

        # Derivatives
        CA_t = tf.gradients(CA, t)[0]
        CB_t = tf.gradients(CB, t)[0]
        CC_t = tf.gradients(CC, t)[0]
        CD_t = tf.gradients(CD, t)[0]
        CE_t = tf.gradients(CE, t)[0]

        # Residence time
        tau = self.reactor['V']/Qin

        # Reaction rates
        r1 = kappa[0]*CA*CB*Ccat
        r2 = kappa[1]*CB*CB*Ccat
        r3 = kappa[2]*CB

        # Formation rates
        RA = -r1;
        RB = -r1-2*r2-r3
        RC = r1
        RD = r2
        RE = r3

        # Residual (from equation)
        f0 = CA_t - ( (self.reactor['Cin'][0]-CA)/tau + RA )
        f1 = CB_t - ( (self.reactor['Cin'][1]-CB)/tau + RB )
        f2 = CC_t - ( (self.reactor['Cin'][2]-CC)/tau + RC )
        f3 = CD_t - ( (self.reactor['Cin'][3]-CD)/tau + RD )
        f4 = CE_t - ( (self.reactor['Cin'][4]-CE)/tau + RE )

        fstack = tf.concat([f0,f1,f2,f3,f4], 1)

        ftot = tf.reduce_sum(tf.square(tf.divide(fstack, self.scaling_residuals)))

        return ftot


    def net_f_TolueneHydrogenation(self, t):

        # Parameter
        if self.constrained == 'Linear':
        	kappa = self.kappa_min + (self.kappa_max - self.kappa_min)*self.teta
        elif self.constrained == 'Exp':
        	kappa = tf.exp(self.teta)
        elif self.constrained == 'None':
        	kappa = self.teta

        # Kinetic parameters
        kH1 = kappa[0]
        kD1 = kappa[1]
        k2 = kappa[2]
        KrelA = kappa[3]
        KrelB = 1.
        KrelC = kappa[4]

        # Predicted variable (from the first neural network)
        utot = self.net_u(t)
        if (self.scaling_output_variables == False):
            CA = utot[:,0:1]
            CB = utot[:,1:2]
            CC = utot[:,2:3]
        else:
            CA = utot[:,0:1]*(self.Ymax[0]-self.Ymin[0])+self.Ymin[0]
            CB = utot[:,1:2]*(self.Ymax[1]-self.Ymin[1])+self.Ymin[1]
            CC = utot[:,2:3]*(self.Ymax[2]-self.Ymin[2])+self.Ymin[2]

        # Derivatives
        CA_t = tf.gradients(CA, t)[0]
        CB_t = tf.gradients(CB, t)[0]
        CC_t = tf.gradients(CC, t)[0]

        # Residence time
        denominator = KrelA*CA + KrelB*CB + KrelC*CC
        psiA = tf.multiply(KrelA,CA)
        psiB = tf.multiply(KrelB,CB)

        # Reaction rates
        rH1 = tf.multiply(kH1,psiA)
        rD1 = tf.multiply(kD1,psiB)
        r2  = tf.multiply(k2,psiB)

        # Formation rates
        RA = tf.add(-rH1,rD1);
        RC = r2
        RB = tf.add(-RA, -RC)

        # Residual (from equation)
        f0 = tf.multiply(CA_t,denominator) - RA
        f1 = tf.multiply(CB_t,denominator) - RB
        f2 = tf.multiply(CC_t,denominator) - RC

        fstack = tf.concat([f0,f1,f2], 1)

        ftot = tf.reduce_sum(tf.square(tf.divide(fstack, self.scaling_residuals)))

        return ftot


    def net_f_AllylChlorination(self, V, T, FtotIn):

        # Parameter
        if self.constrained == 'Linear':
            kappa = self.kappa_min + (self.kappa_max - self.kappa_min)*self.teta
        elif self.constrained == 'Exp':
            kappa = tf.exp(self.teta)
        elif self.constrained == 'None':
            kappa = self.teta

        # Predicted variable (from the first neural network)
        utot = self.net_u(V,T,FtotIn)
        FCl2 = utot[:,0:1]*(self.Ymax[0]-self.Ymin[0])+self.Ymin[0]
        FP   = utot[:,1:2]*(self.Ymax[1]-self.Ymin[1])+self.Ymin[1]
        FA   = utot[:,2:3]*(self.Ymax[2]-self.Ymin[2])+self.Ymin[2]
        FHCl = utot[:,3:4]*(self.Ymax[3]-self.Ymin[3])+self.Ymin[3]
        FD   = utot[:,4:5]*(self.Ymax[4]-self.Ymin[4])+self.Ymin[4]

        # Total flow rate
        Ftot = (FCl2 + FP + FA + FHCl + FD)

        # Derivatives
        FCl2_V = tf.gradients(FCl2,V)[0]
        FP_V   = tf.gradients(FP,V)[0]
        FA_V   = tf.gradients(FA,V)[0]
        FHCl_V = tf.gradients(FHCl,V)[0]
        FD_V   = tf.gradients(FD,V)[0]

        # Partial pressures
        pP = FP/Ftot*self.reactor['P']
        pCl2 = FCl2/Ftot*self.reactor['P']

        # Kinetic constants
        A1 = tf.exp(kappa[0])
        A2 = tf.exp(kappa[1])
        E1 = kappa[2]
        E2 = kappa[3]
        k1 = A1*tf.exp(-E1/8.314/T)
        k2 = A2*tf.exp(-E2/8.314/T)

        # Reaction rates
        r1 = k1*pP*pCl2
        r2 = k2*pP*pCl2

        # Formation rates
        RCl2 = -r1-r2
        RP   = -r1-r2
        RA   = r1
        RHCl = r1
        RD   = r2

        # Residual (from equation)
        f0 = FCl2_V - RCl2
        f1 = FP_V   - RP
        f2 = FA_V   - RA
        f3 = FHCl_V - RHCl
        f4 = FD_V   - RD

        fstack = tf.concat([f0,f1,f2,f3,f4], 1)

        ftot = tf.reduce_sum(tf.square(tf.divide(fstack, self.scaling_residuals)))

        return ftot
