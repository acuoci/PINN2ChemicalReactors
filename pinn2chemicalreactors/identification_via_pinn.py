import sys, getopt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Basic libraries
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

# Scipy
import scipy.io
from scipy.interpolate import griddata

# ODE integrator
from scipy.integrate import odeint

# Neural Network Models
from ninn1d import ninn1d
from pinn1d import pinn1d
from ninn2d import ninn2d
from pinn2d import pinn2d
from ninn3d import ninn3d
from pinn3d import pinn3d

# Reactor Physical Models
from reactor_aceto_acetylation import reactor_aceto_acetylation
from reactor_toluene_hydrogenation import reactor_toluene_hydrogenation
from reactor_allyl_chlorination import reactor_allyl_chlorination

# Tensorflow: compatibility option
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set the seed of random number generators for reproducibility
np.random.seed(1234)
tf.set_random_seed(1234)


def main(argv):
   inputfile = ''
   outputfile = ''
   datafile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:d:",["ifile=","ofile=","dfile="])
   except getopt.GetoptError:
      print('test.py -i <inputfile> -o <outputfile> -d <datafile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile> -d <datafile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      elif opt in ("-d", "--dfile"):
         datafile = arg
   print ('Input file is:  ', inputfile)
   print ('Output file is: ', outputfile)
   print ('Data file is:   ', datafile)
   
   return inputfile, outputfile, datafile
   
   

# Main program
if __name__ == "__main__":

#--------------------------------------------------------------------------------#
# Read user options from file                                                    #
#--------------------------------------------------------------------------------#
    
    # Running mode
    write_output_log = True
    
    # Input and output file names
    inputfile, outputfile, datafile = main(sys.argv[1:])
    
    # Read input file
    with open(inputfile) as finput:
        for line in finput:
            data = line.split()
            
            problem_from_input = data[0]
            policy_point_from_input = data[1]
            N_u_list_from_input = int(data[2])
            noise_sigma_from_input = float(data[3])
            weight_phy_from_input = float(data[4])
            number_internal_layers_from_input = int(data[5])
            neurons_from_input = int(data[6])
            
            break

#--------------------------------------------------------------------------------#
# Reactor data section                                                           #
#--------------------------------------------------------------------------------#

    # Problem name: AcetoAcetylation | TolueneHydrogenation | AllylChlorination
    problem = problem_from_input
    
    # Data file name
    if (datafile == ''):
        data_path = r"..\data\matlab"
        if (problem ==  'AcetoAcetylation'):
            exp_file_name = os.path.join(data_path, "aceto_acetylation.mat")
        elif (problem ==  'TolueneHydrogenation'):
            exp_file_name = os.path.join(data_path, "toluene_hydrogenation.mat")
        elif (problem ==  'AllylChlorination'):
            exp_file_name = os.path.join(data_path, "allyl_chlorination.mat")
    else:
        exp_file_name = datafile
        
    if (problem ==  'AcetoAcetylation'):

        # Kinetic constants: true values (used for checking the results only)
        kappa_true = np.array([ 0.053, 0.128, 0.028 ])

        # Kinetic constants: first-guess, min and max values (for +inf use np.infty)
        kappa_guess = np.array([ 0.053, 0.128, 0.028 ])
        kappa_min   = np.array([ 0.01,  0.01,  0.01 ])
        kappa_max   = np.array([ 1.00,  1.00,  1.00 ])

        # Reactor data
        reactor_data = {    'Cin':      [ 0., 6., 0., 0., 0. ],         # inlet concentration [mol/l]
                            'C0':       [0.30, 0.14, 0.08, 0.01, 0],    # initial concentrations [mol/l]
                            'V':        1,                              # volume [l]
                            'Ccat':     0.5,                            # catalyst concentration (constant value) [mol/l]
                            'Qin':      0.3,                            # volumetric flow rate [l/min]
                            'tau':      20.,                            # total time [min]
                            'labels':   ['A', 'B', 'C', 'D', 'E'] }     # labels

    elif (problem ==  'TolueneHydrogenation'):

        # Kinetic constants: true values (used for checking the results only)
        kappa_true = np.array([ 0.023, 0.005, 0.011, 1.9, 1.8 ])

        # Kinetic constants: first-gues, min and max values (for +inf use np.infty)
        kappa_guess = np.array([ 0.023, 0.005, 0.011, 1.90, 1.80 ])
        kappa_min   = np.array([ 0.001, 0.001, 0.001, 0.01, 0.01 ])
        kappa_max   = np.array([ 0.100, 0.100, 0.100, 10.0, 10.0 ])

        # Reactor data
        reactor_data = {    'Cin':      [ 1,0.,0. ],          # inlet concentration [mol/l]
                            'tau':      400.,                 # total time [min]
                            'labels':   ['A', 'B', 'C'] }     # labels


    elif (problem ==  'AllylChlorination'):

        # Kinetic constants: true values (used for checking the results only)
        # A = [mol/m3/s/Pa], E = [J/mol/K]
        kappa_true = np.array([ np.log(9.02e-5), np.log(5.12e-9), 63267.20, 15956.36 ])

        # Kinetic constants: first-guess, min and max values (for +inf use np.infty)
        kappa_guess = np.array([ np.log(9.02e-5), np.log(5.12e-9),  63267.20, 15956.36 ])
        kappa_min   = np.array([ np.log(9.02e-7), np.log(5.12e-11), 43267.20,  5956.36 ])
        kappa_max   = np.array([ np.log(9.02e-3), np.log(5.12e-7),  83267.20, 25956.36 ])

        # Reactor data
        reactor_data = {    'P':        2.02e5,                             # pressure [Pa]
                            'T':        573,                                # temperature [K]
                            'Ltot':     8,                                  # total length [m]
                            'di':       0.05,                               # diameter [m]
                            'Vtot':     3.14159265*0.05*0.05/4*8,           # total volume [m3]
                            'Ftot':     200./3600.,                         # total molar flow rate [mol/s]
                            'Xin':      np.array([1.,4.,0.,0.,0.])/5.,      # inlet molar fractions [-]
                            'labels':   ['Cl2', 'P', 'A', 'HCl', 'D'] }     # labels

#--------------------------------------------------------------------------------#
# Main options                                                                   #
#--------------------------------------------------------------------------------#

    options = {         'constrained':              'Linear',   # 'None' | 'Linear' | 'Exp'
                        'noise_sigma':              0.00,       # Noise level
                        'noise_repetitions':        1,          # Number of repetitions
                        'scaling_output_variables': True,       # Scaling output variables
                        'include_ics':              True,       # include intial points along x1, x2 and x3
                        'include_fcs':              True,       # include final points along x1, x2 and x3
                        'weight_exp':               1.0,        # weight of experimental data
                        'weight_phy':               1.0,        # weight of physical constraints
                        'scaling_residuals_type':   'Ones',     # Scaling policy for residuals: 'Ones' | 'Max' | 'Mean'
                        'scaling_residuals':        1,          # Scaling residuals
                        'policy_point':             'Random',   # Policy for choosing the points: 'Random' | 'Uniform' | 'UserDefined'
                        'last_layer_act_func':      'Linear',   # Activation function of last layer: 'Linear' | 'Sigmoid'
                        'pre_training':             False,      # Pre-training without physics
                        'extra_physical_points':    False
                }

    # Overwrite default values from file
    options['noise_sigma']  = noise_sigma_from_input
    options['weight_phy']   = weight_phy_from_input
    options['policy_point'] = policy_point_from_input

    # Preprocess parameter constraints
    if options['constrained'] == 'Linear':
    	teta_guess = (kappa_guess-kappa_min)/(kappa_max-kappa_min)
    	teta_true  = (kappa_true-kappa_min)/(kappa_max-kappa_min)
    elif options['constrained'] == 'Exp':
    	teta_guess = np.log(kappa_guess)
    	teta_true  = np.log(kappa_true)
    	kappa_min  = np.log(kappa_min)
    	kappa_max  = np.log(kappa_max)
    elif options['constrained'] == 'None':
    	teta_guess = kappa_guess
    	teta_true  = kappa_true

#--------------------------------------------------------------------------------#
# Read input data                                                                #
#--------------------------------------------------------------------------------#

    # Experimental/Synthetic data
    x1_name = 'x1'
    x2_name = 'x2'
    x3_name = 'x3'
    xO_name = 'X_overall'
    dataset_name = 'Y_overall'

    # Load (pseudo-)experimental data from MATLAB calculations
    data = scipy.io.loadmat(exp_file_name)

    # Clean data
    x1 = data[x1_name].flatten()[:,None]
    x2 = data[x2_name].flatten()[:,None]
    x3 = data[x3_name].flatten()[:,None]
    YComplete = np.real(data[dataset_name])
    ns = YComplete.shape[1]
    YClean = YComplete[:,0:ns]

    # Analyze data
    Ymax  = np.max(YClean,  axis=0)
    Ymin  = np.min(YClean,  axis=0)
    Ymean = np.mean(YClean, axis=0)
    print("Max values:  ", Ymax)
    print("Min values:  ", Ymin)
    print("Mean values: ", Ymean)

    # Problem size
    nx1 = len(x1)
    nx2 = len(x2)
    nx3 = len(x3)

    size_X = 3
    if (nx2 == 1 and nx3 == 1):
        size_X = 1
    if (nx2 != 1 and nx3 == 1):
        size_X = 2
    X_star = (np.real(data[xO_name]))[:,0:(size_X)]

    # Print on the screen
    print("Domain size:             %d %d %d" % (nx1,nx2,nx3) )
    print("Total number of entries: %d" % X_star.shape[0])
    print("Size of input layer:     %d" % X_star.shape[1])


#--------------------------------------------------------------------------------#
# PINN data section                                                              #
#--------------------------------------------------------------------------------#

    # Network topology
    layers = np.zeros(1+number_internal_layers_from_input+1, dtype=int)
    layers[0] = int(size_X)
    for ii in range(1+number_internal_layers_from_input):
        layers[ii+1] = int(neurons_from_input)
    layers[1+number_internal_layers_from_input] = int(ns)
    print("Neural network topology:", layers)

    # Number of supplementary points where to enforce physical laws (i.e. transport equations)
    N_u_list = [ N_u_list_from_input ]


#--------------------------------------------------------------------------------#
# Preprocessing section                                                          #
#--------------------------------------------------------------------------------#

    if (write_output_log == True):
        f_out = open(outputfile, "a+")

    for jj in range(options['noise_repetitions']) :

        # Generate noise
        delta_basis = np.random.normal(0, options['noise_sigma'], np.shape(data[dataset_name]))
        for kk in range(ns) :
            delta_basis[:,kk] *= Ymax[kk]
        Y = YClean + delta_basis

        # Organize data
        u_star = Y

        # Domain bounds (min and max independent coordinates)
        lb = X_star.min(0)
        ub = X_star.max(0)
        print('Minimum X: ', lb)
        print('Maximum X: ', ub)

        for j in range(len(N_u_list)) :

            N_u = int(N_u_list[j])

            # Policy1: Experimental data for training (randomly chosen + initial conditions)
            if (options['policy_point'] == 'Random'):
                idx_rnd = np.random.choice(X_star.shape[0], N_u, replace=False)
                idx_ics = np.zeros(0, dtype=int);
                idx_fcs = np.zeros(0, dtype=int);
                if options['include_fcs'] == True :
                    idx_fcs = np.arange(nx1-1,nx1*nx2*nx3,nx1)
                if options['include_ics'] == True :
                    idx_ics = np.arange(0,nx1*nx2*nx3,nx1)
                idx_tot = np.r_[idx_ics, idx_rnd, idx_fcs]
                idx_tot = np.unique(idx_tot)

            elif (options['policy_point'] == 'Uniform'):
                step = np.floor(nx1/N_u)
                idx_tot = np.zeros(0, dtype=int);
                for j in range(nx3):
                    for i in range(nx2):
                        idx = np.arange(0,nx1,step, dtype=int)
                        shift = nx1*i + nx1*nx2*j
                        idx += shift
                        idx_tot = np.r_[idx_tot, idx]
                idx_tot = np.unique(idx_tot)

            # Print selected indices
            print('Selected points (indices):', idx_tot)

            # Training points and true values
            X_u_train = X_star[idx_tot]
            u_train = u_star[idx_tot,:]


            # Physical points
            if (options['extra_physical_points'] == True):
            
                nx1f = 10
                nx2f = 10
                nx3f = 10

                x1f = np.arange(lb[0],ub[0], (ub[0]-lb[0])/(nx1f-1))
                if (size_X > 1):
                    x2f = np.arange(lb[1],ub[1],(ub[1]-lb[1])/(nx2f-1))
                if (size_X > 2):
                    x3f = np.arange(lb[2],ub[2],(ub[2]-lb[2])/(nx3f-1))

                if (size_X == 1):
                    X_f_train = np.array(x1f).T.reshape(-1,1)
                elif (size_X == 2):
                    X_f_train = np.array(np.meshgrid(x1f, x2f)).T.reshape(-1,2)
                elif (size_X == 3):
                    X_f_train = np.array(np.meshgrid(x1f, x2f, x3f)).T.reshape(-1,3)

            else:
                X_f_train = X_u_train

            print(X_u_train)
            print(X_f_train)

            # Scaling input data
            if options['scaling_output_variables'] == True:
                u_train_scaled = np.divide((u_train-Ymin),(Ymax-Ymin))
            else:
                u_train_scaled = u_train
            print(u_train_scaled.max())
            print(u_train_scaled.min())

            # Scaling factors for residuals
            options['scaling_residuals'] = np.ones(ns);
            if options['scaling_residuals_type'] == 'Max' :
                options['scaling_residuals'] = Ymax
            elif options['scaling_residuals_type'] == 'Mean' :
                options['scaling_residuals'] = Ymean
            print("Scaling factors for residuals: ", options['scaling_residuals'])

            # Summary on the screen
            print("Number of target variables:    %d" % ns)
            print("Number of user-defined points: %d" % N_u)
            print("Number of exp. points(total):  %d" % len(u_star))
            print("Number of training points:     %d" % len(X_u_train))

        #--------------------------------------------------------------------------------#
        # NINN training section                                                          #
        #--------------------------------------------------------------------------------#
            if (options['pre_training'] == True):
                weights = []
                biases = []
                if (size_X == 1):
                    model_ninn = ninn1d(X_u_train, u_train_scaled, layers, lb, ub, options, weights, biases, True)
                elif (size_X == 2):
                    model_ninn = ninn2d(X_u_train, u_train_scaled, layers, lb, ub, options, weights, biases, True)
                elif (size_X == 3):
                    model_ninn = ninn3d(X_u_train, u_train_scaled, layers, lb, ub, options, weights, biases, True)

                # Training NINN
                start_time = time.time()
                model_ninn.train(0)
                elapsed = time.time() - start_time
                print('Training time: %.4f' % (elapsed))

        #--------------------------------------------------------------------------------#
        # PINN training section                                                          #
        #--------------------------------------------------------------------------------#
            if (options['pre_training'] == True):
                from_scratch = False
                weights = model_ninn.sess.run(model_ninn.weights)
                biases = model_ninn.sess.run(model_ninn.biases)
                for ii in range(len(weights)):
                    noise = np.random.random_sample(weights[ii].shape)/10.+1.
                    weights[ii] = np.multiply(weights[ii],np.float32(noise))
                    noise = np.random.random_sample(biases[ii].shape)/10.+1.
                    biases[ii] = np.multiply(biases[ii],np.float32(noise))
            else:
                from_scratch = True
                weights = []
                biases = []

            # PINN Definition
            if (size_X == 1):
                model_pinn = pinn1d(problem, X_u_train, u_train_scaled, X_f_train, layers, lb, ub, Ymin, Ymax, options, teta_guess, kappa_min, kappa_max, weights, biases, from_scratch, reactor_data)
            elif (size_X == 2):
                model_pinn = pinn2d(problem, X_u_train, u_train_scaled, X_f_train, layers, lb, ub, Ymin, Ymax, options, teta_guess, kappa_min, kappa_max, weights, biases, from_scratch, reactor_data)
            elif (size_X == 3):
                model_pinn = pinn3d(problem, X_u_train, u_train_scaled, layers, lb, ub, Ymin, Ymax, options, teta_guess, kappa_min, kappa_max, weights, biases, from_scratch, reactor_data)

            # Training PINN
            start_time = time.time()
            model_pinn.train(0)
            elapsed = time.time() - start_time
            print('Training time: %.4f' % (elapsed))

            # Neural network predicted variables
            u_nn, f_pred = model_pinn.predict(X_star)
            error_u = np.linalg.norm(u_star-u_nn,2)/np.linalg.norm(u_star,2)
            print('Avg error exp. PINN: %e' % (error_u))

            # Predicted parameters
            teta_pred = model_pinn.sess.run(model_pinn.teta)
            if (options['constrained'] == 'Linear'):
                kappa_pred = kappa_min + (kappa_max - kappa_min)*teta_pred
            elif (options['constrained'] == 'Exp'):
                kappa_pred = np.exp(teta_pred)
            elif (options['constrained'] == 'None'):
                kappa_pred = teta_pred

            error_kappa = np.abs(kappa_pred - kappa_true)/kappa_true*100.
            print('Predicted parameters: ', kappa_pred)
            print('Error(%):             ', error_kappa)

            # Write on file
            if (write_output_log == True):
                f_out.write("%e " % error_u)
                f_out.write("%f " % options['noise_sigma'])
                f_out.write("%d " % N_u)
                f_out.write("%f " % options['weight_phy'])
                f_out.write("%d " % number_internal_layers_from_input)
                f_out.write("%d " % neurons_from_input)
                for item in kappa_pred:
                    f_out.write("%e " % item)
                f_out.write("\n")
                f_out.flush()

        # f_out.close()

    if (write_output_log == True):
        f_out.close()
        sys.exit()

#--------------------------------------------------------------------------------#
# Plotting section                                                               #
#--------------------------------------------------------------------------------#

    # Target species
    tg_species = 0

    # True values
    ExpValues = (Y[:,tg_species]).reshape(len(x3), len(x2), len(x1))

    # Predicted values
    U_nn = (u_nn[:,tg_species]).reshape(len(x3), len(x2), len(x1))

    ## Plot 2D map
    fig = plt.figure(1)

    # 2D map
    if (size_X > 1):
        h = plt.imshow( ExpValues[0,:,:], interpolation='nearest', cmap='rainbow',
                        extent=[x1.min(), x1.max(), x2.min(), x2.max()],
                        origin='lower', aspect='auto' )


        # Add sampling points
        plt.plot(X_u_train[:,0], X_u_train[:,1], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 2, clip_on = False)

        # Add vertical white lines
        idxl = [ int(0.25*nx1), int(0.50*nx1), int(0.75*nx1) ]
        line = np.linspace(x2.min(), x2.max(), 2)[:,None]
        plt.plot(x1[idxl[0]]*np.ones((2,1)), line, 'w-', linewidth = 1)
        plt.plot(x1[idxl[1]]*np.ones((2,1)), line, 'w-', linewidth = 1)
        plt.plot(x1[idxl[2]]*np.ones((2,1)), line, 'w-', linewidth = 1)

        # Decoration
        plt.axis('off')
        plt.colorbar(h)
        plt.xlabel('time (min)')
        plt.ylabel('Qin (l/min)')
        plt.title('Concentration of A (mol/l)', fontsize = 10)
        plt.show()

        # Plot along the vertical sections
        for j in range(3):
            fig = plt.figure(2)

            plt.plot(x2,ExpValues[0,:,idxl[j]], 'b-', linewidth = 2, label = 'from exp')

            if (options['scaling_output_variables'] == True):
                plt.plot(x2, ( U_nn[0,:,idxl[j]]*(Ymax[tg_species]-Ymin[tg_species])+Ymin[tg_species] ), 'r--', linewidth = 2, label = 'from nn')
            else:
                plt.plot(x2, ( U_nn[0,:,idxl[j]] ), 'r--', linewidth = 2, label = 'from nn')

            plt.xlabel('T [K]')
            plt.ylabel('$F [mol/s]$')
            plt.title('$V=YYY$', fontsize = 10)
            #plt.axis('square')
            plt.xlim([min(x2),max(x2)])
            #plt.ylim([0,1.1*max(ExpValues[idxl[j],:,:])])
            plt.show()


    #--------------------------------------------------------------------------------#
    # Comparison with reactor model solution                                         #
    #--------------------------------------------------------------------------------#

    if (problem == 'AcetoAcetylation'):

        # Selection of variables of interest
        Qin = x2.max()
        Ccat = x3.min()
        x1span = np.linspace(0., reactor_data['tau'])

        # ODE solution with identified parameters
        p = [kappa_pred, reactor_data['Cin'], reactor_data['V'], Qin, Ccat]
        y_iden = odeint(reactor_aceto_acetylation, reactor_data['C0'], x1span, args=(p,))

        # ODE solution with true parameters
        p = [kappa_true, reactor_data['Cin'], reactor_data['V'], Qin, Ccat]
        y_true = odeint(reactor_aceto_acetylation, reactor_data['C0'], x1span, args=(p,))

        # Output options
        out_options = { 'xlabel': 'time (min)', 'ylabel': 'concentration (mol/l)',  'title': 'Concentration profile' }

    elif (problem == 'TolueneHydrogenation'):

        # Selection of variables of interest
        v2 = x2.max()
        v3 = x3.min()
        x1span = np.linspace(0., reactor_data['tau'])

        # ODE solution with identified parameters
        p = [kappa_pred]
        y_iden = odeint(reactor_toluene_hydrogenation, reactor_data['Cin'], x1span, args=(p,))

        # ODE solution with true parameters
        p = [kappa_true]
        y_true = odeint(reactor_toluene_hydrogenation, reactor_data['Cin'], x1span, args=(p,))

        # Output options
        out_options = { 'xlabel': 'time (min)', 'ylabel': 'concentration (mol/l)',  'title': 'Concentration profile' }

    elif (problem == 'AllylChlorination'):

        # Selection of variables of interest
        T = x2.max()
        F0 = x3.min() * reactor_data['Xin']
        x1span = np.linspace(0., reactor_data['Vtot'])

        # ODE solution with identified parameters
        p = [kappa_pred, T, reactor_data['P']]
        y_iden = odeint(reactor_allyl_chlorination, F0, x1span, args=(p,))

        # ODE solution with true parameters
        p = [kappa_true, T, reactor_data['P']]
        y_true = odeint(reactor_allyl_chlorination, F0, x1span, args=(p,))

        # Output options
        out_options = { 'xlabel': 'volume (m3)', 'ylabel': 'molar flow rate (mol/s)',  'title': 'Molar flow rate profile' }


    # Plotting: identified vs true solution
    plt.axis('on')
    plt.title(out_options['title'], fontsize = 10)
    plt.xlabel(out_options['xlabel'])
    plt.ylabel(out_options['ylabel'])
    plt.plot(x1span,y_iden, '--',  linewidth = 2, label = 'Identified teta')
    plt.plot(x1span,y_true, '-',   linewidth = 2, label = 'True teta')
    plt.legend()
    plt.show()

    # Selection of profile to be plotted
    ipoint = len(x2)-1

    # Compare profiles: true, identified and from neural network
    for j in range(ns):

        u_from_true = (u_star[:,j].reshape(len(x3), len(x2), len(x1)))[0,ipoint,:]
        u_from_nn = (u_nn[:,j].reshape(len(x3), len(x2), len(x1)))[0,ipoint,:]

        plt.axis('on')
        plt.title(out_options['title'] + ' ' + reactor_data['labels'][j], fontsize = 10)
        plt.xlabel(out_options['xlabel'])
        plt.ylabel(out_options['ylabel'])
        plt.plot(x1, u_from_true, 'b-',  linewidth = 2, label = 'true')

        if (options['scaling_output_variables'] == True):
            y_from_nn = (u_from_nn*(Ymax[j]-Ymin[j])+Ymin[j])
        else:
            y_from_nn = u_from_nn
        plt.plot(x1, y_from_nn, 'g-', linewidth = 1, label = 'from DNN')

        plt.plot(x1span, y_iden[:,j:(j+1)], 'r--', linewidth = 2, label = 'identified')
        plt.legend()
        plt.show()
