
import numpy as np

import pickle

import gzip

import pbp

from sklearn.metrics import roc_auc_score

class PBP_net:

    #need to go through the dataset multiple times
    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40,
        normalize = False,R=3,ndims = [200,100,200],n_stream_batch=1,mini_batch=100,mode='single'):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        # if normalize:
        #     self.std_X_train = np.std(X_train, 0)
        #     self.std_X_train[ self.std_X_train == 0 ] = 1
        #     self.mean_X_train = np.mean(X_train, 0)
        # else:
        #     self.std_X_train = np.ones(X_train.shape[ 1 ])
        #     self.mean_X_train = np.zeros(X_train.shape[ 1 ])

        # X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
        #     np.full(X_train.shape, self.std_X_train)
        self.R = R
        self.nmod = len(ndims)#3
        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        self.stream_batch = n_stream_batch
        self.mode = mode

        self.mini_batch = mini_batch

        self.y_train_normalized = y_train#(y_train - self.mean_y_train) / self.std_y_train

        self.X_train = X_train

        self.n_epochs = n_epochs

        self.N_turns = self.X_train.shape[0]/self.mini_batch

        self.test_point = int(0.05*self.X_train.shape[0]/self.mini_batch)

        # We construct the network

        # n_units_per_layer = \
        #     np.concatenate(([ X_train.shape[ 1 ] ], n_hidden, [ 1 ]))

        # concatnate first, then feed in NN
        n_units_per_layer = \
            np.concatenate(([ self.nmod*self.R ], n_hidden, [ 1 ]))

        self.running_score = []

        
        self.pbp_instance = \
            pbp.PBP(n_units_per_layer, self.mean_y_train, self.std_y_train,self.R, ndims,n_stream_batch)

        # We iterate the learning process
        '''
        curr = 0
        N = X_train.shape[0]
        nb = 100
        while curr<N:
            self.pbp_instance.do_pbp(X_train[curr:curr + nb], y_train_normalized[curr: curr + nb], n_epochs)
            curr = curr + nb
            print(curr)
        print('done')
        '''


        # We are done!

    def pbp_train(self,X_test,y_test,help_str=''):

        if self.mode == 'single':
            self.pbp_instance.do_pbp(self.X_train, self.y_train_normalized, self.n_epochs)
        else:
            count = 0
            turn = 0
            mini_batch = self.mini_batch
            while count+mini_batch<=self.X_train.shape[0]:
                
                X_sub = self.X_train[count:count + mini_batch]
                y_sub = self.y_train_normalized[count:count + mini_batch]

                self.pbp_instance.do_pbp(X_sub, y_sub, self.n_epochs)

                count = count + mini_batch
                print('finish  %d / %d '%(count,self.X_train.shape[0]) + help_str)
                
                turn=turn+1

            
            return self.running_score


    def re_train(self, X_train, y_train, n_epochs):

        """
            Function that re-trains the network on some data.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_epochs     Numer of epochs for which to train the
                                network. 
        """

        # We normalize the training data 

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train

        self.pbp_instance.do_pbp(X_train, y_train_normalized, n_epochs)

    def predict(self, X_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data
            
    
            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        X_test = np.array(X_test, ndmin = 2)

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        m, v, v_noise = self.pbp_instance.get_predictive_mean_and_variance(X_test)

        # We are done!

        return m, v, v_noise

    def predict_deterministic(self, X_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data
            
    
            @return o       The predictive value for the test target variables.

        """

        # X_test = np.array(X_test, ndmin = 2)

        # We normalize the test set

        # X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
        #     np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        o,var_m,var_v = self.pbp_instance.get_deterministic_output(X_test)
        

        # We are done!

        return o,var_m,var_v

    def sample_weights(self):

        """
            Function that draws a sample from the posterior approximation
            to the weights distribution.

        """
 
        self.pbp_instance.sample_w()

    def save_to_file(self, filename):

        """
            Function that stores the network in a file.

            @param filename   The name of the file.
            
        """

        # We save the network to a file using pickle

        def save_object(obj, filename):

            result = pickle.dumps(obj)
            with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
            dest.close()

        save_object(self, filename)

def load_PBP_net_from_file(filename):

    """
        Function that load a network from a file.

        @param filename   The name of the file.
        
    """

    def load_object(filename):

        with gzip.GzipFile(filename, 'rb') as \
            source: result = source.read()
        ret = pickle.loads(result)
        source.close()

        return ret

    # We load the dictionary with the network parameters

    PBP_network = load_object(filename)

    return PBP_network
