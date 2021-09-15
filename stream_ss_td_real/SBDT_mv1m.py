
import math

import numpy as np
import time
import sys
sys.path.append('SBDT_net/')
import SBDT_net

np.random.seed(1)



# load movielens 1m
ndims = [6040, 3706]
ind = np.loadtxt('./data_real/movielens_1m/mv_train_ind.txt').astype(int)
y = np.loadtxt('./data_real/movielens_1m/mv_train_y.txt').astype(int)
ind_test = np.loadtxt('./data_real/movielens_1m/mv_test_ind.txt').astype(int)
y_test = np.loadtxt('./data_real/movielens_1m/mv_test_y.txt').astype(int)



X_train = ind
y_train = y
perm = np.random.permutation(X_train.shape[0])
X_train = X_train[perm]
y_train = y_train[perm]


X_test = ind_test
y_test = y_test

print('loaded')

n_hidden_units = 50
n_epochs =  1
n_stream_batch = 1


# mini_batch_list = [256]
# R_list = [3,5,8,10]

mini_batch_list = [64,128,512]
R_list = [8]



avg_num=3

help_str = 'mv_'+str(R_list[0])

mode = 'minibatch'#'single' #'minibatch'

for mini_batch in mini_batch_list:
    for R in R_list:

        mse_list = np.zeros(avg_num)

        set_start = time.time()

        time_list = np.zeros(avg_num)

        for i in range(avg_num):

            fold_start = time.time()

        # shuffel
            X_train = ind
            y_train = y
            perm = np.random.permutation(X_train.shape[0])
            # perm = perm[:1000]
            X_train = X_train[perm].astype(np.int32)
            y_train = y_train[perm].astype(np.int32)

            X_test = ind_test.astype(np.int32)
            y_test = y_test.astype(np.int32)

            net = SBDT_net.PBP_net(X_train, y_train,
                [ n_hidden_units, n_hidden_units ], normalize = True, n_epochs = n_epochs,\
                    R=R,ndims=ndims,n_stream_batch = n_stream_batch,mode=mode,mini_batch=mini_batch)


            running_performance = np.array(net.pbp_train(X_test,y_test,help_str))
            # file_name = 'running_result/%s.txt'%(help_str)
            # np.savetxt(file_name,np.c_[running_performance])
            # print("\n  saved!\n")

            total_turn = float(X_train.shape[0])/mini_batch

            running_time = (time.time()-fold_start)*total_turn/100

            time_list[i] = running_time

            m, a, b = net.predict_deterministic(X_test)

        # We compute the test MSE

            mse = np.mean((y_test - m)**2)
            print('mse = %f'%(mse))
            print('a, b, mean(tau), var(tau)')
            print('take %g seconds to finish fold %d'%(time.time()-fold_start, i))
            print(a, b, a/b, a/(b**2))

            mse_list[i] = mse 

        
        print('\navg of mse: %.6g , std of mse is %.6g'% (mse_list.mean(),mse_list.std()))
        print('\n take %g seconds to finish the setting'%(time.time()-set_start))

