
import math

import numpy as np
import time
import sys
sys.path.append('SBDT_net/')
import SBDT_net
# import data_loader_pan as data_loader
from sklearn.metrics import roc_auc_score




'''
load anime
'''

ndims = [25838, 4066]

ind = np.loadtxt('./data_binary/anime/anime_train_ind.txt').astype(int)
y = np.loadtxt('./data_binary/anime/anime_train_y.txt').astype(int)
ind_test = np.loadtxt('./data_binary/anime/anime_test_ind.txt').astype(int)
y_test = np.loadtxt('./data_binary/anime/anime_test_y.txt').astype(int)



print('loaded')

avg_num=1

n_hidden_units = 50
n_epochs = 1
n_stream_batch = 1



R_list = [3,5,8,10]
mini_batch_list = [256]#[64,128,512]


help_str = 'anime'+str(R_list[0])

mode = 'minibatch' 
# mode = 'single' 
for mini_batch in mini_batch_list:
    for R in R_list:

        auc_list = np.zeros(avg_num)

        set_start = time.time()

        time_list = np.zeros(avg_num)

        for i in range(avg_num):

            fold_start = time.time()
            
        # shuffel
            X_train = ind
            y_train = y
            perm = np.random.permutation(X_train.shape[0])
            # perm = perm[:500]
            X_train = X_train[perm].astype(np.int32)
            y_train = y_train[perm].astype(np.int32)

            X_test = ind_test.astype(np.int32)
            y_test = y_test.astype(np.int32)

            net = SBDT_net.PBP_net(X_train, y_train,
                [ n_hidden_units, n_hidden_units ], normalize = True, n_epochs = n_epochs,\
                    R=R,ndims=ndims,n_stream_batch = n_stream_batch,mode=mode,mini_batch=mini_batch)
             
             
            running_performance = np.array(net.pbp_train(X_test,y_test,help_str))

            total_turn = float(X_train.shape[0])/mini_batch

            running_time = (time.time()-fold_start)*total_turn/100

            time_list[i] = running_time


            m, a, b = net.predict_deterministic(X_test)


        # We compute the test AUC
            auc = roc_auc_score(y_test,m)

        # rmse = np.sqrt(np.mean((y_test - m)**2))

            print('auc=%f'%(auc))
            print('a, b, mean(tau), var(tau)')
            print('take %g seconds to finish fold %d'%(time.time()-fold_start, i))
            print(a, b, a/b, a/(b**2))
            
            auc_list[i] = auc


        print('\navg of auc: %.6g , std of auc is %.6g'% (auc_list.mean(),auc_list.std()))

        f= open("new_result/anime_result_v1.txt","a+")
        f.write(' R = %d, mini_batch =%s '%(R, mini_batch))
        f.write('\navg of auc: %.6g , std of auc is %.6g'% (auc_list.mean(),auc_list.std()))
        f.write('\n the exact value is %s'%str(auc_list))
        f.write(' mean(tau): %.5g, var(tau): %.5g'%(a/b, a/(b**2)))
        final_time = time.time()-set_start
        f.write('\n take %.4g seconds to finish the setting, avg time is %.4g '%(final_time,final_time/avg_num))

        f.write('\n\n')
        f.close()


