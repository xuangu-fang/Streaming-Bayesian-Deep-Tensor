import numpy as np
import pandas as pd

def load_SG(fold_path='../data/SG/',num = 0):

    train_path = fold_path + 'SG' + '_train_' + str(num) + '.txt'
    test_path = fold_path + 'SG' +'_test_' + str(num) + '.txt'

    y = []
    ind = []
    with open(train_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y.append(float(items[-1]))
            ind.append([float(idx) for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)

    ind_test = []
    y_test = []
    with open(test_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y_test.append(float(items[-1]))
            ind_test.append([float(idx) for idx in items[0:-1]])
        ind_test = np.array(ind_test)
        y_test = np.array(y_test)

    return ind.astype(int),y,ind_test.astype(int),y_test

def load_mvlens_100k(fold_path='../data/mvlens_100k/',num = 0):

    train_path = fold_path + 'mv' + '_train_' + str(num) + '.txt'
    test_path = fold_path + 'mv' +'_test_' + str(num) + '.txt'

    y = []
    ind = []
    with open(train_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y.append(float(items[-1]))
            ind.append([float(idx) for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)

    ind_test = []
    y_test = []
    with open(test_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y_test.append(float(items[-1]))
            ind_test.append([float(idx) for idx in items[0:-1]])
        ind_test = np.array(ind_test)
        y_test = np.array(y_test)

    return ind.astype(int),y,ind_test.astype(int),y_test


def load_Gowalla(fold_path='../data/Gowalla/',num = 0):
    train_path = fold_path + 'Gowalla' + '_train_' + str(num) + '.txt'
    test_path = fold_path  + 'Gowalla' + '_test_' + str(num) + '.txt'

    y = []
    ind = []
    with open(train_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y.append(float(items[-1]))
            ind.append([float(idx) for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)

    ind_test = []
    y_test = []
    with open(test_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y_test.append(float(items[-1]))
            ind_test.append([float(idx) for idx in items[0:-1]])
        ind_test = np.array(ind_test)
        y_test = np.array(y_test)

    return ind.astype(int),y,ind_test.astype(int),y_test


def load_acc_train(fold_path='../data/acc/ibm-large-tensor.txt'):
    y = []
    ind = []
    with open(fold_path, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            y.append(float(items[-1]))
            ind.append([float(idx) for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)
        #y[y>10]=10
        #y = y/17.427057
    return ind.astype(int), y 

#ind,y = load_acc_train()

def load_acc_test_long(fold_path='../data/acc/acc_test_long.txt'):
    
    ind_test = []
    y_test = []
    with open(fold_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y_test.append(float(items[-1]))
            ind_test.append([float(idx) for idx in items[0:-1]])
        ind_test = np.array(ind_test)
        y_test = np.array(y_test)

    return ind_test.astype(int),y_test



def load_acc_test(fold_path='../data/acc/',num = 0):
    test_path = fold_path  + 'acc' + '_test_' + str(num) + '.txt'
    ind_test = []
    y_test = []
    with open(test_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y_test.append(float(items[-1]))
            ind_test.append([float(idx) for idx in items[0:-1]])
        ind_test = np.array(ind_test)
        y_test = np.array(y_test)
        #y_test = y_test/17.427057

    return ind_test.astype(int),y_test

def load_CTR(fold_path='../data/CTR/',num = 0):
    train_path = fold_path + 'CTR' + '_train_' + str(num) + '.txt'
    test_path = fold_path  + 'CTR' + '_test_' + str(num) + '.txt'

    y = []
    ind = []
    with open(train_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y.append(float(items[-1]))
            ind.append([float(idx) for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)

    ind_test = []
    y_test = []
    with open(test_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y_test.append(float(items[-1]))
            ind_test.append([float(idx) for idx in items[0:-1]])
        ind_test = np.array(ind_test)
        y_test = np.array(y_test)

    return ind.astype(int),y,ind_test.astype(int),y_test

def load_alog(fold_path = '../data/alog_dfnt/', num=0):

    train_path = fold_path + 'alog_dfnt_train_' + str(num) + '.txt'
    test_path = fold_path + 'alog_dfnt_test_' + str(num) + '.txt'

    y = []
    ind = []
    with open(train_path, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            y.append(float(items[-1]))
            ind.append([float(idx) for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)

        ind = ind[:9000]
        y = y[:9000]

    ind_test = []
    y_test = []
    with open(test_path, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            y_test.append(float(items[-1]))
            ind_test.append([float(idx) for idx in items[0:-1]])
        ind_test = np.array(ind_test)
        y_test = np.array(y_test)

    return ind.astype(int), y, ind_test.astype(int),y_test


def load_CTR_new(fold_path='../data/CTR/'):
    load_train = pd.read_csv('../data/CTR/CTR_train_1m_new.txt',sep = ' ',header=None).values
    load_test = pd.read_csv('../data/CTR/CTR_test_100k_new.txt',sep = ' ',header=None).values

    ind = load_train[:,0:-1]
    y = load_train[:,-1]

    ind_test = load_test[:,0:-1]
    y_test = load_test[:,-1]

    return ind,y,ind_test,y_test
    
def load_toy(fold_path = '../data/toy_data/',num=0):

    train_path = fold_path + 'toy_train_new'+str(num)+'.txt'
    test_path = fold_path + 'toy_test_new'+str(num)+'.txt' 
    
    y = []
    ind = []
    with open(train_path, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            y.append(float(items[-1]))
            ind.append([float(idx) for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)
    
    ind_test = []
    y_test = []
    with open(test_path, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            y_test.append(float(items[-1]))
            ind_test.append([float(idx) for idx in items[0:-1]])
        ind_test = np.array(ind_test)
        y_test = np.array(y_test)

    return ind.astype(int),y, ind_test.astype(int),y_test

def load_climate(fold_path = '../data/climate_dfnt/', num=0):
    train_path = fold_path + 'climate' + '_train_' + str(num) + '_dfnt.txt'
    test_path = fold_path + 'climate' +'_test_' + str(num) + '_dfnt.txt'
    
    y = []
    ind = []
    with open(train_path, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            y.append(float(items[-1]))
            ind.append([float(idx) for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)
    
    ind_test = []
    y_test = []

    with open(test_path, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            y_test.append(float(items[-1]))
            ind_test.append([float(idx) for idx in items[0:-1]])
        ind_test = np.array(ind_test)
        y_test = np.array(y_test)

    return ind.astype(int),y,ind_test.astype(int),y_test




