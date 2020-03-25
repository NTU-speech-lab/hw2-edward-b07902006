import numpy as np
import pandas as pd
import sys

X_test_fpath = sys.argv[1]
output_fpath = sys.argv[2]
bad = [0,126,210,211,212,358,507]
a = []
features = [' Not in universe',' ?']
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
feature = np.array(content)
for i in range(len(content)):
    if content[i] in features:
        bad.append(i)
# add one-hot
def one_hot(X):
    #age
    add = np.zeros((X.shape[0],7))
    bins = [-np.inf,20,25,35,45,55,65,np.inf]
    for i in range(X.shape[0]):
        for j in range(7):
            if bins[j] <= X[i][0] < bins[j + 1]:
                add[i][j] = 1
                break
    X = np.hstack([X,add])
    #wage per hour
    add = np.zeros((X.shape[0],6))
    bin = [0,300,1000,1500,2000,2500,np.inf]
    for i in range(X.shape[0]):
        for j in range(6):
            if bins[j] <= X[i][126] < bins[j + 1]:
                add[i][j] = 1
                break
    X = np.hstack([X,add])
    #capital gains
    add = np.zeros((X.shape[0],8))
    bins = [0,1000,3000,5000,7000,10000,20000,50000,np.inf]
    for i in range(X.shape[0]):
        for j in range(8):
            if bins[j] <= X[i][210] < bins[j + 1]:
                add[i][j] = 1
                break
    X = np.hstack([X,add])
    #capital loss
    add = np.zeros((X.shape[0],4))
    bins = [0,1500,2000,2500,np.inf]
    for i in range(X.shape[0]):
        for j in range(4):
            if bins[j] <= X[i][211] < bins[j + 1]:
                add[i][j] = 1
                break
    X = np.hstack([X,add])
    #dividents of stocks
    add = np.zeros((X.shape[0],7))
    bins = [0,500,1500,3000,5000,7500,9999,np.inf]
    for i in range(X.shape[0]):
        for j in range(7):
            if bins[j] <= X[i][212] < bins[j + 1]:
                add[i][j] = 1
                break
    X = np.hstack([X,add])
    #num persons worked for employer
    add = np.zeros((X.shape[0],7))
    bins = [0,1,2,3,4,5,6,np.inf]
    for i in range(X.shape[0]):
        for j in range(7):
            if bins[j] <= X[i][358] < bins[j + 1]:
                add[i][j] = 1
                break
    X = np.hstack([X,add])
    #weeks worked in year
    add = np.zeros((X.shape[0],6))
    bins = [0,10,20,30,40,50,np.inf]
    for i in range(X.shape[0]):
        for j in range(6):
            if bins[j] <= X[i][507] < bins[j + 1]:
                add[i][j] = 1
                break
    X = np.hstack([X,add])
    return X
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    X_test = one_hot(X_test)
    X_test = np.delete(X_test,bad,axis=1)
def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 0.00000001, 0.99999999)

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)
def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std
X_mean = np.load('X_mean.npy')
X_std = np.load('X_std.npy')
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
w = np.load('weight_best.npy')
b = np.load('b_best.npy')
predictions = _predict(X_test, w, b)
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))