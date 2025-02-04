import numpy as np
import pandas as pd
import time

np.random.seed(0)
X_train_fpath = '/Users/edward/Desktop/code/EEML/hw2/X_train.csv'
Y_train_fpath = '/Users/edward/Desktop/code/EEML/hw2/Y_train'
X_test_fpath = '/Users/edward/Desktop/code/EEML/hw2/X_test'
output_fpath = './output_{}.csv'
bad = [0,126,210,211,212,358,507]
a = []
features = [' Not in universe',' ?']
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
feature = np.array(content)
for i in range(len(content)):
    if content[i] in features:
        bad.append(i)
feature = np.delete(feature,[0])
feature = np.delete(feature,bad)
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
# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    X_train = one_hot(X_train)
    X_train = np.delete(X_train,bad,axis=1)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    X_test = one_hot(X_test)
    X_test = np.delete(X_test,bad,axis=1)
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

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
    
# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))
def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

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
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc
def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b,lamb):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1) + 2 * w * lamb
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad
# Zero initialization for weights ans bias
w = np.zeros((data_dim,)) 
b = np.zeros((1,))

# Some parameters for training    
max_iter = 500
batch_size = 16
learning_rate = 0.01

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []
lamb = 0
# Calcuate the number of parameter updates
step = 1

# Iterative training
for epoch in range(max_iter):
    # Random shuffle at the begging of each epoch
    X_train, Y_train = _shuffle(X_train, Y_train)
        
    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[idx*batch_size:(idx+1)*batch_size]

        # Compute the gradient
        w_grad, b_grad = _gradient(X, Y, w, b,lamb)
            
        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate/np.sqrt(step) * w_grad
        b = b - learning_rate/np.sqrt(step) * b_grad

        step = step + 1
            
    # Compute loss and accuracy of training set and development set
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))
np.save('weight_best.npy',w)
np.save('b_best.npy',b)
import matplotlib.pyplot as plt

# Loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()

# Predict testing labels
predictions = _predict(X_test, w, b)
with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

