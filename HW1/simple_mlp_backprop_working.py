import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale

epsilon = 1e-5 

# NOTICE!!!! The data/hidden units 
# are in shape of (dimension, num_data).
# For example, for input data of size (4, 150),
# multiply with the first layer matrix of size (20, 4)
# gives you the first layer hidden units of size (20, 150).

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def accuracy(y, y_hat):
    return np.sum(y == np.argmax(y_hat, axis=0) ) / float(len(y))

# https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps, axis = 0)


# forward pass
# z is the neuron before activation
# a is the neuron after activation, i.e. a = \psi(z)
def fprop(x, y, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = softmax(z2)
    # cross-entropy loss
    # log_likelihood = -np.log(a2[y, range(len(y))]+epsilon)
    log_likelihood = -np.log(a2[y, range(len(y))])
    # log_likelihood = -(y * np.log(a2+epsilon) + (1-y) * np.log(1-a2+epsilon))
    loss = np.sum(log_likelihood) / len(y)

    ret = {'x': x, 'y': y, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret

# Reference: https://towardsdatascience.com/neural-network-on-iris-data-4e99601a42c8\
# Reference: https://deepnotes.io/softmax-crossentropy
# back-propagation
def bprop(fprop_cache):
    x, y, z1, a1, z2, a2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'a1', 'z2', 'a2', 'loss')]
    # TODO: complete the back-propagation and store the gradients
    # in the variables:
    # i.e., db1 is the gradient for b1 (d means partial gradient)
    # return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}
    num_samples = len(y)	    
    dz2 = a2	    
    dz2[y,range(num_samples)] -= 1
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(fprop_cache['W2'].T, dz2) , (sigmoid(z1) * (1-sigmoid(z1))))
    dW1 = np.dot(dz1, x.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


# get train data
iris = datasets.load_iris()
X = iris.data
# normalize data
X = scale(X)
y = iris.target
num_classes = len(set(y))

# You can change the hyper-parameters
# and run several times to get desirable performance.
num_hidden_units = 10
num_epoch = 100
lr = 1e-3

# initialize random parameters and inputs
# num_hidden_units*num_features
W1 = scale(np.random.randn(num_hidden_units, X.shape[1]), axis=1)
# num_hidden_units*1
b1 = np.random.randn(num_hidden_units, 1)
# num_classes*num_hidden_units
W2 = np.random.randn(num_classes, num_hidden_units)
# num_classes*1
b2 = np.random.randn(num_classes, 1)
params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

losses = np.zeros(num_epoch)
accuracies = np.zeros(num_epoch)


# We are doing batch gradient descent and only report the training
# losses and accuracies.
for ii in range(num_epoch):
  fprop_cache = fprop(X.T, y, params)  
  losses[ii] = fprop_cache['loss']
  accuracies[ii] =  accuracy(y, fprop_cache['a2'])
  bprop_cache = bprop(fprop_cache)  
  print("Epoch", ii, "loss", losses[ii], "accuracy", accuracies[ii])

  for key in params:
    params[key] -= np.mean(bprop_cache[key], axis = 1, keepdims=True) * lr

plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
plt.plot(accuracies)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()







