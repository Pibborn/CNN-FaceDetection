# some of the code from http://nbviewer.ipython.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from lasagne.layers import get_all_params

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective

import pickle

debug = 1

def load_faces(path):
    debug_var = 0 # no change necessary for debugging, please ignore
    X = []
    y = []
    with open(path, 'rb') as f:
        next(f)  # skip header
        for line in f:
            xi = line.split(',')
            xi[-1] = xi[-1].rstrip('\n') # remove newline from last token
            yi = xi[-1]
            X.append(xi[0:-1]) 
            y.append(yi)
            if debug == 3 and debug_var < 3:
                temp = np.reshape(xi[0:-1], (30, 30, 3))
                plt.imshow(np.uint8(temp))
                plt.show()
                #print xi[0:-1]
                #print yi
                debug_var += 1
    # Theano works with fp32 precision
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int32)

    # apply some very simple normalization to the data
    X -= X.mean()
    X /= X.std()

    if debug == 1:
        print str(X.shape)

    X = X.reshape(
        -1,  # number of samples, -1 makes it so that this number is determined automatically
        3,   # 3 color channel
        30,  # first image dimension (vertical)
        30,  # second image dimension (horizontal)
    )

    return X, y

def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    # default loss
    losses = objective(layers, *args, **kwargs)
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = get_all_params(layers[-1], regularizable=True)
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    return losses

def main(argList):
    path = (argList[1])
    print "loading data from "+path+"..."
    X, y = load_faces(path)
    print "done."   
    # Nolearn allows you to skip Lasagne's incoming keyword, which specifies how the layers are connected.
    # Instead, nolearn will automatically assume that layers are connected in the order they appear in the list.
    layers1 = [
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

    (Conv2DLayer, {'num_filters': 15, 'filter_size': (3, 3)}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 10, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 10, 'filter_size': (3, 3)}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (DenseLayer, {'num_units': 50}),

    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
    ]

    net0 = NeuralNet(
        layers=layers1,
        update_learning_rate=0.0002,
        max_epochs=1000,
        objective_l2=0.0025,
        verbose=3,
        train_split=TrainSplit(eval_size=0.2)
    )

    print "training neural net..."
    net0.fit(X, y)
    print "done."
    print "pickling the model..."
    with open(argList[2], 'wb') as f:
        pickle.dump(net0, f, -1)
    print "done."

main(sys.argv)
