import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pylab

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

debug = False

def loadExamples(csvPath):
    X = []
    y = []
    with open(csvPath, 'rb') as f:
        next(f)  # skip header
        showed_image = 0
        for line in f:
            xi = line.split(',')
            xi[-1] = xi[-1].strip('\n') # remove newline, the last token
            yi = xi[-1]
            X.append(xi[0:-1])
            y.append(yi)
            if debug == True and showed_image == 0:
                xi_array = np.array(xi)
                xi_array = np.reshape(xi_array, (30, 30, 3))
                plt.imshow(np.uint8(xi_array))
                plt.show()
                showed_image += 1
    # Theano works with fp32 precision
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int32)
    # apply some very simple normalization to the data
    X -= X.mean()
    X /= X.std()

    X = X.reshape(
        -1,  # number of examples, -1 makes it so that this number is determined automatically
        3,   # 3 color channel
        30,  # first image dimension (vertical)
        30  # second image dimension (horizontal)
    )
    return X, y

def main(argList):
    with open(argList[2], 'r') as f:
        net = pickle.load(f)
    examples, values = loadExamples(argList[1])
    labels = net.predict(examples)
    print "predicted labels: "
    print labels
    print "actual lables:"
    print values

main(sys.argv)
