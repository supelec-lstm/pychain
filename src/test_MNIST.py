import sys
import os
from copy import deepcopy
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import time
import numpy as np
import matplotlib.pyplot as plt
from graph import *
from node import *
from MNIST import *

def init_function(n):
    return (np.random.rand(n) * 0.2 - 0.1) / 1000

def shuffle_dataset(X, Y):
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]

def one_hot_encode(Y):
	return (np.dot(np.ones((Y.shape[0], 10)), np.diag(np.arange(10))) == Y).astype(float)


def normalized_dataset(X):
    mean_X = np.mean(X, axis=0)
    #stddev_X = np.std(X, axis=0)
    return (X - mean_X) / 255

def get_predicted_class(predicted_y):
    return np.argmax(predicted_y, axis=1)

def accuracy(graph, X, Y):
    true_positive = 0
    predicted_y = graph.propagate(X)
    predicted_class = get_predicted_class(predicted_y)
    print(Y.shape[0])
    return np.sum(Y.flatten() == predicted_class) / Y.shape[0]

def visualize(graph, X, Y, nb_samples=25):
    images = X[:nb_samples].reshape((nb_samples, 28, 28)) * 255
    labels = Y[:nb_samples]
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(5, 5, i+1)
        plt.imshow(image, cmap='Greys', vmin=0, vmax=255, interpolation='none')
        plt.title(str(get_predicted_class(graph.propagate([X[i]])))+ ' ' + str(labels[i]))
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    X, (nb_rows, nb_columns), Y = get_training_set()
#    input_bias = BiasNeuron('b1')
#    inputs = []
#    for i in range(28*28):
#        inputs.append(InputNeuron('x'+str(i)))
#        input_layer = [input_bias] + inputs
#
#    hidden_layer = []
#    for i in range(10):
#        hidden_layer.append(LinearNeuron('h'+str(i), input_layer, init_function))
#
#    outputs = []
#    for i in range(10):
#        outputs.append(SoftmaxNeuron('y'+str(i), hidden_layer, i))
#
#    expected_output = InputNeuron('expected_y')
#    cost_neuron = CrossEntropyNeuron('cost', [expected_output], outputs)
#
#    neurons = input_layer + hidden_layer + outputs + [expected_output, cost_neuron]
#    network = Network(neurons, inputs, outputs, [expected_output], cost_neuron)
    ""
    input_nod = InputNode()

    w1 = LearnableNode((784,10))

   # w2 = LearnableNode((10,1))

    h1 = MultiplicationNode([input_nod,w1])

    s1 = SoftMaxNode([h1])

    #h2 = MultiplicationNode([s1,w2])

    #s2 = SoftMaxNode([h2])

    expected_output = InputNode()

    #d1 = SubstractionNode([s2, expected_output])

    e1 = SoftmaxCrossEntropyNode([expected_output, s1])

    #d1 = SigmoidCrossEntropyNode([s2, expected_output])

    c1 = ConstantGradientNode([e1])

    graph = Graph([input_nod, w1, h1, s1, expected_output, e1, c1], input_nod, s1, expected_output, c1, [w1])
    
    ""
    
    X, (nb_rows, nb_columns), Y = get_training_set()
    print(X.shape)
    X, Y = shuffle_dataset(X, Y)
    X = normalized_dataset(X)
    Y = Y.reshape((len(Y), 1))
    ohe_Y = one_hot_encode(Y)

    X_test, (_, _), Y_test = get_test_set()
    print(X_test.shape)
    X_test, Y_test = shuffle_dataset(X_test, Y_test)
    X_test = normalized_dataset(X_test)
    Y_test = Y_test.reshape((len(Y_test), 1))

    w11 = deepcopy(w1.evaluate())

    batch_size = 128 #128
    start_time = time.time()
    for i in range(0, X.shape[0], batch_size):
        #network.stochastic_gradient_descent(X, Y, 0.3)
        print(i)
        print(graph.batch_descent_gradient(0.8, X[i:i+batch_size], ohe_Y[i:i+batch_size]))
        if (i % 2048) == 0:
            print('ACCURACY TRAINING:', accuracy(graph, X, Y))
            print('ACCURACY TEST:', accuracy(graph, X_test, Y_test))
        print('ACCURACY TEST:', accuracy(graph, X_test, Y_test))
        print('DURATION: ', time.time() - start_time)
    visualize(graph, X_test, Y_test)

    print('w11', w11)
    w12 = w1.evaluate().copy()
    print('w12', w12)
    wtest=w11-w12
    print(wtest.any())