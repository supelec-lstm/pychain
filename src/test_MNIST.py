import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import time
import numpy as np
import matplotlib.pyplot as plt
from network import *
from neuron import *
from mnist import *

def init_function(n):
	return (np.random.rand(n) * 0.2 - 0.1) / 1000

def shuffle_dataset(X, Y):
	p = np.random.permutation(X.shape[0])
	return X[p], Y[p]

def normalized_dataset(X):
	mean_X = np.mean(X, axis=0)
	#stddev_X = np.std(X, axis=0)
	return (X - mean_X) / 255

def get_predicted_class(predicted_y):
	return np.argmax(predicted_y)

def accuracy(network, X, Y):
	true_positive = 0
	for x, y in zip(X, Y):
		predicted_y = network.propagate(x)
		predicted_class = get_predicted_class(predicted_y)
		if predicted_class == y:
			true_positive += 1
	return true_positive / X.shape[0]

def visualize(network, X, Y, nb_samples=25):
	images = X[:nb_samples].reshape((nb_samples, 28, 28)) * 255
	labels = Y[:nb_samples]
	for i, (image, label) in enumerate(zip(images, labels)):
		plt.subplot(5, 5, i+1)
		plt.imshow(image, cmap='Greys', vmin=0, vmax=255, interpolation='none')
		plt.title(str(get_predicted_class(network.propagate(X[i])))+ ' ' + str(labels[i]))
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	X, (nb_rows, nb_columns), Y = get_training_set('examples/mnist')
	input_bias = BiasNeuron('b1')
	inputs = []
	for i in range(28*28):
		inputs.append(InputNeuron('x'+str(i)))
	input_layer = [input_bias] + inputs

	hidden_layer = []
	for i in range(10):
		hidden_layer.append(LinearNeuron('h'+str(i), input_layer, init_function))

	outputs = []
	for i in range(10):
		outputs.append(SoftmaxNeuron('y'+str(i), hidden_layer, i))

	expected_output = InputNeuron('expected_y')
	cost_neuron = CrossEntropyNeuron('cost', [expected_output], outputs)

	neurons = input_layer + hidden_layer + outputs + [expected_output, cost_neuron]
	network = Network(neurons, inputs, outputs, [expected_output], cost_neuron)

	X, (nb_rows, nb_columns), Y = get_training_set('examples/mnist')
	print(X.shape)
	X, Y = shuffle_dataset(X, Y)
	X = normalized_dataset(X)
	Y = Y.reshape((len(Y), 1))

	X_test, (_, _), Y_test = get_test_set('examples/mnist')
	print(X_test.shape)
	X_test, Y_test = shuffle_dataset(X_test, Y_test)
	X_test = normalized_dataset(X_test)
	Y_test = Y_test.reshape((len(Y_test), 1))

	batch_size = 128
	start_time = time.time()
	for i in range(0, X.shape[0], batch_size):
		#network.stochastic_gradient_descent(X, Y, 0.3)
		print(i)
		network.batch_gradient_descent(X[i:i+batch_size], Y[i:i+batch_size], 0.3)
		if (i % 2048) == 0:
			print('ACCURACY TRAINING:', accuracy(network, X, Y))
			print('ACCURACY TEST:', accuracy(network, X_test, Y_test))
	print('ACCURACY TEST:', accuracy(network, X_test, Y_test))
	print('DURATION: ', time.time() - start_time)
	visualize(network, X_test, Y_test)