import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import time
import numpy as np
import matplotlib.pyplot as plt
from graph import *
from node import *
from mnist import *

def init_function(shape):
	return (np.random.rand(*shape) * 0.2 - 0.1)

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
		plt.title(str(get_predicted_class(graph.propagate(np.array([X[i]])))[0])+ ' ' + str(labels[i]))
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	input_node = InputNode()
	bias1_node = AddBiasNode(input_node)

	weights1_node = LearnableNode((28*28+1, 10), init_function)
	prod1_node = MultiplicationNode(bias1_node, weights1_node)

	output_node = SoftmaxNode(prod1_node)
	expected_output_node = InputNode()
	cost_node = SoftmaxCrossEntropyNode(expected_output_node, output_node)

	nodes = [input_node, bias1_node, weights1_node, prod1_node, output_node, expected_output_node, cost_node]
	graph = Graph(nodes, input_node, output_node, expected_output_node, cost_node, [weights1_node]) 

	X, (nb_rows, nb_columns), Y = get_training_set('examples/mnist')
	print(X.shape)
	X, Y = shuffle_dataset(X, Y)
	X = normalized_dataset(X)
	Y = Y.reshape((len(Y), 1))
	ohe_Y = one_hot_encode(Y)

	X_test, (_, _), Y_test = get_test_set('examples/mnist')
	print(X_test.shape)
	X_test, Y_test = shuffle_dataset(X_test, Y_test)
	X_test = normalized_dataset(X_test)
	Y_test = Y_test.reshape((len(Y_test), 1))

	batch_size = 128
	start_time = time.time()
	for i in range(0, X.shape[0], batch_size):
		print(i)
		print(graph.batch_gradient_descent(X[i:i+batch_size], ohe_Y[i:i+batch_size], 0.3) / 128)
		if (i % 2048) == 0:
			print('ACCURACY TRAINING:', accuracy(graph, X, Y))
			print('ACCURACY TEST:', accuracy(graph, X_test, Y_test))
	print('ACCURACY TEST:', accuracy(graph, X_test, Y_test))
	print('DURATION: ', time.time() - start_time)
	visualize(graph, X_test, Y_test)