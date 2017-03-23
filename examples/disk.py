import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import numpy as np
import matplotlib.pyplot as plt
from graph import *
from node import *
from optimization_algorithm import *
from toy_datasets import *


def visualize(graph, n):
	x1min, x1max = -0.75, 0.75
	x2min, x2max = -0.75, 0.75
	dx1 = (x1max - x1min) / n
	dx2 = (x2max - x2min) / n
	Y = np.zeros((n, n))
	x2 = x2min
	for j in range(n):
		Y[:,j] = graph.propagate([np.array([polynomial_mapping([x1min+i*dx1, x2min+j*dx2]) for i in range(n)])])[0].flatten()
	plt.imshow(Y, extent=[x1min, x1max, x2min, x2max], vmin=0, vmax=1, interpolation='none', origin='lower')
	plt.colorbar()
	plt.show()

def fully_connected(layers, nb_inputs):
	nodes = []
	learnable_nodes = []

	input_node = InputNode()
	nodes.append(input_node)

	cur_input_node = input_node
	prev_size = nb_inputs+1
	for i, size in enumerate(layers):
		# layer
		bias_node = AddBiasNode([cur_input_node])
		weights_node = LearnableNode(np.random.rand(prev_size, size)*1-0.5)
		prod_node = MultiplicationNode([bias_node, weights_node])
		if i+1 < len(layers):
			activation_node = TanhNode([prod_node])
		else:
			activation_node = SigmoidNode([prod_node])
		# save the nodes
		learnable_nodes.append(weights_node)
		nodes += [bias_node, weights_node, prod_node, activation_node]
		cur_input_node = activation_node
		prev_size = size + 1

	expected_output_node = InputNode()
	#sub_node = SubstractionNode([expected_output_node, cur_input_node])
	cost_node = SigmoidCrossEntropyNode([expected_output_node, cur_input_node])
	#cost_node = Norm2Node([sub_node])

	nodes += [expected_output_node, cost_node]
	return Graph(nodes, [input_node], [cur_input_node], [expected_output_node], cost_node, learnable_nodes)

def polynomial_mapping(x):
	return (x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2)

def rate_convergence(layers):
	N = 100
	n = 1000
	c = 0
	for _ in range(N):
		graph = fully_connected(layers)
		for _ in range(n):
			graph.batch_gradient_descent([X], [Y], 1)
		if is_success(graph):
			print(True)
			c += 1
		else:
			print(False)
	return c / N

if __name__ == '__main__':
	N = 500
	X, Y = disk_dataset(N)
	plt.plot(X[Y == 0, 0], X[Y == 0, 1], 'bo')
	plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'ro')
	plt.axis('equal')
	plt.axis([-0.75, 0.75, -0.75, 0.75])
	plt.show()

	# With augmented outputs
	X = augmented_dataset(X, polynomial_mapping, 5)
	Y = Y.reshape((len(Y), 1))

	graph = fully_connected([1], 5)
	sgd = GradientDescent(graph.get_learnable_nodes(), 0.1)
	for _ in range(1000):
		graph.propagate([X])
		cost = graph.backpropagate([Y])
		sgd.optimize(X.shape[0])
		print(cost / 100)

	plt.figure()
	visualize(graph, 100)
