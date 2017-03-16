import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import numpy as np
import matplotlib.pyplot as plt
from graph import *
from node import *
from toy_datasets import *

def evaluate(graph):
	print('(0, 0) -> ' + str(graph.propagate([np.array([[0, 0]])])))
	print('(1, 0) -> ' + str(graph.propagate([np.array([[1, 0]])])))
	print('(0, 1) -> ' + str(graph.propagate([np.array([[0, 1]])])))
	print('(1, 1) -> ' + str(graph.propagate([np.array([[1, 1]])])))

def is_success(graph):
	return graph.propagate([np.array([[0, 0]])])[0] < 0.5 and \
		graph.propagate([np.array([[1, 0]])])[0] > 0.5 and \
		graph.propagate([np.array([[0, 1]])])[0] > 0.5 and \
		graph.propagate([np.array([[1, 1]])])[0] < 0.5

def visualize(graph, n):
	x1min, x1max = -0.5, 1.5
	x2min, x2max = -0.5, 1.5
	dx1 = (x1max - x1min) / n
	dx2 = (x2max - x2min) / n
	Y = np.zeros((n, n))
	x2 = x2min
	for j in range(n):
		Y[:,j] = graph.propagate([np.array([[x1min+i*dx1, x2min+j*dx2] for i in range(n)])])[0].flatten()
	plt.imshow(Y, extent=[x1min, x1max, x2min, x2max], vmin=0, vmax=1, interpolation='none', origin='lower')
	plt.colorbar()
	plt.show()

def fully_connected(layers):
	nodes = []
	learnable_nodes = []

	input_node = InputNode()
	nodes.append(input_node)

	cur_input_node = input_node
	prev_size = 3
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
	X, Y = xor_dataset()
	Y = Y.reshape((len(Y), 1))

	#print(rate_convergence([4, 4, 1]))
	graph = fully_connected([4, 4, 1])
	for _ in range(1000):
		print(graph.batch_gradient_descent([X], [Y], 0.1) / 4)
		#print(type(graph.nodes[4]))
		#print(graph.nodes[4].evaluate())
	evaluate(graph)
	print(is_success(graph))
	visualize(graph, 100)
