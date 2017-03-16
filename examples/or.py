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
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.colorbar()
	plt.show()

if __name__ == '__main__':
	input_node = InputNode()
	bias_node = AddBiasNode([input_node])
	weights_node = LearnableNode(np.random.rand(3, 1))
	prod_node = MultiplicationNode([bias_node, weights_node])
	output_node = SigmoidNode([prod_node])
	expected_output_node = InputNode()
	cost_node = SigmoidCrossEntropyNode([expected_output_node, output_node])
	nodes = [input_node, bias_node, weights_node, prod_node, output_node, expected_output_node, \
		cost_node]
	graph = Graph(nodes, [input_node], [output_node], [expected_output_node], cost_node, [weights_node])

	X, Y = or_dataset()
	Y = Y.reshape((len(Y), 1))
	for _ in range(100):
		print(graph.batch_gradient_descent([X], [Y], 1) / 100)
	print(weights_node.w)
	evaluate(graph)
	visualize(graph, 100)
