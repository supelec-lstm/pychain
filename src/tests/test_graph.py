import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

import numpy as np
from node import *
from graph import *

def init_ones(shape):
	return np.ones(shape)

def test_simple_graph():
	input_node = InputNode()
	weights_node = LearnableNode((3, 1), init_ones)
	output_node = MultiplicationNode(input_node, weights_node)
	expected_output_node = InputNode()
	sub_node = SubstractionNode(output_node, expected_output_node)
	cost_node = Norm2Node(sub_node)
	nodes = [input_node, weights_node, output_node, expected_output_node, sub_node, cost_node]
	graph = Graph(nodes, input_node, output_node, expected_output_node, cost_node, [weights_node])

	x1 = np.array([[1, 2, 3]])
	y1 = np.array([[5]])
	assert graph.propagate(x1) == 6
	cost = graph.backpropagate(y1)
	assert cost == 1
	assert np.array_equal(weights_node.acc_dJdw, np.array([[2, 4, 6]])) 

	x2 = np.array([[1, 2, 3], [4, 5, 6]])
	y2 = np.array([[5], [12]])
	assert np.array_equal(graph.propagate(x2), np.array([[6], [15]]))
	cost = graph.backpropagate(y2)
	assert cost == 10
	# [2, 4, 6] + [26, 34, 42]
	assert np.array_equal(weights_node.acc_dJdw, np.array([[28, 38, 48]])) 

