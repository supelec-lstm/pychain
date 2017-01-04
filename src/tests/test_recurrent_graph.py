import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

import pytest
import numpy as np
from node import *
from graph import *
from recurrent_graph import *

def init_ones(shape):
	return np.ones(shape)

@pytest.fixture
def graph1():
	input_node = InputNode()
	expected_output = InputNode()
	substraction_node = SubstractionNode(expected_output, input_node)
	cost_node = Norm2Node(substraction_node)
	nodes = [input_node, expected_output, substraction_node, cost_node]
	return Graph(nodes, [input_node], [input_node], [expected_output], cost_node, [])

@pytest.fixture
def graph2():
	input_node = InputNode()
	learnable_node = LearnableNode(init_ones((1, 1)))
	multiplication_node = MultiplicationNode(input_node, learnable_node)
	expected_output = InputNode()
	substraction_node = SubstractionNode(expected_output, multiplication_node)
	cost_node = Norm2Node(substraction_node)
	nodes = [input_node, learnable_node, multiplication_node, expected_output, substraction_node, cost_node]
	return Graph(nodes, [input_node], [multiplication_node], [expected_output], cost_node, [learnable_node])

@pytest.fixture
def graph3():
	input_node = InputNode()
	delay_once_node = DelayOnceNode()
	concatenation_node = ConcatenationNode(input_node, delay_once_node)
	learnable_node = LearnableNode(init_ones((2, 1)))
	multiplication_node = MultiplicationNode(concatenation_node, learnable_node)
	expected_output = InputNode()
	substraction_node = SubstractionNode(expected_output, multiplication_node)
	cost_node = Norm2Node(substraction_node)
	# set the recurrence
	delay_once_node.set_parents([multiplication_node])
	nodes = [input_node, delay_once_node, concatenation_node, learnable_node, \
		     multiplication_node, expected_output, substraction_node, cost_node]
	return Graph(nodes, [input_node], [multiplication_node], [expected_output], cost_node, [learnable_node])

@pytest.fixture
def graph4():
	# A deeper network
	input_node = InputNode()
	delay_once_node = DelayOnceNode()
	concatenation_node = ConcatenationNode(input_node, delay_once_node)
	learnable_node1 = LearnableNode(init_ones((3, 2)))
	multiplication_node1 = MultiplicationNode(concatenation_node, learnable_node1)
	hidden_node = TanhNode(multiplication_node1)

	learnable_node2 = LearnableNode(init_ones((2, 1)))
	multiplication_node2 = MultiplicationNode(hidden_node, learnable_node2)
	sigmoid_node = SigmoidNode(multiplication_node2)
	expected_output = InputNode()
	substraction_node = SubstractionNode(expected_output, sigmoid_node)
	cost_node = Norm2Node(substraction_node)
	# set the recurrence
	delay_once_node.set_parents([hidden_node])

	nodes = [input_node, delay_once_node, concatenation_node, learnable_node1, \
		     multiplication_node1, hidden_node, learnable_node2, multiplication_node2, \
		     sigmoid_node, expected_output, substraction_node, cost_node]
	return Graph(nodes, [input_node], [sigmoid_node], [expected_output], cost_node, [learnable_node1, learnable_node2])

def test_create_nodes1(graph1):
	recurrent_graph = RecurrentGraph(graph1)
	unfolded_graph, matchings = recurrent_graph.create_nodes(3)
	nodes = unfolded_graph.nodes
	assert matchings == [{0: nodes[0], 1: nodes[1], 2: nodes[2], 3: nodes[3]}, \
						 {0: nodes[4], 1: nodes[5], 2: nodes[6], 3: nodes[7]}, \
						 {0: nodes[8], 1: nodes[9], 2: nodes[10], 3: nodes[11]}]
	assert len(nodes) == 13
	assert unfolded_graph.input_nodes == [nodes[0], nodes[4], nodes[8]]
	assert unfolded_graph.output_nodes == [nodes[0], nodes[4], nodes[8]]
	assert unfolded_graph.expected_output_nodes == [nodes[1], nodes[5], nodes[9]]
	assert unfolded_graph.cost_node == nodes[12]
	assert unfolded_graph.learnable_nodes == []

def test_create_nodes2(graph2):
	recurrent_graph = RecurrentGraph(graph2)
	unfolded_graph, matchings = recurrent_graph.create_nodes(3)
	nodes = unfolded_graph.nodes
	assert matchings == [{0: nodes[1], 1: nodes[0], 2: nodes[2], 3: nodes[3], 4: nodes[4], 5: nodes[5]}, \
						 {0: nodes[6], 1: nodes[0], 2: nodes[7], 3: nodes[8], 4: nodes[9], 5: nodes[10]}, \
						 {0: nodes[11], 1: nodes[0], 2: nodes[12], 3: nodes[13], 4: nodes[14], 5: nodes[15]}]
	assert len(nodes) == 17
	assert unfolded_graph.input_nodes == [nodes[1], nodes[6], nodes[11]]
	assert unfolded_graph.output_nodes == [nodes[2], nodes[7], nodes[12]]
	assert unfolded_graph.expected_output_nodes == [nodes[3], nodes[8], nodes[13]]
	assert unfolded_graph.cost_node == nodes[16]
	assert unfolded_graph.learnable_nodes == [nodes[0]]

def test_create_nodes3(graph3):
	recurrent_graph = RecurrentGraph(graph3, [(1, 2)])
	unfolded_graph, matchings = recurrent_graph.create_nodes(3)
	nodes = unfolded_graph.nodes
	assert matchings == [{0: nodes[1], 1: nodes[20], 2: nodes[2], 3: nodes[0], 4: nodes[3], 5: nodes[4], 6: nodes[5], 7: nodes[6]}, \
						 {0: nodes[7], 1: nodes[3], 2: nodes[8], 3: nodes[0], 4: nodes[9], 5: nodes[10], 6: nodes[11], 7: nodes[12]}, \
						 {0: nodes[13], 1: nodes[9], 2: nodes[14], 3: nodes[0], 4: nodes[15], 5: nodes[16], 6: nodes[17], 7: nodes[18]}]
	assert len(nodes) == 21
	assert unfolded_graph.input_nodes == [nodes[1], nodes[7], nodes[13], nodes[20]]
	assert unfolded_graph.output_nodes == [nodes[3], nodes[9], nodes[15], nodes[15]]
	assert unfolded_graph.expected_output_nodes == [nodes[4], nodes[10], nodes[16]]
	assert unfolded_graph.cost_node == nodes[19]
	assert unfolded_graph.learnable_nodes == [nodes[0]]

def test_create_nodes4(graph4):
	recurrent_graph = RecurrentGraph(graph4, [(1, 1)])
	unfolded_graph, matchings = recurrent_graph.create_nodes(2)
	nodes = unfolded_graph.nodes
	assert matchings == [{0: nodes[2], 1: nodes[21], 2: nodes[3], 3: nodes[0], 4: nodes[4], 5: nodes[5], \
						 6: nodes[1], 7: nodes[6], 8: nodes[7], 9: nodes[8], 10: nodes[9], 11: nodes[10]}, \
						{0: nodes[11], 1: nodes[5], 2: nodes[12], 3: nodes[0], 4: nodes[13], 5: nodes[14], \
						 6: nodes[1], 7: nodes[15], 8: nodes[16], 9: nodes[17], 10: nodes[18], 11: nodes[19]}]
	assert len(nodes) == 22

def test_create_links1(graph1):
	recurrent_graph = RecurrentGraph(graph1)
	unfolded_graph = recurrent_graph.unfold(3)
	nodes = unfolded_graph.nodes
	# 1
	assert nodes[0].parents == [] and nodes[0].children == [(nodes[2], 1)]
	assert nodes[1].parents == [] and nodes[1].children == [(nodes[2], 0)]
	assert nodes[2].parents == [nodes[1], nodes[0]] and nodes[2].children == [(nodes[3], 0)]
	assert nodes[3].parents == [nodes[2]] and nodes[3].children == [(nodes[12], 0)]
	# 2
	assert nodes[4].parents == [] and nodes[4].children == [(nodes[6], 1)]
	assert nodes[5].parents == [] and nodes[5].children == [(nodes[6], 0)]
	assert nodes[6].parents == [nodes[5], nodes[4]] and nodes[6].children == [(nodes[7], 0)]
	assert nodes[7].parents == [nodes[6]] and nodes[7].children == [(nodes[12], 1)]
	# 3
	assert nodes[8].parents == [] and nodes[8].children == [(nodes[10], 1)]
	assert nodes[9].parents == [] and nodes[9].children == [(nodes[10], 0)]
	assert nodes[10].parents == [nodes[9], nodes[8]] and nodes[10].children == [(nodes[11], 0)]
	assert nodes[11].parents == [nodes[10]] and nodes[11].children == [(nodes[12], 2)]
	# cost node
	assert nodes[12].parents == [nodes[3], nodes[7], nodes[11]]

def test_create_links2(graph2):
	recurrent_graph = RecurrentGraph(graph2)
	unfolded_graph = recurrent_graph.unfold(3)
	nodes = unfolded_graph.nodes
	# weights
	assert nodes[0].parents == [] and nodes[0].children == [(nodes[2], 1), (nodes[7], 1), (nodes[12], 1)]
	# 1
	assert nodes[1].parents == [] and nodes[1].children == [(nodes[2], 0)]
	assert nodes[2].parents == [nodes[1], nodes[0]] and nodes[2].children == [(nodes[4], 1)]
	assert nodes[3].parents == [] and nodes[3].children == [(nodes[4], 0)]
	assert nodes[4].parents == [nodes[3], nodes[2]] and nodes[4].children == [(nodes[5], 0)]
	assert nodes[5].parents == [nodes[4]] and nodes[5].children == [(nodes[16], 0)]
	# 2
	assert nodes[6].parents == [] and nodes[6].children == [(nodes[7], 0)]
	assert nodes[7].parents == [nodes[6], nodes[0]] and nodes[7].children == [(nodes[9], 1)]
	assert nodes[8].parents == [] and nodes[8].children == [(nodes[9], 0)]
	assert nodes[9].parents == [nodes[8], nodes[7]] and nodes[9].children == [(nodes[10], 0)]
	assert nodes[10].parents == [nodes[9]] and nodes[10].children == [(nodes[16], 1)]
	# 3
	assert nodes[11].parents == [] and nodes[11].children == [(nodes[12], 0)]
	assert nodes[12].parents == [nodes[11], nodes[0]] and nodes[12].children == [(nodes[14], 1)]
	assert nodes[13].parents == [] and nodes[13].children == [(nodes[14], 0)]
	assert nodes[14].parents == [nodes[13], nodes[12]] and nodes[14].children == [(nodes[15], 0)]
	assert nodes[15].parents == [nodes[14]] and nodes[15].children == [(nodes[16], 2)]
	# cost node
	assert nodes[16].parents == [nodes[5], nodes[10], nodes[15]]

def test_create_links3(graph3):
	recurrent_graph = RecurrentGraph(graph3, [(1, 1)])
	unfolded_graph = recurrent_graph.unfold(3)
	nodes = unfolded_graph.nodes
	# weights
	assert nodes[0].parents == [] and nodes[0].children == [(nodes[3], 1), (nodes[9], 1), (nodes[15], 1)]
	# 1
	assert nodes[1].parents == [] and nodes[1].children == [(nodes[2], 0)]
	assert nodes[2].parents == [nodes[1], nodes[20]] and nodes[2].children == [(nodes[3], 0)]
	assert nodes[3].parents == [nodes[2], nodes[0]] and nodes[3].children == [(nodes[5], 1), (nodes[8], 1)]
	assert nodes[4].parents == [] and nodes[4].children == [(nodes[5], 0)]
	assert nodes[5].parents == [nodes[4], nodes[3]] and nodes[5].children == [(nodes[6], 0)]
	assert nodes[6].parents == [nodes[5]] and nodes[6].children == [(nodes[19], 0)]
	# 2
	assert nodes[7].parents == [] and nodes[7].children == [(nodes[8], 0)]
	assert nodes[8].parents == [nodes[7], nodes[3]] and nodes[8].children == [(nodes[9], 0)]
	assert nodes[9].parents == [nodes[8], nodes[0]] and nodes[9].children == [(nodes[11], 1), (nodes[14], 1)]
	assert nodes[10].parents == [] and nodes[10].children == [(nodes[11], 0)]
	assert nodes[11].parents == [nodes[10], nodes[9]] and nodes[11].children == [(nodes[12], 0)]
	assert nodes[12].parents == [nodes[11]] and nodes[12].children == [(nodes[19], 1)]
	# 3
	assert nodes[13].parents == [] and nodes[13].children == [(nodes[14], 0)]
	assert nodes[14].parents == [nodes[13], nodes[9]] and nodes[14].children == [(nodes[15], 0)]
	assert nodes[15].parents == [nodes[14], nodes[0]]and nodes[15].children == [(nodes[17], 1)]
	assert nodes[16].parents == [] and nodes[16].children == [(nodes[17], 0)]
	assert nodes[17].parents == [nodes[16], nodes[15]] and nodes[17].children == [(nodes[18], 0)]
	assert nodes[18].parents == [nodes[17]] and nodes[18].children == [(nodes[19], 2)]
	# cost node
	assert nodes[19].parents == [nodes[6], nodes[12], nodes[18]]
	# hidden nodes
	assert nodes[20].parents == [] and nodes[20].children == [(nodes[2], 1)]

def test_create_links4(graph4):
	recurrent_graph = RecurrentGraph(graph4, [(1, 1)])
	unfolded_graph = unfolded_graph = recurrent_graph.unfold(2)
	nodes = unfolded_graph.nodes
	# weights
	assert nodes[0].parents == [] and nodes[0].children == [(nodes[4], 1), (nodes[13], 1)]
	assert nodes[1].parents == [] and nodes[1].children == [(nodes[6], 1), (nodes[15], 1)]
	# 1
	assert nodes[2].parents == [] and nodes[2].children == [(nodes[3], 0)]
	assert nodes[3].parents == [nodes[2], nodes[21]] and nodes[3].children == [(nodes[4], 0)]
	assert nodes[4].parents == [nodes[3], nodes[0]] and nodes[4].children == [(nodes[5], 0)]
	assert nodes[5].parents == [nodes[4]] and nodes[5].children == [(nodes[6], 0), (nodes[12], 1)]
	assert nodes[6].parents == [nodes[5], nodes[1]] and nodes[6].children == [(nodes[7], 0)]
	assert nodes[7].parents == [nodes[6]] and nodes[7].children == [(nodes[9], 1)]
	assert nodes[8].parents == [] and nodes[8].children == [(nodes[9], 0)]
	assert nodes[9].parents == [nodes[8], nodes[7]] and nodes[9].children == [(nodes[10], 0)]
	assert nodes[10].parents == [nodes[9]] and nodes[10].children == [(nodes[20], 0)]
	# 2
	assert nodes[11].parents == [] and nodes[11].children == [(nodes[12], 0)]
	assert nodes[12].parents == [nodes[11], nodes[5]] and nodes[12].children == [(nodes[13], 0)]
	assert nodes[13].parents == [nodes[12], nodes[0]] and nodes[13].children == [(nodes[14], 0)]
	assert nodes[14].parents == [nodes[13]] and nodes[14].children == [(nodes[15], 0)]
	assert nodes[15].parents == [nodes[14], nodes[1]] and nodes[15].children == [(nodes[16], 0)]
	assert nodes[16].parents == [nodes[15]] and nodes[16].children == [(nodes[18], 1)]
	assert nodes[17].parents == [] and nodes[17].children == [(nodes[18], 0)]
	assert nodes[18].parents == [nodes[17], nodes[16]] and nodes[18].children == [(nodes[19], 0)]
	assert nodes[19].parents == [nodes[18]] and nodes[19].children == [(nodes[20], 1)]
	# cost
	assert nodes[20].parents == [nodes[10], nodes[19]]
	# hidden nodes
	assert nodes[21].parents == [] and nodes[21].children == [(nodes[3], 1)]

def test_weights(graph3):
	recurrent_graph = RecurrentGraph(graph3)
	assert len(recurrent_graph.weights) == 1
	assert np.array_equal(recurrent_graph.weights[0], np.array([[1], [1]]))

def test_propagate1(graph3):
	recurrent_graph = RecurrentGraph(graph3, [(1, 1)])
	output = recurrent_graph.propagate([np.array([[2]])])
	assert output == [np.array([[2]])]

	output = recurrent_graph.propagate([np.array([[2]]), np.array([[5]])])
	assert output == [np.array([[2]]), np.array([[7]])]

	output = recurrent_graph.propagate([np.array([[2]]), np.array([[5]]), np.array([[3]])])
	assert output == [np.array([[2]]), np.array([[7]]), np.array([[10]])]

def test_backpropagate1(graph3):
	recurrent_graph = RecurrentGraph(graph3, [(1, 1)])
	cost = recurrent_graph.backpropagate([np.array([[2]])], [np.array([[5]])], 0.1)
	assert cost == 9
	assert np.array_equal(recurrent_graph.weights[0], np.array([[2.2], [1]]))

def test_backpropagate2(graph3):
	recurrent_graph = RecurrentGraph(graph3, [(1, 1)])
	cost = recurrent_graph.backpropagate([np.array([[2]]), np.array([[1]])], [np.array([[5]]), np.array([[2]])], 0.1)
	assert cost == 10
	assert np.array_equal(recurrent_graph.weights[0], np.array([[1.6], [0.6]]))

def test_batch_backpropagate(graph3):
	recurrent_graph = RecurrentGraph(graph3, [(1, 1)])
	costs = recurrent_graph.batch_backpropagate([np.array([[2]]), np.array([[1]])], [np.array([[5]]), np.array([[2]])], 0.1, 1)
	assert np.allclose(costs, [9, 4.84])
	assert np.allclose(recurrent_graph.weights[0], np.array([[1.76], [0.12]]))

def sigmoid(t):
	return 1 / (1 + np.exp(-t))

def test_propagate2(graph4):
	recurrent_graph = RecurrentGraph(graph4, [(1, 2)])
	output = recurrent_graph.propagate([np.array([[2]]), np.array([[5]])])
	output1 = sigmoid(2*np.tanh(2))
	output2 = sigmoid(2*np.tanh(5+2*np.tanh(2)))
	assert output == [np.array([[output1]]), np.array([[output2]])]

def test_backpropagate3(graph4):
	recurrent_graph = RecurrentGraph(graph4, [(1, 2)])
	cost = recurrent_graph.backpropagate([np.array([[2]]), np.array([[5]])], [np.array([[3]]), np.array([[2]])], 0.1)
	h1 = np.tanh(2)
	h2 = np.tanh(5+2*np.tanh(2))
	y1 = sigmoid(2*h1)
	y2 = sigmoid(2*h2)
	e1 = 3 - y1
	e2 = 2 - y2
	expected_cost = np.square(e1) + np.square(e2)
	assert cost == expected_cost

	t1 = -2*e1*y1*(1-y1)
	t2 = -2*e2*y2*(1-y2)
	grad1_w2 = np.array([[t1*h1], [t1*h1]])
	grad2_w2 = np.array([[t2*h2], [t2*h2]])
	expected_w2 = np.ones((2, 1)) - 0.1 * (grad1_w2 + grad2_w2)
	assert np.allclose(recurrent_graph.weights[1], expected_w2)

	u1 = (t1 + 2*t2*(1-h2**2))*(1-h1**2)
	u2 = t2*(1-h2**2)
	grad1_w1 = np.array([[2*u1, 2*u1], [0, 0], [0, 0]])
	grad2_w1 = np.array([[5*u2, 5*u2], [np.tanh(2)*u2, np.tanh(2)*u2], [np.tanh(2)*u2, np.tanh(2)*u2]])
	expected_w1 = np.ones((3, 2)) - 0.1 * (grad1_w1 + grad2_w1)
	assert np.allclose(recurrent_graph.weights[0], expected_w1)