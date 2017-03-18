import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

import pytest
import numpy as np
from node import *
from graph import *
from layer import *
from recurrent_graph import *
from optimization_algorithm import *

def init_ones(shape):
	return np.ones(shape)

@pytest.fixture
def layer1():
	input_node = InputNode()
	expected_output = InputNode()
	substraction_node = SubstractionNode([expected_output, input_node])
	cost_node = Norm2Node([substraction_node])
	nodes = [input_node, expected_output, substraction_node, cost_node]
	return Layer(nodes, [input_node], [input_node], [], [], [expected_output], cost_node, [])

@pytest.fixture
def layer2():
	input_node = InputNode()
	learnable_node = LearnableNode(init_ones((1, 1)))
	multiplication_node = MultiplicationNode([input_node, learnable_node])
	expected_output = InputNode()
	substraction_node = SubstractionNode([expected_output, multiplication_node])
	cost_node = Norm2Node([substraction_node])
	nodes = [input_node, learnable_node, multiplication_node, expected_output, substraction_node, cost_node]
	return Layer(nodes, [input_node], [multiplication_node], [], [], [expected_output], \
		cost_node, [learnable_node])

@pytest.fixture
def layer3():
	input_node = InputNode()
	hidden_input_node = InputNode()
	concatenation_node = ConcatenationNode([input_node, hidden_input_node])
	learnable_node = LearnableNode(init_ones((2, 1)))
	multiplication_node = MultiplicationNode([concatenation_node, learnable_node])
	expected_output = InputNode()
	substraction_node = SubstractionNode([expected_output, multiplication_node])
	cost_node = Norm2Node([substraction_node])
	nodes = [input_node, hidden_input_node, concatenation_node, learnable_node, \
		multiplication_node, expected_output, substraction_node, cost_node]
	return Layer(nodes, [input_node], [multiplication_node], [hidden_input_node], [multiplication_node], \
		[expected_output], cost_node, [learnable_node])

@pytest.fixture
def layer4():
	# A deeper network
	input_node = InputNode()
	hidden_input_node = InputNode()
	concatenation_node = ConcatenationNode([input_node, hidden_input_node])
	learnable_node1 = LearnableNode(init_ones((3, 2)))
	multiplication_node1 = MultiplicationNode([concatenation_node, learnable_node1])
	hidden_output_node = TanhNode([multiplication_node1])

	learnable_node2 = LearnableNode(init_ones((2, 1)))
	multiplication_node2 = MultiplicationNode([hidden_output_node, learnable_node2])
	sigmoid_node = SigmoidNode([multiplication_node2])
	expected_output = InputNode()
	substraction_node = SubstractionNode([expected_output, sigmoid_node])
	cost_node = Norm2Node([substraction_node])

	nodes = [input_node, hidden_input_node, concatenation_node, learnable_node1, \
		multiplication_node1, hidden_output_node, learnable_node2, multiplication_node2, \
		sigmoid_node, expected_output, substraction_node, cost_node]
	return Layer(nodes, [input_node], [sigmoid_node], [hidden_input_node], [hidden_output_node], \
		[expected_output], cost_node, [learnable_node1, learnable_node2])

@pytest.fixture
def layer5():
	# for the comparison with RTRL
	input_node = InputNode()
	hidden_input_node = InputNode()
	concatenation_node = ConcatenationNode([input_node, hidden_input_node])
	learnable_node = LearnableNode(init_ones((2, 1)))
	multiplication_node = MultiplicationNode([concatenation_node, learnable_node])
	sigmoid_node = SigmoidNode([multiplication_node])
	expected_output = InputNode()
	substraction_node = SubstractionNode([expected_output, sigmoid_node])
	cost_node = Norm2Node([substraction_node])
	nodes = [input_node, hidden_input_node, concatenation_node, learnable_node, \
		     multiplication_node, sigmoid_node, expected_output, substraction_node, cost_node]
	return Layer(nodes, [input_node], [sigmoid_node], [hidden_input_node], [sigmoid_node], \
		[expected_output], cost_node, [learnable_node])

def test_propagate1(layer3):
	recurrent_graph = RecurrentGraph(layer3, 1, [(1, 1)])
	output = recurrent_graph.propagate([np.array([[2]])])
	assert output == [np.array([[2]])]

	recurrent_graph = RecurrentGraph(layer3, 2, [(1, 1)])
	output = recurrent_graph.propagate([np.array([[2]]), np.array([[5]])])
	assert output == [np.array([[2]]), np.array([[7]])]

	recurrent_graph = RecurrentGraph(layer3, 3, [(1, 1)])
	output = recurrent_graph.propagate([np.array([[2]]), np.array([[5]]), np.array([[3]])])
	assert output == [np.array([[2]]), np.array([[7]]), np.array([[10]])]

def test_backpropagate1(layer3):
	recurrent_graph = RecurrentGraph(layer3, 1, [(1, 1)])
	sgd = GradientDescent(recurrent_graph.get_learnable_nodes(), 0.1)
	recurrent_graph.propagate([np.array([[2]])])
	cost = recurrent_graph.backpropagate([np.array([[5]])])
	sgd.optimize()
	assert cost == 9
	assert np.array_equal(recurrent_graph.layers[0].learnable_nodes[0].w, np.array([[2.2], [1]]))

def test_backpropagate2(layer3):
	recurrent_graph = RecurrentGraph(layer3, 2, [(1, 1)])
	sgd = GradientDescent(recurrent_graph.get_learnable_nodes(), 0.1)
	recurrent_graph.propagate([np.array([[2]]), np.array([[1]])])
	cost = recurrent_graph.backpropagate([np.array([[5]]), np.array([[2]])])
	sgd.optimize()
	assert cost == 10
	assert np.array_equal(recurrent_graph.layers[0].learnable_nodes[0].w, np.array([[1.6], [0.6]]))

def sigmoid(t):
	return 1 / (1 + np.exp(-t))

def test_propagate2(layer4):
	recurrent_graph = RecurrentGraph(layer4, 2, [(1, 2)])
	output = recurrent_graph.propagate([np.array([[2]]), np.array([[5]])])
	output1 = sigmoid(2*np.tanh(2))
	output2 = sigmoid(2*np.tanh(5+2*np.tanh(2)))
	assert output == [np.array([[output1]]), np.array([[output2]])]

def test_backpropagate3(layer4):
	recurrent_graph = RecurrentGraph(layer4, 2, [(1, 2)])
	sgd = GradientDescent(recurrent_graph.get_learnable_nodes(), 0.1)
	recurrent_graph.propagate([np.array([[2]]), np.array([[5]])])
	cost = recurrent_graph.backpropagate([np.array([[3]]), np.array([[2]])])
	sgd.optimize()

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
	assert np.allclose(recurrent_graph.layers[0].learnable_nodes[1].w, expected_w2)

	u1 = (t1 + 2*t2*(1-h2**2))*(1-h1**2)
	u2 = t2*(1-h2**2)
	grad1_w1 = np.array([[2*u1, 2*u1], [0, 0], [0, 0]])
	grad2_w1 = np.array([[5*u2, 5*u2], [np.tanh(2)*u2, np.tanh(2)*u2], [np.tanh(2)*u2, np.tanh(2)*u2]])
	expected_w1 = np.ones((3, 2)) - 0.1 * (grad1_w1 + grad2_w1)
	assert np.allclose(recurrent_graph.layers[0].learnable_nodes[0].w, expected_w1)

def test_propagate3(layer5):
	recurrent_graph = RecurrentGraph(layer5, 2, [(1, 1)])
	output = recurrent_graph.propagate([np.array([[1]]), np.array([[2]])])
	output1 = sigmoid(1)
	output2 = sigmoid(2 + sigmoid(1))
	assert output == [np.array([[output1]]), np.array([[output2]])]

def test_backpropagation4(layer5):
	recurrent_graph = RecurrentGraph(layer5, 2, [(1, 1)])
	sgd = GradientDescent(recurrent_graph.get_learnable_nodes(), 1)
	output = recurrent_graph.propagate([np.array([[1]]), np.array([[2]])])
	cost = recurrent_graph.backpropagate([np.array([[2]]), np.array([[3]])])
	sgd.optimize()
	
	expected_cost = 5.8586149168896
	assert cost == expected_cost

	expected_weights = np.array([[ 2.01896294], [ 1.17305715]])
	assert np.allclose(recurrent_graph.layers[0].learnable_nodes[0].w, expected_weights)