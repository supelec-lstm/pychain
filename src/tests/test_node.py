import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

import numpy as np
from node import *

def test_add_child():
	node1 = Node()
	node2 = Node([node1])
	node3 = Node([node1])
	assert node1.children == [(node2, 0), (node3, 0)]

def test_input_node():
	value = np.array([[1,1,1],[2,2,2]])
	input_node = InputNode(value)
	assert np.array_equal(input_node.evaluate(), value)

	value2 = np.array([4,2])
	input_node.set_value(value2)
	assert np.array_equal(input_node.evaluate(), value2)

def test_output_node():
	input_node = InputNode()
	node = ConstantGradientNode([input_node])
	assert node.get_gradient(0) == 1

def test_norm2_node():
	value = np.array([[1, 2], [3, 4]])	
	input_node = InputNode(value)
	norm2_node = Norm2Node(input_node)
	ouput_node = ConstantGradientNode([norm2_node])
	assert norm2_node.evaluate() == 30

	assert np.array_equal(norm2_node.get_gradient(0), 2*value.T)

def init_ones(shape):
	return np.ones(shape)

def test_learnable_node():	
	learnable_node = LearnableNode((3, 2), init_ones)
	norm2_node = Norm2Node(learnable_node)
	ouput_node = ConstantGradientNode([norm2_node])
	assert np.array_equal(learnable_node.evaluate(), np.ones((3, 2)))
	assert norm2_node.evaluate() == 6

	assert np.array_equal(norm2_node.get_gradient(0), 2*np.ones((2, 3)))
	assert np.array_equal(learnable_node.get_gradient(0), 2*np.ones((2, 3)))

	learnable_node.reset_memoization()
	learnable_node.get_gradient(0)
	assert np.array_equal(learnable_node.acc_dJdw, 4*np.ones((2, 3)))

	learnable_node.descend_gradient(0.7, 13)
	assert np.array_equal(learnable_node.w, (1 - 4*0.7/13) * np.ones((3, 2)))

	learnable_node.reset_accumulator()
	assert np.array_equal(learnable_node.acc_dJdw, np.zeros((2, 3)))

def test_addition_node():
	node_in1 = InputNode(np.array([[1, 1], [2, 2]]))
	node_in2 = InputNode(np.array([[1, 2], [3, 4]]))
	node_add = AdditionNode(node_in1, node_in2)
	node_fun = Norm2Node(node_add)
	node_out = ConstantGradientNode([node_fun])

	assert np.array_equal(node_add.evaluate(), np.array([[2, 3], [5, 6]]))
	assert node_fun.evaluate() == 74
	assert np.array_equal(node_add.get_gradient(0), np.array([[4, 10], [6, 12]]))

def test_multiplication_node():
	node_in1 = InputNode(np.array([[1, 1], [2, 2]]))
	node_in2 = InputNode(np.array([[1, 2], [3, 4]]))
	node_dot = MultiplicationNode(node_in1, node_in2)
	node_fun = Norm2Node(node_dot)
	node_out = ConstantGradientNode([node_fun])

	assert np.array_equal(node_dot.evaluate(), np.array([[4, 6], [8, 12]]))
	assert node_fun.evaluate() == 260
	assert np.array_equal(node_dot.get_gradient(0), np.array([[32, 64], [72, 144]]))

def test_cross_entropy_node():
	node_in1 = InputNode(np.array([[1, 1], [2, 2]]))
	node_in2 = InputNode(np.array([[1, 2], [3, 4]]))
	node_sce = SoftmaxCrossEntropyNode(node_in1, node_in2)
	node_fun = Norm2Node(node_sce)
	node_out = ConstantGradientNode([node_fun])

	assert node_sce.evaluate() == - (0 + np.log(2) + 2*np.log(3) + 2*np.log(4))
	node_fun.evaluate()
	assert np.array_equal(node_sce.get_gradient(0), np.array([[0, -np.log(3)], [-np.log(2), -np.log(4)]]))
	assert np.array_equal(node_sce.get_gradient(1), -np.array([[1, 2/3], [1/2, 2/4]]))