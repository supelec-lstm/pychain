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

	value = np.array([4,2])
	input_node.set_value(value)
	assert np.array_equal(input_node.evaluate(), value)

def test_output_node():
	input_node = InputNode()
	node = ConstantGradientNode([input_node])
	assert node.get_gradient(0) == 1

def test_norm2_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])	
	input_node = InputNode(value)
	norm2_node = Norm2Node(input_node)
	output_node = ConstantGradientNode([norm2_node])

	assert norm2_node.evaluate() == 91
	assert np.array_equal(norm2_node.get_gradient(0), 2*value)

def test_add_bias_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])	
	input_node = InputNode(value)
	add_bias_node = AddBiasNode(input_node)
	norm2_node = Norm2Node(add_bias_node)
	output_node = ConstantGradientNode([norm2_node])

	assert np.array_equal(add_bias_node.evaluate(), np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]]))
	assert norm2_node.evaluate() == 94
	assert np.array_equal(add_bias_node.get_gradient(0), 2*value)

def test_sigmoid_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])		
	input_node = InputNode(value)
	sigmoid_node = SigmoidNode(input_node)
	norm2_node = Norm2Node(sigmoid_node)
	output_node = ConstantGradientNode([norm2_node])

	y = 1/(1+np.exp(-value))
	assert np.array_equal(sigmoid_node.evaluate(), y)
	norm2_node.evaluate()
	assert np.allclose(sigmoid_node.get_gradient(0), 2*y*y*(1-y))

def test_tanh_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])	
	input_node = InputNode(value)
	tanh_node = TanhNode(input_node)
	norm2_node = Norm2Node(tanh_node)
	output_node = ConstantGradientNode([norm2_node])

	y = np.tanh(value)
	assert np.array_equal(tanh_node.evaluate(), y)
	norm2_node.evaluate()
	assert np.array_equal(tanh_node.get_gradient(0), 2*y*(1-y*y))

def test_relu_node():
	value = np.array([[-1, 2], [3, -4], [-5, 6]])	
	input_node = InputNode(value)
	relu_node = ReluNode(input_node)
	norm2_node = Norm2Node(relu_node)
	output_node = ConstantGradientNode([norm2_node])

	y = np.array([[0, 2], [3, 0], [0, 6]])
	assert np.array_equal(relu_node.evaluate(), y)
	norm2_node.evaluate()
	assert np.array_equal(relu_node.get_gradient(0), 2*y*np.array([[0, 1], [1, 0], [0, 1]]))

def test_softmax_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])		
	input_node = InputNode(value)
	softmax_node = SoftmaxNode(input_node)
	norm2_node = Norm2Node(softmax_node)
	output_node = ConstantGradientNode([norm2_node])

	sum1, sum2, sum3 = np.exp(1)+np.exp(2), np.exp(3)+np.exp(4), np.exp(5)+np.exp(6)
	y11, y12 = np.exp(1)/sum1, np.exp(2)/sum1 
	y21, y22 = np.exp(3)/sum2, np.exp(4)/sum2 
	y31, y32 = np.exp(5)/sum3, np.exp(6)/sum3
	y = np.array([[y11, y12], [y21, y22], [y31, y32]])
	assert np.array_equal(softmax_node.evaluate(), y)
	norm2_node.evaluate()
	dx11 = 2*(y11*(1-y11)*y11 - y11*y12*y12)
	dx12 = 2*(-y11*y12*y11 + y12*(1-y12)*y12)
	dx21 = 2*(y21*(1-y21)*y21 - y21*y22*y22)
	dx22 = 2*(-y21*y22*y21 + y22*(1-y22)*y22)
	dx31 = 2*(y31*(1-y31)*y31 - y31*y32*y32)
	dx32 = 2*(-y31*y32*y31 + y32*(1-y32)*y32)
	assert np.allclose(softmax_node.get_gradient(0), np.array([[dx11, dx12], [dx21, dx22], [dx31, dx32]]))

def scalar_multiplication_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])	
	input_node = InputNode(value)
	scalar_node = ScalarMultiplicationNode(input_node, 3)
	norm2_node = Norm2Node(add_bias_node)
	output_node = ConstantGradientNode([norm2_node])

	assert np.array_equal(add_bias_node.evaluate(), 3*value)
	norm2_node.evaluate()
	assert np.array_equal(add_bias_node.get_gradient(0), 6*value)

def init_ones(shape):
	return np.ones(shape)

def test_learnable_node():	
	learnable_node = LearnableNode((3, 2), init_ones)
	norm2_node = Norm2Node(learnable_node)
	output_node = ConstantGradientNode([norm2_node])

	assert np.array_equal(learnable_node.evaluate(), np.ones((3, 2)))
	norm2_node.evaluate()
	learnable_node.get_gradient(0)
	assert np.array_equal(learnable_node.acc_dJdw, 2*np.ones((3, 2)))

	learnable_node.reset_memoization()
	learnable_node.get_gradient(0)
	assert np.array_equal(learnable_node.acc_dJdw, 4*np.ones((3, 2)))

	learnable_node.descend_gradient(0.7, 13)
	assert np.array_equal(learnable_node.w, (1 - 4*0.7/13) * np.ones((3, 2)))

	learnable_node.reset_accumulator()
	assert np.array_equal(learnable_node.acc_dJdw, np.zeros((3, 2)))

def test_addition_node():
	node_in1 = InputNode(np.array([[1, 1], [2, 2], [3, 3]]))
	node_in2 = InputNode(np.array([[1, 2], [3, 4], [5, 6]]))
	node_add = AdditionNode(node_in1, node_in2)
	node_fun = Norm2Node(node_add)
	node_out = ConstantGradientNode([node_fun])

	y = np.array([[2, 3], [5, 6], [8, 9]])
	assert np.array_equal(node_add.evaluate(), y)
	node_fun.evaluate()
	assert np.array_equal(node_add.get_gradient(0), 2*y)
	assert np.array_equal(node_add.get_gradient(1), 2*y)

def test_substraction_node():
	node_in1 = InputNode(np.array([[1, 1], [2, 2], [3, 3]]))
	node_in2 = InputNode(np.array([[1, 2], [3, 4], [5, 6]]))
	node_sub = SubstractionNode(node_in1, node_in2)
	node_fun = Norm2Node(node_sub)
	node_out = ConstantGradientNode([node_fun])

	y = np.array([[0, -1], [-1, -2], [-2, -3]])
	assert np.array_equal(node_sub.evaluate(), y)
	node_fun.evaluate()
	assert np.array_equal(node_sub.get_gradient(0), 2*y)
	assert np.array_equal(node_sub.get_gradient(1), -2*y)

def test_multiplication_node():
	node_in1 = InputNode(np.array([[1, 1], [2, 2]]))
	node_in2 = InputNode(np.array([[1, 2], [3, 4]]))
	node_dot = MultiplicationNode(node_in1, node_in2)
	node_fun = Norm2Node(node_dot)
	node_out = ConstantGradientNode([node_fun])

	assert np.array_equal(node_dot.evaluate(), np.array([[4, 6], [8, 12]]))
	node_fun.evaluate()
	assert np.array_equal(node_dot.get_gradient(0), np.array([[32, 72], [64, 144]]))
	assert np.array_equal(node_dot.get_gradient(1), np.array([[40, 60], [40, 60]]))

def test_softmax_cross_entropy_node():
	node_in1 = InputNode(np.array([[1, 1], [2, 2]]))
	node_in2 = InputNode(np.array([[1, 2], [3, 4]]))
	node_sce = SoftmaxCrossEntropyNode(node_in1, node_in2)
	node_out = ConstantGradientNode([node_sce])

	assert node_sce.evaluate() == - (0 + np.log(2) + 2*np.log(3) + 2*np.log(4))
	node_sce.evaluate()
	assert np.array_equal(node_sce.get_gradient(0), np.array([[0, -np.log(2)], [-np.log(3), -np.log(4)]]))
	assert np.array_equal(node_sce.get_gradient(1), -np.array([[1, 1/2], [2/3, 2/4]]))

def test_sigmoid_cross_entropy_node():
	node_in1 = InputNode(np.array([[0], [1]]))
	node_in2 = InputNode(np.array([[0.3], [0.6]]))
	node_sce = SigmoidCrossEntropyNode(node_in1, node_in2)
	node_fun = Norm2Node(node_sce)
	node_out = ConstantGradientNode([node_fun])

	#print(node_sce.evaluate())
	#assert node_sce.evaluate() == -(np.log(0.7) + np.log(0.6))
	#node_fun.evaluate()
	#assert np.array_equal(node_sce.get_gradient(0), np.array([[0, -np.log(3)], [-np.log(2), -np.log(4)]]))
	#assert np.array_equal(node_sce.get_gradient(1), -np.array([[1, 2/3], [1/2, 2/4]]))