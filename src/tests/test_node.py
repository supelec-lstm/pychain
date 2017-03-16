import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

import numpy as np
from node import *

def test_add_child():
	node1 = Node()
	node2 = Node([node1])
	node3 = Node([node1])
	assert node1.children == [[(node2, 0), (node3, 0)]]

def test_add_parents():
	node1 = Node()
	node2 = Node()
	node3 = Node()
	node3.set_parents([node1, node2])
	assert node3.parents == [(node1, 0), (node2, 0)]
	assert node1.children == [[(node3, 0)]]
	assert node2.children == [[(node3, 1)]]

def test_input_node():
	value = np.array([[1, 1, 1], [2, 2, 2]])
	input_node = InputNode(value)
	assert np.array_equal(input_node.get_output(), value)

	input_node.reset_memoization()
	value = np.array([[4, 2]])
	input_node.set_value(value)
	print(input_node.get_output())
	assert np.array_equal(input_node.get_output(), value)

def test_output_node():
	input_node = InputNode()
	node = GradientInputNode([input_node])
	assert node.get_gradient(0) == 1

def test_norm2_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])	
	input_node = InputNode(value)
	norm2_node = Norm2Node([input_node])
	output_node = GradientInputNode([norm2_node])

	assert norm2_node.get_output() == 91
	assert np.array_equal(norm2_node.get_gradient(0), 2*value)

def test_add_bias_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])	
	input_node = InputNode(value)
	add_bias_node = AddBiasNode([input_node])
	norm2_node = Norm2Node([add_bias_node])
	output_node = GradientInputNode([norm2_node])

	assert np.array_equal(add_bias_node.get_output(), np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]]))
	assert norm2_node.get_output() == 94
	assert np.array_equal(add_bias_node.get_gradient(0), 2*value)

def test_identity_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])		
	input_node = InputNode(value)
	id_node = IdentityNode([input_node])
	norm2_node = Norm2Node([id_node])
	output_node = GradientInputNode([norm2_node])

	assert np.array_equal(id_node.get_output(), value)
	norm2_node.get_output()
	assert np.allclose(id_node.get_gradient(0), 2*value)

def test_sigmoid_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])		
	input_node = InputNode(value)
	sigmoid_node = SigmoidNode([input_node])
	norm2_node = Norm2Node([sigmoid_node])
	output_node = GradientInputNode([norm2_node])

	y = 1/(1+np.exp(-value))
	assert np.array_equal(sigmoid_node.get_output(), y)
	norm2_node.get_output()
	assert np.allclose(sigmoid_node.get_gradient(0), 2*y*y*(1-y))

def test_tanh_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])	
	input_node = InputNode(value)
	tanh_node = TanhNode([input_node])
	norm2_node = Norm2Node([tanh_node])
	output_node = GradientInputNode([norm2_node])

	y = np.tanh(value)
	assert np.array_equal(tanh_node.get_output(), y)
	norm2_node.get_output()
	assert np.array_equal(tanh_node.get_gradient(0), 2*y*(1-y*y))

def test_relu_node():
	value = np.array([[-1, 2], [3, -4], [-5, 6]])	
	input_node = InputNode(value)
	relu_node = ReluNode([input_node])
	norm2_node = Norm2Node([relu_node])
	output_node = GradientInputNode([norm2_node])

	y = np.array([[0, 2], [3, 0], [0, 6]])
	assert np.array_equal(relu_node.get_output(), y)
	norm2_node.get_output()
	assert np.array_equal(relu_node.get_gradient(0), 2*y*np.array([[0, 1], [1, 0], [0, 1]]))

def test_softmax_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])		
	input_node = InputNode(value)
	softmax_node = SoftmaxNode([input_node])
	norm2_node = Norm2Node([softmax_node])
	output_node = GradientInputNode([norm2_node])

	sum1, sum2, sum3 = np.exp(1)+np.exp(2), np.exp(3)+np.exp(4), np.exp(5)+np.exp(6)
	y11, y12 = np.exp(1)/sum1, np.exp(2)/sum1 
	y21, y22 = np.exp(3)/sum2, np.exp(4)/sum2 
	y31, y32 = np.exp(5)/sum3, np.exp(6)/sum3
	y = np.array([[y11, y12], [y21, y22], [y31, y32]])
	assert np.array_equal(softmax_node.get_output(), y)
	norm2_node.get_output()
	dx11 = 2*(y11*(1-y11)*y11 - y12*y11*y12)
	dx12 = 2*(-y11*y12*y11 + y12*(1-y12)*y12)
	dx21 = 2*(y21*(1-y21)*y21 - y21*y22*y22)
	dx22 = 2*(-y21*y22*y21 + y22*(1-y22)*y22)
	dx31 = 2*(y31*(1-y31)*y31 - y31*y32*y32)
	dx32 = 2*(-y31*y32*y31 + y32*(1-y32)*y32)
	print(softmax_node.get_gradient(0))
	print(np.array([[dx11, dx12], [dx21, dx22], [dx31, dx32]]))
	assert np.allclose(softmax_node.get_gradient(0), np.array([[dx11, dx12], [dx21, dx22], [dx31, dx32]]))

def scalar_multiplication_node():
	value = np.array([[1, 2], [3, 4], [5, 6]])	
	input_node = InputNode(value)
	scalar_node = ScalarMultiplicationNode([input_node], 3)
	norm2_node = Norm2Node([scalar_node])
	output_node = GradientInputNode([norm2_node])

	assert np.array_equal(scalar_node.get_output(), 3*value)
	norm2_node.get_output()
	assert np.array_equal(scalar_node.get_gradient(0), 6*value)

def test_selection_node():
	value = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])	
	input_node = InputNode(value)
	selection_node = SelectionNode([input_node], 1, 3)
	norm2_node = Norm2Node([selection_node])
	output_node = GradientInputNode([norm2_node])

	assert np.array_equal(selection_node.get_output(), np.array([[2, 3], [6, 7]]))
	norm2_node.get_output()
	print(selection_node.get_gradient(0))
	assert np.array_equal(selection_node.get_gradient(0), np.array([[0, 4, 6, 0], [0, 12, 14, 0]]))

def init_ones(shape):
	return np.ones(shape)

def test_learnable_node():	
	learnable_node = LearnableNode(init_ones((3, 2)))
	norm2_node = Norm2Node([learnable_node])
	output_node = GradientInputNode([norm2_node])

	assert np.array_equal(learnable_node.get_output(), np.ones((3, 2)))
	norm2_node.get_output()
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
	node_fun = Norm2Node([node_add])
	node_out = GradientInputNode([node_fun])

	y = np.array([[2, 3], [5, 6], [8, 9]])
	assert np.array_equal(node_add.get_output(), y)
	node_fun.get_output()
	print(node_add.get_gradient(0))
	assert np.array_equal(node_add.get_gradient(0), 2*y)
	assert np.array_equal(node_add.get_gradient(1), 2*y)

def test_substraction_node():
	node_in1 = InputNode(np.array([[1, 1], [2, 2], [3, 3]]))
	node_in2 = InputNode(np.array([[1, 2], [3, 4], [5, 6]]))
	node_sub = SubstractionNode(node_in1, node_in2)
	node_fun = Norm2Node([node_sub])
	node_out = GradientInputNode([node_fun])

	y = np.array([[0, -1], [-1, -2], [-2, -3]])
	assert np.array_equal(node_sub.get_output(), y)
	node_fun.get_output()
	assert np.array_equal(node_sub.get_gradient(0), 2*y)
	assert np.array_equal(node_sub.get_gradient(1), -2*y)

def test_multiplication_node():
	node_in1 = InputNode(np.array([[1, 1], [2, 2]]))
	node_in2 = InputNode(np.array([[1, 2], [3, 4]]))
	node_dot = MultiplicationNode(node_in1, node_in2)
	node_fun = Norm2Node([node_dot])
	node_out = GradientInputNode([node_fun])

	assert np.array_equal(node_dot.get_output(), np.array([[4, 6], [8, 12]]))
	node_fun.get_output()
	assert np.array_equal(node_dot.get_gradient(0), np.array([[32, 72], [64, 144]]))
	assert np.array_equal(node_dot.get_gradient(1), np.array([[40, 60], [40, 60]]))

def test_ew_multiplication_node():
	node_in1 = InputNode(np.array([[1, 1], [2, 2]]))
	node_in2 = InputNode(np.array([[1, 2], [3, 4]]))
	node_dot = EWMultiplicationNode(node_in1, node_in2)
	node_fun = Norm2Node([node_dot])
	node_out = GradientInputNode([node_fun])

	assert np.array_equal(node_dot.get_output(), np.array([[1, 2], [6, 8]]))
	node_fun.get_output()
	assert np.array_equal(node_dot.get_gradient(0), np.array([[2, 8], [36, 64]]))
	assert np.array_equal(node_dot.get_gradient(1), np.array([[2, 4], [24, 32]]))

def test_concatenate_node():
    node_in1 = InputNode(np.array([[1, 1], [2, 2]]))
    node_in2 = InputNode(np.array([[1, 2], [3, 4]]))
    node_conca =  ConcatenationNode(node_in1, node_in2)
    node_fun = Norm2Node([node_conca])
    node_out = GradientInputNode([node_fun])

    assert np.array_equal(node_conca.get_output(), np.array([[1, 1, 1, 2], [2, 2, 3, 4]]))
    node_fun.get_output()
    assert np.array_equal(node_conca.get_gradient(0), np.array([[2, 2], [4, 4]]))
    assert np.array_equal(node_conca.get_gradient(1), np.array([[2, 4], [6, 8]]))
 
def test_softmax_cross_entropy_node():
	node_in1 = InputNode(np.array([[1, 1], [2, 2]]))
	node_in2 = InputNode(np.array([[1, 2], [3, 4]]))
	node_sce = SoftmaxCrossEntropyNode(node_in1, node_in2)
	node_out = GradientInputNode([node_sce])

	assert node_sce.get_output() == - (0 + np.log(2) + 2*np.log(3) + 2*np.log(4))
	node_sce.get_output()
	assert np.array_equal(node_sce.get_gradient(0), np.array([[0, -np.log(2)], [-np.log(3), -np.log(4)]]))
	assert np.array_equal(node_sce.get_gradient(1), -np.array([[1, 1/2], [2/3, 2/4]]))

def test_sigmoid_cross_entropy_node():
	node_in1 = InputNode(np.array([[1, 0], [0.5, 0.7]]))
	node_in2 = InputNode(np.array([[0.9, 0.3], [0.4, 0.8]]))
	node_sce = SigmoidCrossEntropyNode(node_in1, node_in2)
	node_out = GradientInputNode([node_sce])

	y = -np.log(0.9) - np.log(0.7) - (0.5*np.log(0.4)+0.5*np.log(0.6)) - (0.7*np.log(0.8)+0.3*np.log(0.2))
	assert np.array_equal(node_sce.get_output(), y)
	dJdin1 = np.array([[-np.log(0.9)+np.log(0.1), -np.log(0.3)+np.log(0.7)], [-np.log(0.4)+np.log(0.6), -np.log(0.8)+np.log(0.2)]])
	assert np.allclose(node_sce.get_gradient(0), dJdin1)
	dJdin2 = np.array([[-1/0.9, 1/0.7], [-0.5/0.4+0.5/0.6, -0.7/0.8+0.3/0.2]])
	assert np.allclose(node_sce.get_gradient(1), dJdin2)

def test_sum_node():
	node_in1 = InputNode(np.array([[1, 0], [0.5, 0.7]]))
	node_in2 = InputNode(np.array([[0.9, 0.3], [0.4, 0.8]]))
	node_in3 = InputNode(np.array([[2, 3], [5, 6]]))
	node_sum = SumNode([node_in1, node_in2, node_in3])
	node_norm = Norm2Node([node_sum])
	node_out = GradientInputNode([node_norm])
	assert np.array_equal(node_sum.get_output(), np.array([[3.9, 3.3], [5.9, 7.5]]))

	node_in1 = InputNode(2)
	node_in2 = InputNode(3)
	node_in3 = InputNode(1)
	node_sum = SumNode([node_in1, node_in2, node_in3])
	node_out = GradientInputNode([node_sum])
	assert node_sum.get_output() == 6
	assert node_sum.get_gradient(0) == 1