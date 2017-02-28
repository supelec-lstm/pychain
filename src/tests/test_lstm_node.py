import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

import numpy as np
import pytest
from lstm_node import *

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

@pytest.fixture
def node1():
    dim_x = 1
    dim_s = 2
    node = LSTMNode(dim_x, dim_s)
    node.learnable_nodes[0].w = np.ones((dim_x + dim_s, dim_s))
    node.learnable_nodes[1].w = np.ones((dim_x + dim_s, dim_s))
    node.learnable_nodes[2].w = np.ones((dim_x + dim_s, dim_s))
    return node

def test_propagation1(node1):
    node = node1

    previous_h = np.array([[3, 4]])
    previous_s = np.array([[1, 2]])
    node_x = InputNode(np.array([[5]]))
    node_h = InputNode(previous_h)
    node_s = InputNode(previous_s)
    node.set_parents([node_x, node_h, node_s])
    output = node.evaluate()

    g = np.array([sigmoid(12)] * 2)
    assert np.allclose(node.g.evaluate(), g)
    i = np.array([np.tanh(12)] * 2)
    assert np.allclose(node.i.evaluate(), i)
    s = previous_s + g*i
    assert np.allclose(node.s_out.evaluate(), s)
    o = np.array([sigmoid(12)] * 2)
    assert np.allclose(node.o.evaluate(), o)
    h = o*np.tanh(s)
    assert np.allclose(node.h_out.evaluate(), h)

def test_backpropagation1(node1):
    node = node1

    previous_h = np.array([[3, 4]])
    previous_s = np.array([[1, 2]])
    node_x = InputNode(np.array([[5]]))
    node_h = InputNode(previous_h)
    node_s = InputNode(previous_s)
    node.set_parents([node_x, node_h, node_s])

    dh = np.array([[4, 3]])
    ds = np.array([[2, 1]])
    y = np.array([[0, 6]])
    node_dh = GradientInputNode([(node, 0)], dh)
    node_ds = GradientInputNode([(node, 1)], ds)
    node_y = InputNode(y)
    sub_node = SubstractionNode((node, 0), node_y)
    norm2_node = Norm2Node(sub_node)
    gradient_node = GradientInputNode([norm2_node])

    output = norm2_node.evaluate()
    node.get_gradient()

    dh = dh + 2 * (node.h_out.evaluate() - y)
    ds = ds + dh*node.o.evaluate()*(1-node.l.evaluate()**2)
    assert np.allclose(node.s_in.get_gradient(), ds)

    wg = node.learnable_nodes[0].w
    wi = node.learnable_nodes[1].w
    wo = node.learnable_nodes[2].w
    g = node.g.evaluate()
    i = node.i.evaluate()
    o = node.o.evaluate()
    l = node.l.evaluate()
    dh = np.dot((ds*i*(1-g)*g), wg.T) + \
        np.dot((ds*g*(1-i**2)), wi.T) + \
        np.dot((dh*l*(1-o)*o), wo.T)
    assert np.allclose(node.h_in.get_gradient(), dh[:,1:])