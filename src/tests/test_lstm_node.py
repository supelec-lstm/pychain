import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

import numpy as np
import pytest
from lstm_node import *
from recurrent_graph import *
from layer import *

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

@pytest.fixture
def layer1():
    x = InputNode(1)
    h_in = InputNode(2)
    s_in = InputNode(3)
    # LSTM
    dim_x = 1
    dim_s = 2
    lstm = LSTMNode(dim_x, dim_s, [x, h_in, s_in])
    lstm.learnable_nodes[0].w = np.ones((dim_x + dim_s, dim_s))
    lstm.learnable_nodes[1].w = np.ones((dim_x + dim_s, dim_s))
    lstm.learnable_nodes[2].w = np.ones((dim_x + dim_s, dim_s))
    # Outputs
    h_out = IdentityNode((lstm, 0))
    s_out = IdentityNode((lstm, 1))
    # Cost
    y = InputNode(4)
    e = SubstractionNode(y, h_out)
    cost = Norm2Node(e)

    nodes = [x, h_in, s_in, lstm, h_out, s_out, y, e, cost]
    return Layer(nodes, [x], [h_out], [h_in, s_in], [h_out, s_out], [y], cost, [lstm])


def test_propagation1(node1):
    node = node1

    previous_h = np.array([[3, 4]])
    previous_s = np.array([[1, 2]])
    node_x = InputNode(np.array([[5]]))
    node_h = InputNode(previous_h)
    node_s = InputNode(previous_s)
    node.set_parents([node_x, node_h, node_s])
    output = node.get_output()

    g = np.array([sigmoid(12)] * 2)
    assert np.allclose(node.g.get_output(), g)
    i = np.array([np.tanh(12)] * 2)
    assert np.allclose(node.i.get_output(), i)
    s = previous_s + g*i
    assert np.allclose(node.s_out.get_output(), s)
    o = np.array([sigmoid(12)] * 2)
    assert np.allclose(node.o.get_output(), o)
    h = o*np.tanh(s)
    assert np.allclose(node.h_out.get_output(), h)

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

    output = norm2_node.get_output()
    node.get_gradient()

    dh = dh + 2 * (node.h_out.get_output() - y)
    ds = ds + dh*node.o.get_output()*(1-node.l.get_output()**2)
    assert np.allclose(node.s_in.get_gradient(), ds)

    wg = node.learnable_nodes[0].w
    wi = node.learnable_nodes[1].w
    wo = node.learnable_nodes[2].w
    g = node.g.get_output()
    i = node.i.get_output()
    o = node.o.get_output()
    l = node.l.get_output()
    dh = np.dot((ds*i*(1-g)*g), wg.T) + \
        np.dot((ds*g*(1-i**2)), wi.T) + \
        np.dot((dh*l*(1-o)*o), wo.T)
    assert np.allclose(node.h_in.get_gradient(), dh[:,1:])

def test_propagate_layer1(layer1):
    layer = layer1
    graph = RecurrentGraph(layer, 1, [(1, 2), (1, 2)])

    previous_h = np.array([[3, 4]])
    previous_s = np.array([[1, 2]])
    x = [np.array([[5]])]
    outputs = graph.propagate(x, [previous_h, previous_s])

    g = np.array([sigmoid(12)] * 2)
    i = np.array([np.tanh(12)] * 2)
    s = previous_s + g*i
    o = np.array([sigmoid(12)] * 2)
    h = o*np.tanh(s)
    print(outputs)
    print(h, s)
    lstm = graph.layers[0].learnable_nodes[0]
    print(lstm.output_nodes[0].get_output())
    print(lstm.output_nodes[1].get_output())
    print(graph.layers[0].hidden_output_nodes[0].get_output())
    print(graph.layers[0].hidden_output_nodes[1].get_output())
    assert np.allclose(outputs[0][0], h)

def test_backpropagate_layer1(layer1):
    layer = layer1
    graph = RecurrentGraph(layer, 1, [(1, 2), (1, 2)])

    previous_h = np.array([[3, 4]])
    previous_s = np.array([[1, 2]])
    x = [np.array([[5]])]
    outputs = graph.propagate(x, [previous_h, previous_s])

    dh = np.array([[4, 3]])
    ds = np.array([[2, 1]])
    y = [np.array([[0, 6]])]
    graph.backpropagate(y, [dh, ds])

    g = np.array([sigmoid(12)] * 2)
    i = np.array([np.tanh(12)] * 2)
    s = previous_s + g*i
    l = np.tanh(s)
    o = np.array([sigmoid(12)] * 2)
    h = o*l
    dh = dh + 2 * (h - y[0])
    ds = ds + dh*o*(1-l**2)
    lstm = graph.layers[0].learnable_nodes[0]
    wg = lstm.learnable_nodes[0].w
    wi = lstm.learnable_nodes[1].w
    wo = lstm.learnable_nodes[2].w
    dh = np.dot((ds*i*(1-g)*g), wg.T) + \
        np.dot((ds*g*(1-i**2)), wi.T) + \
        np.dot((dh*l*(1-o)*o), wo.T)

    assert np.allclose(graph.layers[0].hidden_input_nodes[0].get_gradient(0), dh[:,1:])
    assert np.allclose(graph.layers[0].hidden_input_nodes[1].get_gradient(0), ds)