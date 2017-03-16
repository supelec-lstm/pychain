import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../../../reber-grammar/')))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../')))

import pickle
import time
import matplotlib.pyplot as plt
import reber
import symmetrical_reber
from lstm_node import *
from layer import *
from recurrent_graph import *
from lstm import *

letters = ['B', 'T', 'P', 'S', 'X', 'V', 'E']
letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

learning_rate = 0.1

train_path = '../reber-datasets/reber_train_2.4M.txt'
test_path = '../reber-datasets/reber_test_1M.txt'
automaton = reber.create_automaton()

def string_to_sequence_column(string):
    sequence = np.zeros((len(string), len(letters), 1))
    for i, letter in enumerate(string):
        sequence[i, letter_to_index[letter]] = 1
    return sequence

def string_to_sequence_line(string):
    sequence = np.zeros((len(string), 1, len(letters)))
    for i, letter in enumerate(string):
        sequence[i, 0, letter_to_index[letter]] = 1
    return sequence

def create_weights():
    weights = Weights(len(letters), len(letters))
    weights.wg = np.ones((len(letters), 2 * len(letters)))
    weights.wi = np.ones((len(letters), 2 * len(letters)))
    weights.wo = np.ones((len(letters), 2 * len(letters)))
    return weights

def create_layer():
    x = InputNode(1)
    h_in = InputNode(2)
    s_in = InputNode(3)
    # LSTM
    lstm = LSTMNode(len(letters), len(letters), [x, h_in, s_in])
    lstm.learnable_nodes[0].w = np.ones((2 * len(letters), len(letters)))
    lstm.learnable_nodes[1].w = np.ones((2 * len(letters), len(letters)))
    lstm.learnable_nodes[2].w = np.ones((2 * len(letters), len(letters)))
    # Outputs
    h_out = IdentityNode([(lstm, 0)])
    s_out = IdentityNode([(lstm, 1)])
    # Cost
    y = InputNode(4)
    e = SubstractionNode([y, h_out])
    cost = Norm2Node([e])

    nodes = [x, h_in, s_in, lstm, h_out, s_out, y, e, cost]
    return Layer(nodes, [x], [h_out], [h_in, s_in], [h_out, s_out], [y], cost, [lstm])

def convert_output_to_columns(output):
    return [y[0].T for y in output]

def test_propagate():
    f = open(train_path)
    N = 100
    for _ in range(N):
        string = f.readline().strip()

        # LSTM Network
        weights = create_weights()
        sequence = string_to_sequence_column(string)
        network = LstmNetwork(weights, len(sequence)-1)
        output_network = network.propagate(sequence[:-1])

        # LSTM Graph
        layer = create_layer()
        sequence = string_to_sequence_line(string)
        graph = RecurrentGraph(layer, len(sequence)-1, [(1, 7), (1, 7)])
        output_graph = convert_output_to_columns(graph.propagate(sequence[:-1]))

        print(output_network)
        print(output_graph)
        assert np.allclose(output_network, output_graph)

def test_backpropagate():
    f = open(train_path)
    string = f.readline().strip()
    N = 100

    for _ in range(N):
        # LSTM Network
        weights = create_weights()
        sequence = string_to_sequence_column(string)
        network = LstmNetwork(weights, len(sequence)-1)
        network.propagate(sequence[:-1])
        network.backpropagate(sequence[1:])
        network.descend_gradient(learning_rate)

        # LSTM Graph
        layer = create_layer()
        sequence = string_to_sequence_line(string)
        graph = RecurrentGraph(layer, len(sequence)-1, [(1, 7), (1, 7)])
        graph.propagate(sequence[:-1])
        graph.backpropagate(sequence[1:])
        graph.descend_gradient(learning_rate)

        # Check costs
        for cell, layer in zip(reversed(network.cells), reversed(graph.layers)):
            assert np.allclose(cell.dJ, layer.cost_node.parents[0][0].get_gradient(1).T)

        # Checks dh and ds
        for i, (cell, layer) in enumerate(list(zip(reversed(network.cells), reversed(graph.layers)))):
            node = layer.learnable_nodes[0]
            assert np.allclose(cell.ds, node.s_in.get_gradient().T)
            assert np.allclose(cell.dh, node.h_in.get_gradient().T)

        # Check weights after gradient descent
        assert np.allclose(weights.wi, layer.learnable_nodes[0].wi.w.T)
        assert np.allclose(weights.wg, layer.learnable_nodes[0].wg.w.T)
        assert np.allclose(weights.wo, layer.learnable_nodes[0].wo.w.T)