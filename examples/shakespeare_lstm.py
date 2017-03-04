import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import pickle
import time
import matplotlib.pyplot as plt
from lstm_node import *
from layer import *
from recurrent_graph import *

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',\
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',\
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ',\
            ',', '.', '?', ';', ':', "'", '"', '[', ']',\
             '-', '(', ')', '&', '!']

letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

num_lstms = 2
dim_s = 128
hidden_shapes = [(1, dim_s), (1, dim_s)] * num_lstms
learning_rate = 0.1

def string_to_sequence(string):
    sequence = np.zeros((len(string), 1, len(letters)))
    for i, letter in enumerate(string):
        if not letter in letters:
            return []
        sequence[i, 0, letter_to_index[letter]] = 1
    return sequence

def sequence_to_string(sequence):
    return ''.join([letters[np.argmax(x[0])] for x in sequence])

def sample_sequence_to_string(sequence):
    return ''.join([letters[np.random.choice(len(letters), p=x[0].flatten())] for x in sequence])

def learn_shakespeare(layer, path, N):
    f = open(path)

    for i, line in zip(range(N), f):
        if i % 100 == 0:
            print(i)
        if i % 1000 == 0:
            sample(layer, 20)
        string = f.readline().strip().upper()
        sequence = string_to_sequence(string)
        if len(sequence) > 1:
            X = sequence[:-1]
            Y = sequence[1:]
            graph = RecurrentGraph(layer, len(sequence)-1, hidden_shapes)
            graph.propagate(sequence[:-1])
            graph.backpropagate(sequence[1:])
            graph.descend_gradient(learning_rate)

    f.close()

def sample(layer, n):
    for i in range(ord('A'), ord('Z')+1):
        s = chr(i).upper()
        x = string_to_sequence(s)[0]
        graph = RecurrentGraph(layer, n, hidden_shapes)
        result = graph.propagate_self_feeding(x)
        print(chr(i) + sample_sequence_to_string(result))
        print(chr(i) + sequence_to_string(result))

def create_layer():
    # Input
    x = InputNode()
    hidden_inputs = []
    hidden_outputs = []
    lstms = []
    # LSTMs
    parent = x
    for i in range(num_lstms):
        h_in = InputNode()
        s_in = InputNode()
        dim_x = dim_s
        if i == 0:
            dim_x = len(letters)
        lstm = LSTMNode(dim_x, dim_s, [parent, h_in, s_in])
        h_out = IdentityNode((lstm, 0))
        s_out = IdentityNode((lstm, 1))
        parent = h_out
        # Add to containers
        hidden_inputs += [h_in, s_in]
        hidden_outputs += [h_out, s_out]
        lstms.append(lstm)
    # Softmax
    w = LearnableNode(np.random.randn(dim_s, len(letters)))
    mult = MultiplicationNode(parent, w)
    out = SoftmaxNode(mult)
    # Cost
    y = InputNode()
    e = SubstractionNode(y, out)
    cost = Norm2Node(e)

    nodes = hidden_inputs + hidden_outputs + lstms + [x, w, mult, out, y, e, cost]
    return Layer(nodes, [x], [out], hidden_inputs, hidden_outputs, [y], cost, [w] + lstms)

if __name__ == '__main__':
    layer = create_layer()
    learn_shakespeare(layer, 'examples/shakespeare/shakespeare_karpathy.txt', 10000)
    pickle.dump(layer, open('shakespeare.pickle', 'wb'))