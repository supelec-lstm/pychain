import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import pickle
import time
from datetime import datetime
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
learning_rate = 2e-3
len_seq = 50
nb_seq_per_batch = 50
hidden_shapes = [(nb_seq_per_batch, dim_s), (nb_seq_per_batch, dim_s)] * num_lstms

def string_to_sequences(string):
    sequences = np.zeros((len_seq, nb_seq_per_batch, len(letters)))
    for i_seq in range(nb_seq_per_batch):
        for i, letter in enumerate(string[i_seq*len_seq:(i_seq+1)*len_seq]):
            if not letter in letters:
                sequences[i, i_seq, letter_to_index[' ']] = 1
            else:
                sequences[i, i_seq, letter_to_index[letter]] = 1
    return sequences

def sequence_to_string(sequence):
    return ''.join([letters[np.argmax(x[0])] for x in sequence])

def sample_sequence_to_string(sequence):
    return ''.join([letters[np.random.choice(len(letters), p=x[0].flatten())] for x in sequence])

def learn_shakespeare(layer, path, N):
    # Create the graph
    graph = RecurrentGraph(layer, len_seq - 1, hidden_shapes)
    # Learn
    i_pass = 1
    i_batch = 1
    while True:
        # Read file
        f = open(path)
        text = f.read().upper()
        f.close()
        len_batch = len_seq*nb_seq_per_batch
        nb_batches = int(len(text) / len_batch)
        for i in range(nb_batches):
            # Take a new batch
            string = text[i*len_batch:(i+1)*len_batch]
            sequences = string_to_sequences(string)
            # Propagate and backpropagate the batch
            graph.propagate(sequences[:-1])
            cost = graph.backpropagate(sequences[1:]) / len_seq / nb_seq_per_batch
            graph.descend_gradient(learning_rate, nb_seq_per_batch)
            # Save and sample
            if i != 0 and i_batch % 1000 == 0:
                sample(graph)
                save_layer(layer, i_batch)
            # Print info
            print('pass: ' + str(i_pass) + ', batch: ' + str(i+1) + '/' + str(nb_batches) + \
                ', cost: ' + str(cost))
            i_batch += 1
        i_pass += 1

def sample(graph):
    for i in range(ord('A'), ord('Z')+1):
        s = chr(i).upper()
        x = string_to_sequence(s)[0]
        result = graph.propagate_self_feeding(x)
        print(chr(i) + sample_sequence_to_string(result))
        print(chr(i) + sequence_to_string(result))
    #print(layer.learnable_nodes[0].w)

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
    w = LearnableNode(0.1 * np.random.randn(dim_s, len(letters)))
    mult = MultiplicationNode(parent, w)
    out = SoftmaxNode(mult)
    # Cost
    y = InputNode()
    e = SubstractionNode(y, out)
    cost = Norm2Node(e)

    nodes = hidden_inputs + hidden_outputs + lstms + [x, w, mult, out, y, e, cost]
    return Layer(nodes, [x], [out], hidden_inputs, hidden_outputs, [y], cost, [w] + lstms)

def test(graph):
    string = 'FIRST CITIZEN:\nBEFORE WE PROCEED ANY FURTHER, HEAR'
    sequence = string_to_sequence(string)
    result = graph.propagate(sequence[:-1])
    for letter, y in zip(string[:5], result[:5]):
        print(letter, [(l, p) for l, p in zip(letters, y[0].flatten())])

def save_layer(layer, i_batch):
    path = 'models/' + str(datetime.now()) + '_b:' +  str(i_batch) + '.pickle'
    pickle.dump(layer, open(path, 'wb'))

if __name__ == '__main__':
    layer = create_layer()
    save_layer(layer, 0)
    learn_shakespeare(layer, 'examples/shakespeare/shakespeare_karpathy.txt', 40000)