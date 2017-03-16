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

# Read file
path = 'examples/shakespeare/shakespeare_karpathy.txt'
f = open(path)
text = f.read()
f.close()

# Create vocab
letters = sorted(list(set(text)))
print('Vocab size:' + str(len(letters)))

letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

num_lstms = 2
dim_s = 128
learning_rate = 2e-3
len_seq = 50
nb_seq_per_batch = 50
hidden_shapes = [(nb_seq_per_batch, dim_s), (nb_seq_per_batch, dim_s)] * num_lstms

def string_to_sequences(string, nb_seq=1, len_seq=None):
    len_seq = len_seq or int(len(string) / nb_seq)
    sequences = np.zeros((len_seq, nb_seq, len(letters)))
    for i_seq in range(nb_seq):
        for i, letter in enumerate(string[i_seq*len_seq:(i_seq+1)*len_seq]):
            sequences[i, i_seq, letter_to_index[letter]] = 1
    return sequences

def learn_shakespeare(layer):
    # Create the graph
    graph = RecurrentGraph(layer, len_seq - 1, hidden_shapes)
    # Learn
    i_pass = 1
    i_batch = 1
    while True:
        len_batch = len_seq*nb_seq_per_batch
        nb_batches = int(len(text) / len_batch)
        for i in range(nb_batches):
            t_start = time.time()
            # Take a new batch
            string = text[i*len_batch:(i+1)*len_batch]
            sequences = string_to_sequences(string, nb_seq_per_batch)
            # Propagate and backpropagate the batch
            graph.propagate(sequences[:-1])
            cost = graph.backpropagate(sequences[1:]) / len_seq / nb_seq_per_batch
            # Get gradient and params norm
            lstm_node = layer.learnable_nodes[1]
            grad_norm = 0
            param_norm = 0
            for w in lstm_node.learnable_nodes:
                grad_norm += ((w.acc_dJdw/nb_seq_per_batch)**2).sum()
                param_norm += (w.w**2).sum()
            # Desend gradient
            graph.descend_gradient(learning_rate, nb_seq_per_batch)
            # Print info
            print('pass: ' + str(i_pass) + ', batch: ' + str(i+1) + '/' + str(nb_batches) + \
                ', cost: ' + str(cost) + ', time: ' + str(time.time() - t_start)  + \
                ', grad/param norm: ' + str(np.sqrt(grad_norm/param_norm)))
            # Save
            if i_batch % 1000 == 0:
                save_layer(layer, i_batch)
            i_batch += 1
        i_pass += 1

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
        lstm = LSTMWFGNode(dim_x, dim_s, [parent, h_in, s_in])
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
    cost = SoftmaxCrossEntropyNode(y, out)
    #e = SubstractionNode(y, out)
    #cost = Norm2Node(e)

    #nodes = hidden_inputs + hidden_outputs + lstms + [x, w, mult, out, y, e, cost]
    nodes = hidden_inputs + hidden_outputs + lstms + [x, w, mult, out, y, cost]
    return Layer(nodes, [x], [out], hidden_inputs, hidden_outputs, [y], cost, [w] + lstms)

def save_layer(layer, i_batch):
    path = 'models/' + str(datetime.now()) + '_b:' +  str(i_batch) + '.pickle'
    pickle.dump(layer, open(path, 'wb'))

if __name__ == '__main__':
    layer = create_layer()
    learn_shakespeare(layer)