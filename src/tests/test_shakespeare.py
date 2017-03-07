import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

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
dim_s = 64
hidden_shapes = [(1, dim_s), (1, dim_s)] * num_lstms
learning_rate = 2e-3
len_seq = 10

def string_to_sequence(string):
    sequence = np.zeros((len(string), 1, len(letters)))
    for i, letter in enumerate(string):
        if not letter in letters:
            sequence[i, 0, letter_to_index[' ']] = 1
        else:
            sequence[i, 0, letter_to_index[letter]] = 1
    return sequence

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
        lstm.learnable_nodes[0].w = np.ones((dim_x + dim_s, dim_s))
        lstm.learnable_nodes[1].w = np.ones((dim_x + dim_s, dim_s))
        lstm.learnable_nodes[2].w = np.ones((dim_x + dim_s, dim_s))
        h_out = IdentityNode((lstm, 0))
        s_out = IdentityNode((lstm, 1))
        parent = h_out
        # Add to containers
        hidden_inputs += [h_in, s_in]
        hidden_outputs += [h_out, s_out]
        lstms.append(lstm)
    # Softmax
    w = LearnableNode(np.ones((dim_s, len(letters))))
    mult = MultiplicationNode(parent, w)
    out = SoftmaxNode(mult)
    # Cost
    y = InputNode()
    e = SubstractionNode(y, out)
    cost = Norm2Node(e)

    nodes = hidden_inputs + hidden_outputs + lstms + [x, w, mult, out, y, e, cost]
    return Layer(nodes, [x], [out], hidden_inputs, hidden_outputs, [y], cost, [w] + lstms)

def test_keep_same_network():
    layer1 = create_layer()
    layer2 = create_layer()

    # For layer1 we create an unique graph
    graph1 = RecurrentGraph(layer1, len_seq - 1, hidden_shapes)

    # Number of iterations
    N = 10

    # Read file
    f = open('examples/shakespeare/shakespeare_karpathy.txt')
    text = f.read().upper()
    f.close()

    # Test the outputs of the graphs after each iteration
    for i in range(0, min(len(text), N * len_seq), len_seq):
        string = text[i:i+50]
        sequence = string_to_sequence(string)

        # Graph1
        outputs1 = graph1.propagate(sequence[:-1])
        graph1.backpropagate(sequence[1:])
        graph1.descend_gradient(learning_rate)

        # For layer2, we create a new graph at each iteration
        graph2 = RecurrentGraph(layer2, len_seq - 1, hidden_shapes)
        outputs2 = graph2.propagate(sequence[:-1])
        graph2.backpropagate(sequence[1:])
        graph2.descend_gradient(learning_rate)

        # Test
        print(outputs1[0])
        print(outputs2[0])
        assert np.allclose(outputs1, outputs2)