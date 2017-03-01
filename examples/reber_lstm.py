import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../../reber-grammar/')))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src/')))

import pickle
import time
import matplotlib.pyplot as plt
import reber
import symmetrical_reber
from lstm_node import *
from layer import *
from recurrent_graph import *

letters = ['B', 'T', 'P', 'S', 'X', 'V', 'E']
letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

learning_rate = 0.1

train_path = '../reber-datasets/reber_train_2.4M.txt'
test_path = '../reber-datasets/reber_test_1M.txt'
automaton = reber.create_automaton()

"""train_path = '../reber-datasets/symmetrical_reber_train_2.4M.txt'
test_path = '../reber-datasets/symmetrical_reber_test_1M.txt'
automaton = symmetrical_reber.create_automaton(0.5)"""

def string_to_sequence(string):
    sequence = np.zeros((len(string), 1, len(letters)))
    for i, letter in enumerate(string):
        sequence[i, 0, letter_to_index[letter]] = 1
    return sequence

def train_reber(layer, N):
    start_time = time.time()
    f = open(train_path)
    t = []
    accuracies = []
    for i in range(N):
        if i % 1000 == 0:
            t.append(i)
            print(i)
            test_reber(layer)
            accuracies.append(accuracy(layer, 1000))
            print(accuracies[-1])
        string = f.readline().strip()
        sequence = string_to_sequence(string)
        graph = RecurrentGraph(layer, len(sequence)-1, [(1, 7), (1, 7)])
        graph.propagate(sequence[:-1])
        graph.backpropagate(sequence[1:])
        graph.descend_gradient(learning_rate)
    f.close()
    """plt.plot(t, accuracies)
    plt.xlabel('Nombre de chaines')
    plt.ylabel('Précision')
    plt.title("Courbe d'apprentissage avec un état caché à {} poids ({:.2f}s)".format(weights.dim_s, time.time() - start_time))
    plt.show()"""

def predict_correctly(layer, string, threshold):
    sequence = string_to_sequence(string)
    graph = RecurrentGraph(layer, len(sequence)-1, [(1, 7), (1, 7)])
    result = graph.propagate(sequence[:-1])
    cur_state = automaton.start
    for i, (x, y) in enumerate(zip(sequence[:-1], result)):
        cur_state = cur_state.next(string[i])
        predicted_transitions = {letters[j] for j, activated in enumerate(y[0].flatten() > threshold) if activated}
        if set(predicted_transitions) != set(cur_state.transitions.keys()):
            return False
    return True

def accuracy(layer, N):
    f = open(test_path)
    c = 0
    for i in range(N):
        if i % 1000 == 0:
            print(i)
        string = f.readline().strip()
        if predict_correctly(layer, string, 0.3):
            c += 1
    return c / N

def test_reber(layer):
    string = 'BTSSXXVV'
    sequence = string_to_sequence(string)
    graph = RecurrentGraph(layer, len(sequence)-1, [(1, 7), (1, 7)])
    result = graph.propagate(sequence[:-1])
    for letter, y in zip(string, result):
        print(letter, [(l, p) for l, p in zip(letters, y[0].flatten())])

def create_layer():
    x = InputNode(1)
    h_in = InputNode(2)
    s_in = InputNode(3)
    # LSTM
    lstm = LSTMNode(len(letters), len(letters), [x, h_in, s_in])
    # Outputs
    h_out = IdentityNode((lstm, 0))
    s_out = IdentityNode((lstm, 1))
    # Cost
    y = InputNode(4)
    e = SubstractionNode(y, h_out)
    cost = Norm2Node(e)

    nodes = [x, h_in, s_in, lstm, h_out, s_out, y, e, cost]
    return Layer(nodes, [x], [h_out], [h_in, s_in], [h_out, s_out], [y], cost, [lstm])


if __name__ == '__main__':
    layer = create_layer()
    train_reber(layer, 20000)
    print(accuracy(layer, 100000))
    #test_reber(layer)