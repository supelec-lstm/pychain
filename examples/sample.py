import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import pickle
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
len_seq = 1000
hidden_shapes = [(1, dim_s), (1, dim_s)] * num_lstms

def letter_to_vector(letter):
    vector = np.zeros((1, len(letters)))
    vector[0,letter_to_index[letter]] = 1
    return vector

def argmax(output):
    chosen_letter = np.zeros((1, len(letters)))
    chosen_letter[0,np.argmax(output)] = 1
    return chosen_letter

def stochastic(output):
    chosen_letter = np.zeros((1, len(letters)))
    chosen_letter[0,np.random.choice(len(letters), p=output.flatten())] = 1
    return chosen_letter

def sequence_to_string(sequence):
    return ''.join([index_to_letter[np.argmax(y)] for y in sequence])

def sample(graph):
    for letter in letters[0]:
        x = letter_to_vector(letter)
        result = graph.generate(stochastic, x)
        print(letter + sequence_to_string(result))

if __name__ == '__main__':
    layer = pickle.load(open('models/2017-03-19 14:11:32.218324_b:9000.pickle', 'rb'))
    graph = RecurrentGraph(layer, len_seq - 1, hidden_shapes)
    sample(graph)