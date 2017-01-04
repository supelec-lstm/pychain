import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../../reber-grammar/')))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import pickle
import matplotlib.pyplot as plt
import reber
import symmetrical_reber
from recurrent_graph import *

letters = ['B', 'T', 'P', 'S', 'X', 'V', 'E']
letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

"""nb_hidden_units = 12
train_path = '../reber-datasets/symmetrical_reber_train_2.4M.txt'
test_path = '../reber-datasets/symmetrical_reber_test_1M.txt'
automaton = symmetrical_reber.create_automaton(0.5)"""

nb_hidden_units = 2
train_path = '../reber-datasets/reber_train_2.4M.txt'
test_path = '../reber-datasets/reber_test_1M.txt'
automaton = reber.create_automaton()

def init_function(shape):
    return (np.ones(shape))

def string_to_sequence(string):
	sequence = []
	for letter in string:
		sequence.append(np.array([[1 if letter_to_index[letter] == j else 0 for j in range(7)]]))
	return sequence

def train_reber(graph, N):
	f = open(train_path)
	learning_rate = 0.1
	for i in range(N):
		if i % 1000 == 0:
			print(i)
		string = f.readline().strip()
		print(string)
		sequence = string_to_sequence(string)
		cost = graph.backpropagate(sequence[0:-1], sequence[1:], learning_rate)
	f.close()

def predict_correctly(graph, string, threshold):
	sequence = string_to_sequence(string)
	predicted_sequence = graph.propagate(sequence[0:-1])
	cur_state = automaton.start
	for letter, scores in zip(string[:-1], predicted_sequence):
		cur_state = cur_state.next(letter)
		predicted_transitions = {letters[j] for j, activated in enumerate(scores.flatten() > threshold) if activated}
		if set(predicted_transitions) != set(cur_state.transitions.keys()):
			return False
	return True

def predict_last_letter(graph, string, threshold):
	sequence = string_to_sequence(string)
	predicted_sequence = graph.propagate(sequence[0:-1])
	cur_state = automaton.start
	predicted_transitions = {letters[j] for j, activated in enumerate(predicted_sequence[-2].flatten() > threshold) if activated}
	return predicted_transitions == {string[-2]}

def accuracy(graph, N):
	f = open(test_path)
	c = 0
	for i in range(N):
		if i % 1000 == 0:
			print(i)
		string = f.readline().strip()
		if predict_correctly(graph, string, 0.3):
			c += 1
	return c / N

def accuracy_last_letter(graph, N):
	f = open(test_path)
	c = 0
	for i in range(N):
		if i % 1000 == 0:
			print(i)
		string = f.readline().strip()
		if predict_last_letter(graph, string, 0.3):
			c += 1
	return c / N

def create_graph():
	input_node = InputNode()
	delay_once_node = DelayOnceNode()
	concatenation_node = ConcatenationNode(input_node, delay_once_node)
	learnable_node = LearnableNode(init_function((7 + nb_hidden_units + 7, nb_hidden_units + 7)))
	multiplication_node = MultiplicationNode(concatenation_node, learnable_node)
	sigmoid_node = SigmoidNode(multiplication_node)
	selection_node = SelectionNode(sigmoid_node, 0, 7)
	expected_output = InputNode()
	substraction_node = SubstractionNode(expected_output, selection_node)
	cost_node = Norm2Node(substraction_node)
	# set the recurrence
	delay_once_node.set_parents([sigmoid_node])

	nodes = [input_node, delay_once_node, concatenation_node, learnable_node, multiplication_node, \
		     sigmoid_node, selection_node, expected_output, substraction_node, cost_node]
	graph = Graph(nodes, [input_node], [selection_node], [expected_output], cost_node, [learnable_node])

	return RecurrentGraph(graph, [(1, nb_hidden_units + 7)])

if __name__ == '__main__':
	graph = create_graph()
	train_reber(graph, 1)
	print(graph.weights[0].shape)
	print(graph.weights[0])
	# Save the graph
	#pickle.dump(graph, open('reber.pickle', 'wb'))
	#graph = pickle.load(open('reber.pickle', 'rb'))
	#print(accuracy(graph, 10000))
	#print(accuracy_last_letter(graph, 10000))