import pickle
import matplotlib.pyplot as plt
import reber
import symmetrical_reber
from recurrent_graph import *

letters = []
"""for i in range(ord('a'), ord('z') + 1):
	letters.append(chr(i))"""
for i in range(ord('A'), ord('Z') + 1):
	letters.append(chr(i))
letters += [' ', ',', '.', '?', ';', ':', "'", '-', '!', '\n']
letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

train_path = 'examples/shakespeare/shakespeare_karpathy.txt'

learning_rate = 0.1

def init_function(shape):
    return (np.random.rand(*shape) - 1)

def string_to_sequence(string):
	sequence = []
	for letter in string.upper():
		if letter in letters:
			sequence.append(np.array([[1 if letter_to_index[letter] == j else 0 for j in range(len(letters))]]))
	return sequence

def train_shakespeare(graph, N):
	f = open(train_path)
	for i in range(N):
		if i % 1000 == 0:
			print(i)
			for i in range(ord('A'), ord('Z')+1):
				print(sample(graph, chr(i), 30))
		string = f.readline()
		if(len(string) > 1):
			sequence = string_to_sequence(string)
			cost = graph.backpropagate(sequence[0:-1], sequence[1:], learning_rate)
	f.close()

def train_shakespeare_batch(graph, N, batch_size):
	f = open(train_path)
	for i in range(N // 1000):
		string = ''
		for _ in range(1000):
			string += f.readline()
		sequence = string_to_sequence(string)
		graph.batch_backpropagate(sequence[0:-1], sequence[1:], learning_rate, batch_size)
		print(i * 1000)
		for i in range(ord('A'), ord('Z')+1):
			print(sample(graph, chr(i), 30))
	f.close()

def sample(graph, first_letter, length):
	string = first_letter
	for _ in range(length):
		sequence = string_to_sequence(string)
		scores = graph.propagate(sequence)
		string += letters[np.argmax(scores[-1])]
		#exp_scores = np.exp(scores[-1])
		#softmax = exp_scores / np.sum(exp_scores)
		#string += letters[np.random.choice(range(len(letters)), p=softmax.ravel())]
	return string

def create_graph_fully_connected(nb_hidden_units):
	input_node = InputNode()
	delay_once_node = DelayOnceNode()
	concatenation_node = ConcatenationNode(input_node, delay_once_node)
	learnable_node = LearnableNode(init_function((len(letters) + nb_hidden_units + len(letters), nb_hidden_units + len(letters))))
	multiplication_node = MultiplicationNode(concatenation_node, learnable_node)
	sigmoid_node = SigmoidNode(multiplication_node)
	selection_node = SelectionNode(sigmoid_node, 0, len(letters))
	expected_output = InputNode()
	substraction_node = SubstractionNode(expected_output, selection_node)
	cost_node = Norm2Node(substraction_node)
	# set the recurrence
	delay_once_node.set_parents([sigmoid_node])

	nodes = [input_node, delay_once_node, concatenation_node, learnable_node, multiplication_node, \
		     sigmoid_node, selection_node, expected_output, substraction_node, cost_node]
	graph = Graph(nodes, [input_node], [selection_node], [expected_output], cost_node, [learnable_node])

	return RecurrentGraph(graph, [(1, nb_hidden_units + len(letters))])

def create_graph():
	input_node = InputNode()
	delay_once_node = DelayOnceNode()
	concatenation_node = ConcatenationNode(input_node, delay_once_node)
	learnable_node1 = LearnableNode(init_function((len(letters) + len(letters), 36)))
	multiplication_node1 = MultiplicationNode(concatenation_node, learnable_node1)
	hidden_node = TanhNode(multiplication_node1)

	learnable_node2 = LearnableNode(init_function((36, len(letters))))
	multiplication_node2 = MultiplicationNode(hidden_node, learnable_node2)
	sigmoid_node = SigmoidNode(multiplication_node2)
	expected_output = InputNode()
	substraction_node = SubstractionNode(expected_output, sigmoid_node)
	cost_node = Norm2Node(substraction_node)
	# set the recurrence
	delay_once_node.set_parents([sigmoid_node])

	nodes = [input_node, delay_once_node, concatenation_node, learnable_node1, multiplication_node1, \
		     hidden_node, learnable_node2, multiplication_node2, sigmoid_node, expected_output, \
		     substraction_node, cost_node]
	graph = Graph(nodes, [input_node], [sigmoid_node], [expected_output], cost_node, [learnable_node1, learnable_node2])

	return RecurrentGraph(graph, [(1, len(letters))])

if __name__ == '__main__':
	graph = create_graph()
	train_shakespeare_batch(graph, 40000, 10)
	# Save the graph
	pickle.dump(graph, open('shakespeare.pickle', 'wb'))
	graph = pickle.load(open('shakespeare.pickle', 'rb'))
	for i in range(ord('A'), ord('Z')+1):
		print(sample(graph, chr(i), 30))