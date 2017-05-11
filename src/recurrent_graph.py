from node import *
from graph import *

class RecurrentGraph:
	def __init__(self, layer, length, hidden_shapes):
		self.hidden_shapes = hidden_shapes

		# Unfold the layer
		self.layers = [layer.clone() for _ in range(length)]

	def propagate(self, sequence, H=None):
		# Reset memoization
		self.reset_memoization()
		# Init H
		if H is None:
			H = [np.zeros(shape) for shape in self.hidden_shapes]
		# Propagate
		outputs = []
		for x, layer in zip(sequence, self.layers):
			output, H = layer.get_output([x], H)
			outputs.append(output)
		return outputs

	def generate(self, fselect, x, H=None):
		# Reset memoization
		self.reset_memoization()
		# Init H
		if H is None:
			H = [np.zeros(shape) for shape in self.hidden_shapes]
		# Propagate
		outputs = []
		cur_input = x
		i=0
		for layer in self.layers:
			output, H = layer.get_output([cur_input], H)
			cur_input = fselect(output[0])
			if i==500:
				print(cur_input)
			i += 1
			outputs.append(cur_input)
		return outputs

	def generateFromSequence(self, fselect, sequence, H=None):
		# Reset memoization
		self.reset_memoization()
		# Init H
		if H is None:
			H = [np.zeros(shape) for shape in self.hidden_shapes]
		# Propagate
		outputs = []
		cur_input = sequence[0]
		index = 0
		for layer in self.layers:
			if index < len(sequence)-1:
				output, H = layer.get_output([cur_input], H)
				index += 1
				cur_input = sequence[index]
			else:
				output, H = layer.get_output([cur_input], H)
				cur_input = fselect(output[0])
				outputs.append(cur_input)
		return outputs

	def backpropagate(self, expected_sequence, dJdH=None):
		# Init dJdH
		if dJdH is None:
			dJdH = [np.zeros(shape) for shape in self.hidden_shapes]
		# Backpropagate
		total_cost = 0
		for y, layer in zip(reversed(expected_sequence), reversed(self.layers)):
			dJdH, cost = layer.get_gradient([y], dJdH)
			total_cost += cost
		return total_cost

	def reset_memoization(self):
		for layer in self.layers:
			layer.reset_memoization()

	def get_learnable_nodes(self):
		return self.layers[0].get_learnable_nodes()