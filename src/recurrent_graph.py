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

	def propagate_self_feeding(self, x, H=None):
		# Reset memoization
		self.reset_memoization()
		# Init H
		if H is None:
			H = [np.zeros(shape) for shape in self.hidden_shapes]
		# Propagate
		outputs = []
		cur_input = x
		for layer in self.layers:
			output, H = layer.get_output([cur_input], H)
			outputs.append(output)
			cur_input = output[0]
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

	def descend_gradient(self, learning_rate):
		for layer in self.layers:
			layer.descend_gradient(learning_rate)
		# Reset accumulators
		self.reset_accumulators()

	def reset_memoization(self):
		for layer in self.layers:
			layer.reset_memoization()

	def reset_accumulators(self):
		for layer in self.layers:
			layer.reset_accumulators()