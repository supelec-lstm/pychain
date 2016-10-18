from node import *

class Graph:
	def __init__(self, nodes, input_node, output_node, expected_output_node, cost_node, learnable_nodes):
		self.nodes = nodes
		self.input_node = input_node
		self.output_node = output_node
		self.expected_output_node = expected_output_node
		self.cost_node = cost_node
		self.learnable_nodes = learnable_nodes

		# Create a constant gradient neuron
		ConstantGradientNode([cost_node])

	def propagate(self, x):
		self.reset_memoization()
		self.input_node.set_value(x)
		return self.output_node.evaluate()

	def backpropagate(self, y):
		self.expected_output_node.set_value(y)
		cost = self.cost_node.evaluate()
		for node in self.learnable_nodes:
			node.get_gradient(0)
		return cost

	def descend_gradient(self, learning_rate, batch_size):
		for node in self.learnable_nodes:
			node.descend_gradient(learning_rate, batch_size)
		self.reset_accumulators()

	def batch_gradient_descent(self, x, y, learning_rate):
		self.propagate(x)
		cost = self.backpropagate(y)
		self.descend_gradient(learning_rate, x.shape[0])
		return cost

	def reset_memoization(self):
		for node in self.nodes:
			node.reset_memoization()

	def reset_accumulators(self):
		for node in self.learnable_nodes:
			node.reset_accumulator()