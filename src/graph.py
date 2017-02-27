from node import *

class Graph:
	def __init__(self, nodes, input_nodes, output_nodes, expected_output_nodes, cost_node, learnable_nodes):
		self.nodes = nodes
		self.input_nodes = input_nodes
		self.output_nodes = output_nodes
		self.expected_output_nodes = expected_output_nodes
		self.cost_node = cost_node
		self.learnable_nodes = learnable_nodes

		# Create a constant gradient neuron
		OutputNode([cost_node])

	def propagate(self, X):
		self.reset_memoization()
		for x, node in zip(X, self.input_nodes):
			node.set_value(x)
		return [node.evaluate() for node in self.output_nodes]

	def backpropagate(self, Y):
		for y, node in zip(Y, self.expected_output_nodes):
			node.set_value(y)
		cost = self.cost_node.evaluate()
		for node in self.learnable_nodes:
			node.get_gradient(0)
		return cost

	def descend_gradient(self, learning_rate, batch_size):
		for node in self.learnable_nodes:
			node.descend_gradient(learning_rate, batch_size)
		self.reset_accumulators()

	def batch_gradient_descent(self, X, Y, learning_rate):
		self.propagate(X)
		cost = self.backpropagate(Y)
		self.descend_gradient(learning_rate, X[0].shape[0])
		return cost

	def reset_memoization(self):
		for node in self.nodes:
			node.reset_memoization()

	def reset_accumulators(self):
		for node in self.learnable_nodes:
			node.reset_accumulator()

