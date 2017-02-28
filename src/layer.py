from node import *

class Layer:
	def __init__(self, nodes, input_nodes, output_nodes, hidden_input_nodes, hidden_output_nodes,
		expected_output_nodes, cost_node, learnable_nodes):
		self.nodes = nodes
		self.input_nodes = input_nodes
		self.output_nodes = output_nodes
		self.hidden_input_nodes = hidden_input_nodes
		self.hidden_output_nodes = hidden_output_nodes
		self.expected_output_nodes = expected_output_nodes
		self.cost_node = cost_node
		self.learnable_nodes = learnable_nodes

		# Create GradientInputNodes to backpropagate gradient
		self.gradient_input_nodes = [GradientInputNode([node]) for node in self.hidden_output_nodes]
		# Create a GradientInputNode to backpropagate the local cost
		GradientInputNode([self.cost_node])

		# Give a key to each node
		for i, node in enumerate(self.nodes):
			node.key = i

	def evaluate(self, X, H_in):
		# Set the input nodes values
		for node, x in zip(self.input_nodes, X):
			node.set_value(x)
		# Set the hidden input nodes values
		for node, h_in in zip(self.hidden_input_nodes, H_in):
			node.set_value(h_in)
		# Propagate
		return [node.evaluate() for node in self.output_nodes], \
			[node.evaluate() for node in self.hidden_output_nodes]

	def get_gradient(self, Y, dJdH_out):
		# Set the expected output nodes values
		for node, y, in zip(self.expected_output_nodes, Y):
			node.set_value(y)
		# Compute the cost
		cost = self.cost_node.evaluate()
		# Set the gradient nodes values
		for node, dJdh_out in zip(self.gradient_input_nodes, dJdH_out):
			node.set_value(dJdh_out)
		# Compute the gradient with respect to weights
		for node in self.learnable_nodes:
			node.get_gradient(0)
		# Backpropagate
		# Return dJdH_in and the cost
		return [node.get_gradient(0) for node in self.input_nodes], cost

	def descend_gradient(self, learning_rate):
		for node in self.learnable_nodes:
			node.descend_gradient(learning_rate, 1)

	def reset_memoization(self):
		for node in self.learnable_nodes:
			node.reset_memoization()

	def clone(self):
		# Create containers for nodes
		# We clone the nodes but not the weights
		nodes = []
		input_nodes = []
		output_nodes = []
		hidden_input_nodes = []
		hidden_output_nodes = []
		expected_output_nodes = []
		cost_node = None
		learnable_nodes = []
		# Duplicate nodes
		keyToNode = {}
		for node in self.nodes:
			new_node = node.clone()
			# Append the nodes to the right container
			if node in self.input_nodes:
				input_nodes.append(new_node)
			if node in self.output_nodes:
				output_nodes.append(new_node)
			if node in self.hidden_input_nodes:
				hidden_input_nodes.append(new_node)
			if node in self.hidden_output_nodes:
				hidden_output_nodes.append(new_node)
			if node in self.expected_output_nodes:
				expected_output_nodes.append(new_node)
			if node == self.cost_node:
				cost_node = new_node
			if node in self.learnable_nodes:
				learnable_nodes.append(new_node)
			nodes.append(new_node)
			keyToNode[node.key] = new_node
		# Create the links between nodes
		for (node, new_node) in zip(self.nodes, nodes):
			new_node.set_parents([keyToNode[parent.key] for parent in node.parents])
		# Return a new layer
		return Layer(nodes, input_nodes, output_nodes, hidden_input_nodes, hidden_output_nodes, \
			expected_output_nodes, cost_node, learnable_nodes)