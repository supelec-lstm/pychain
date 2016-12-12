from node import *
from graph import *

class RecurrentGraph:
	def __init__(self, graph, hidden_shapes=None):
		self.graph = graph
		# Give a key to each node
		for i, node in enumerate(self.graph.nodes):
			node.key = i
		# the weights should de shared
		self.weights = []
		for node in self.graph.learnable_nodes:
			self.weights.append(node.w)
		# hidden nodes
		self.hidden_shapes = hidden_shapes or []
		self.nb_hidden_nodes = len(self.hidden_shapes)

	def unfold(self, k):
		unfolded_graph, matchings = self.create_nodes(k)
		self.create_links(k, matchings)
		return unfolded_graph

	def propagate(self, sequence):
		graph = self.unfold(1)
		hidden_state = [np.zeros(shape) for shape in self.hidden_shapes]
		outputs = []
		for x in sequence:
			output = graph.propagate([x] + hidden_state)
			outputs.append(output[:-self.nb_hidden_nodes])
			hidden_state = output[-self.nb_hidden_nodes:]
		return outputs

	def backpropagate(self, sequence, expected_sequence, learning_rate):
		graph = self.unfold(len(sequence))
		hidden_state = [np.zeros(shape) for shape in self.hidden_shapes]
		graph.propagate(sequence + hidden_state)
		cost = graph.backpropagate(expected_sequence)
		graph.descend_gradient(learning_rate, 1)
		self.weights = [node.w for node in graph.learnable_nodes]
		return cost

	def batch_backpropagate(self, sequence, expected_sequence, learning_rate, batch_length):
		graph = self.unfold(batch_length)
		hidden_state = [np.zeros(shape) for shape in self.hidden_shapes]
		costs = []
		for i in range(0, len(sequence), batch_length):
			output = graph.propagate(sequence[i:i+batch_length] + hidden_state)
			hidden_state = output[-self.nb_hidden_nodes:]
			costs.append(graph.backpropagate(expected_sequence[i:i+batch_length]))
			graph.descend_gradient(learning_rate, 1)
		self.weights = [node.w for node in graph.learnable_nodes]
		return costs

	def create_nodes(self, k):
		# Create containers for nodes
		nodes = []
		input_nodes = []
		output_nodes = []
		expected_output_nodes = []
		cost_nodes = []
		learnable_nodes = []
		# Remember matchings
		matchings = [{} for _ in range(k)]
		# Fill the lists
		# Copy the learnables nodes
		for node, weight in zip(self.graph.learnable_nodes, self.weights):
			new_node = LearnableNode(weight)
			learnable_nodes.append(new_node)
			nodes.append(new_node)
			for i in range(k):
				matchings[i][node.key] = new_node
		# Replicate the other nodes k times
		for i in range(k):
			for node in self.graph.nodes:
				if node in self.graph.learnable_nodes or type(node) == DelayOnceNode:
					continue
				new_node = type(node)()
				if node in self.graph.input_nodes:
					input_nodes.append(new_node)
				if node in self.graph.output_nodes:
					output_nodes.append(new_node)
				if node in self.graph.expected_output_nodes:
					expected_output_nodes.append(new_node)
				if node is self.graph.cost_node:
					cost_nodes.append(new_node)
				nodes.append(new_node)
				matchings[i][node.key] = new_node
		# Create the cost node
		cost_node = SumNode(cost_nodes)
		nodes.append(cost_node)
		# Create hidden nodes
		for node in self.graph.nodes:
			if type(node) == DelayOnceNode:
				hidden_node = InputNode()
				nodes.append(hidden_node)
				input_nodes.append(hidden_node)
				output_nodes.append(matchings[-1][node.parents[0].key])
				# Change the links
				matchings[0][node.key] = hidden_node
				for i in range(1, k):
					matchings[i][node.key] = matchings[i-1][node.parents[0].key]
		# Create a graph
		graph = Graph(nodes, input_nodes, output_nodes, expected_output_nodes, cost_node, learnable_nodes)
		# Return the graph and the matchings
		return graph, matchings

	def create_links(self, k, matchings):
		# Replicate the links
		for i in range(k):
			for node in self.graph.nodes:
				if node in self.graph.learnable_nodes or type(node) == DelayOnceNode:
					continue
				replicated_node = matchings[i][node.key]
				parents = []
				for other_node in node.parents:
					parents.append(matchings[i][other_node.key])
				replicated_node.set_parents(parents)