from src.node import *

class Graph:
	def __init__(self, nodes, input_node, output_node, expected_output_node, cost_node, learnable_nodes,recurrent_node=None,concatenate_node=None,node_grappin_depart=None,node_grappin_arrivee=None):
		self.nodes = nodes
		self.input_node = input_node
		self.output_node = output_node
		self.expected_output_node = expected_output_node
		self.cost_node = cost_node
		self.learnable_nodes = learnable_nodes
		self.recurrent_nodes = recurrent_node
		self.concatenate_node = concatenate_node
		self.node_grappin_depart = node_grappin_depart
		self.node_grappin_arrivee = node_grappin_arrivee

		# Create a constant gradient neuron
		#ConstantGradientNode([cost_node])

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

class RecurrentNetwork:

	def __init__(self,graph):
		self.graph = graph
		self.graph_unfolded = graph

	def replicate_graph(self):
		new_nodes = []
		new_input_node = []
		new_output_node = []
		new_expected_output_node = []
		new_cost_node = []
		new_node_grappin_arrivee = None
		new_node_grappin_depart = None
		nodes_interessants = [node for node in self.graph.nodes if (node not in self.graph.recurrent_nodes and node not in self.graph.learnable_nodes)]
		for i in range(len(nodes_interessants)):
			new_nodes.append(type(nodes_interessants[i])())
			if nodes_interessants[i] is self.graph.node_grappin_arrivee:
				new_node_grappin_arrivee = new_nodes[i]
			if nodes_interessants[i] is self.graph.node_grappin_depart:
				new_node_grappin_depart = new_nodes[i]
			if nodes_interessants[i] in self.graph.input_node:
				new_input_node.append(new_nodes[i])
			if nodes_interessants[i] in self.graph.output_node:
				new_output_node.append(new_nodes[i])
			if nodes_interessants[i] in self.graph.expected_output_node:
				new_expected_output_node.append(new_nodes[i])
			if nodes_interessants[i] in self.graph.cost_node:
				new_cost_node.append(new_nodes[i])
		for i in range(len(nodes_interessants)):
			for j in range(len(nodes_interessants)):
				if nodes_interessants[j] in nodes_interessants[i].parents:
					new_nodes[i].parents.append(new_nodes[j])
					new_nodes[j].add_child(new_nodes[i],len(new_nodes[i].parents))
					#new_nodes[j].set_parents([new_nodes[i]])
		for i in range(len(nodes_interessants)):
			for learnable in self.graph.learnable_nodes:
				if learnable in nodes_interessants[i].parents:
					new_nodes[i].parents.append(learnable)
					learnable.add_child(new_nodes[i],len(new_nodes[i].parents))
					#new_nodes[i].set_parents([learnable])


		return [new_nodes,new_input_node,new_output_node,new_expected_output_node,new_cost_node,new_node_grappin_depart,new_node_grappin_arrivee]


	def unfold(self,k):
		for j in range(k):
			[new_nodes, new_input_node, new_output_node, new_expected_output_node, new_cost_node,new_node_grappin_depart, new_node_grappin_arrivee] = self.replicate_graph()
			self.graph_unfolded.node_grappin_arrivee.parents.append(new_node_grappin_depart)
			new_node_grappin_depart.add_child(self.graph_unfolded.node_grappin_arrivee,len(self.graph_unfolded.node_grappin_arrivee.parents))
			self.graph_unfolded = Graph(self.graph_unfolded.nodes+new_nodes,self.graph_unfolded.input_node+new_input_node,self.graph_unfolded.output_node+new_output_node,self.graph_unfolded.expected_output_node+new_expected_output_node,self.graph_unfolded.cost_node+new_cost_node,self.graph_unfolded.learnable_nodes,self.graph_unfolded.recurrent_nodes,self.graph_unfolded.concatenate_node,self.graph_unfolded.node_grappin_depart,new_node_grappin_arrivee)
		for node in self.graph_unfolded.recurrent_nodes:
			for child in node.children:
				child[0].parents = [parent for parent in child[0].parents if parent is not node]
		for node in self.graph_unfolded.recurrent_nodes:
			node.children = []
		for node in self.graph_unfolded.recurrent_nodes:
			self.graph_unfolded.node_grappin_arrivee.parents.append(node)
			node.add_child(self.graph_unfolded.node_grappin_arrivee,len(self.graph_unfolded.node_grappin_arrivee.parents))

def init_function(shape):
    return (np.random.rand(*shape) * 0.2 - 0.1)

input = InputNode()
learnable = LearnableNode((2,1),init_function)
recurrent = DelayOnceNode()
concatenate = ConcatenateNode(input,recurrent)
multiplication = MultiplicationNode(learnable,concatenate)
sigmoid = SigmoidNode([multiplication])
expected = InputNode()
cost = Norm2Node([sigmoid,expected])
recurrent.parents.append(sigmoid)
sigmoid.add_child(recurrent,0)
nodes = [input, learnable,concatenate,multiplication,sigmoid,recurrent,expected,cost]
graphe = Graph(nodes,[input],[sigmoid],[expected],[cost],[learnable],[recurrent],[concatenate],sigmoid,concatenate)
graphe_recurrent = RecurrentNetwork(graphe)
graphe_recurrent.unfold(2)
for node in graphe_recurrent.graph_unfolded.nodes:
	print(node, node.parents)
	print(node,node.children)


