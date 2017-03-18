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

    def get_output(self, X, H_in):
        # Set the input nodes values
        for node, x in zip(self.input_nodes, X):
            node.set_value(x)
        # Set the hidden input nodes values
        for node, h_in in zip(self.hidden_input_nodes, H_in):
            node.set_value(h_in)
        # Propagate
        return [node.get_output() for node in self.output_nodes], \
            [node.get_output() for node in self.hidden_output_nodes]

    def get_gradient(self, Y, dJdH_out):
        # Set the expected output nodes values
        for node, y, in zip(self.expected_output_nodes, Y):
            node.set_value(y)
        # Compute the cost
        cost = self.cost_node.get_output()
        # Set the gradient nodes values
        for node, dJdh_out in zip(self.gradient_input_nodes, dJdH_out):
            node.set_value(dJdh_out)
        # Compute the gradient with respect to weights
        for node in self.learnable_nodes:
            node.get_gradient(0)
        # Backpropagate
        # Return dJdH_in and the cost
        return [node.get_gradient(0) for node in self.hidden_input_nodes], cost

    def reset_memoization(self):
        for node in self.nodes:
            node.reset_memoization()

    def get_learnable_nodes(self):
        # Retrieve the learnable nodes contained inside composite nodes
        nodes = []
        for node in self.learnable_nodes:
            nodes += node.get_learnable_nodes()
        return nodes

    def clone(self):
        # Duplicate nodes
        nodes = []
        keyToNode = {}
        for node in self.nodes:
            new_node = node.clone()
            nodes.append(new_node)
            keyToNode[node.key] = new_node
        # Append the nodes to the right containers
        input_nodes = [keyToNode[node.key] for node in self.input_nodes]
        output_nodes = [keyToNode[node.key] for node in self.output_nodes]
        hidden_input_nodes = [keyToNode[node.key] for node in self.hidden_input_nodes]
        hidden_output_nodes = [keyToNode[node.key] for node in self.hidden_output_nodes]
        expected_output_nodes = [keyToNode[node.key] for node in self.expected_output_nodes]
        cost_node = keyToNode[self.cost_node.key]
        learnable_nodes = [keyToNode[node.key] for node in self.learnable_nodes]
        # Create the links between nodes
        for (node, new_node) in zip(self.nodes, nodes):
            new_node.set_parents([(keyToNode[parent.key], i_output) for parent, i_output in node.parents])
        # Return a new layer
        return Layer(nodes, input_nodes, output_nodes, hidden_input_nodes, hidden_output_nodes, \
            expected_output_nodes, cost_node, learnable_nodes)