from node import *

class CompositeNode(LearnableNode):
    def __init__(self, nodes, input_nodes, output_nodes, learnable_nodes, parents=None):
        Node.__init__(self, parents)
        self.nodes = nodes
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.learnable_nodes = learnable_nodes

        # Create GradientInputNode to backpropagate gradient
        self.gradient_input_nodes = [GradientInputNode([node]) for node in self.output_nodes]

        # Give a key to each node
        for i, node in enumerate(self.nodes):
            node.key = i

    def compute_output(self):
        # Set the input nodes values
        for node, x in zip(self.input_nodes, self.x):
            node.set_value(x)
        # Propagate
        return [node.evaluate() for node in self.output_nodes]

    def compute_gradient(self):
        # Set the gradient nodes values
        for node, dJdy in zip(self.gradient_input_nodes, self.dJdy):
            node.set_value(dJdy)
        # Compute the gradient with respect to weights
        for node in self.learnable_nodes:
            node.get_gradient(0)
        # Backpropagate
        return [node.get_gradient() for node in self.input_nodes]

    def descend_gradient(self, learning_rate, batch_size):
        for node in self.learnable_nodes:
            node.descend_gradient(learning_rate, batch_size)

    def reset_memoization(self):
        for node in self.learnable_nodes:
            node.reset_memoization()

    def reset_accumulator(self):
        for node in self.learnable_nodes:
            node.reset_accumulator()

    def clone(self):
        # Duplicate nodes
        nodes = []
        keyToNode = {}
        for node in self.nodes:
            new_node = node.clone()
            nodes.append(new_node)
            keyToNode[node.key] = new_node
        # Append the nodes to the right container
        input_nodes = [keyToNode[node.key] for node in self.input_nodes]
        output_nodes = [keyToNode[node.key] for node in self.output_nodes]
        learnable_nodes = [keyToNode[node.key] for node in self.learnable_nodes]
        # Create the links between nodes
        for (node, new_node) in zip(self.nodes, nodes):
            new_node.set_parents([(keyToNode[parent.key], i_output) for parent, i_output in node.parents])
        # Return a new composite node
        return CompositeNode(nodes, input_nodes, output_nodes, learnable_nodes)


    