class Graph:
    """Create a graph from a set of nodes"""
    
    def __init__(self, nodes, input_node, output_node, expected_output, cost_node, learnable_nodes):
        
        self.nodes = nodes
        self.input_node = input_node
        self.output_node = output_node
        self.expected_output = expected_output
        self.cost_node = cost_node
        self.learnable_nodes = learnable_nodes



    def propagate(self, x):
        """Propagate the inputs in the entire graph and return the output."""
        
        self.reset_memoization()
        self.input_node.set_value(x)
        return self.output_node.evaluate()

    def backpropagate(self, x, y):
        """Backpropagate the gradient through the graph,
            Return the cost."""

        self.propagate()
        self.expected_output.set_value(y)                       #what if we put a matrix as input? will we take care of the gradient for the whole matrix or should we split the matrix and do it by line?
        self.cost_node.evaluate()
        for learnable_node in self.learnable_nodes:
            learnable_node.get_gradient()

        return cost_node.evaluate()

    def descend_gradient(self, learning_rate, batch_size):
        """Descend the gradient in all learnable nodes."""

        for learnable_node in self.learnable_nodes:
            learnable_node.descend_gradient(learning_rate, batch_size)

    def batch_descent_gradient(self, learning_rate,):
        """batch gradient descent"""

        costs = []
        for x, y in zip(X,Y):
            costs.append(self.backpropagate(x,y))
        self.descend_gradient(learning_rate, batch_size = len(X))
        return costs

    def stochastic_gradient_descent(self, learning_rate, X, Y):
        """Stochastic gradient descent"""

        costs = []
        for x, y in zip(X,Y):
            costs.append(self.backpropagate(x,y))
            self.descend_gradient(learning_rate, batch_size = 1)
        return costs

    def reset_memoization(self):
        """Reset the memoization variables in all the nodes."""

        for node in self.nodes:
            node.reset_memoization()

    def reset_accumulators(self):
        """Reset the accumulator variables in all the learnable nodes."""

        for learnable_node in self.learnable_nodes:
            learnable_node.reset_accumulator()