
class Graph:
    """Create a graph from a set of nodes"""
    def __init__(self,nodes):
        
        self.nodes = nodes
        input_node = 
        output_node = 
        expected_output = 
        cost_node =
        learnable_nodes = 



    def propagate(self,x):
        """Propagate the inputs in the entire graph and return the output."""

        input_node.set_value(x)
        cost_node.evaluate()
        return output_node.evaluate() 

    def backpropagate(self,x,y):
        pass

    def descend_gradient(self,learning_rate,batch_size):
        pass

    def batch_descent_gradient(self,learning_rate):
        pass

    def stochastic_gradient_descent(self,learning_rate):
        pass

    def reset_memoization(self):
        """Reset the memoization variables in all the nodes"""

        for node in nodes:
            node.reset_memoization()

    def reset_accumulators(self):
        """Reset the accumulator variables in all the learnable nodes"""

        for learnable_node in learnable_nodes:
            learnable_node.reset_accumulator()
            