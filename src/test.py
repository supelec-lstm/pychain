from node import *
from graph import *
from functionNode import *


input_node = InputNode(np.array([[0],[1]]))

w1 = LearnableNode((2,2))

h1 = MultiplicationNode([w1, input_node])

s1 = SigmoidNode([h1])

expected_output = InputNode()

d1 = SubstractionNode([s1, expected_output])

e1 = Norm2Node([d1])

c1 = ConstantGradientNode([e1])

graph = Graph([input_node, w1, h1, s1, expected_output, d1, e1, c1], input_node, s1, expected_output, c1, [w1])

for i in range(1000):

    graph.batch_descent_gradient(0.1,[np.array([[0],[0]]),np.array([[0],[1]]),np.array([[1],[0]]),np.array([[1],[1]])],[np.array([0]),np.array([1]),np.array([1]),np.array([1])])


