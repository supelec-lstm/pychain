from node import *
from graph import *
from functionNode import *

input_node = InputNode()

w1 = LearnableNode((4,0))

h1 = MultiplicationNode(input1, w1)

s1 = SigmoidNode([h1])

expected_output = InputNode

d1 = SubstractionNode(s1, expected_output)

e1 = Norm2Node([d1])

c1 = ConstantGradientNode([e1])

graph = Graph([input_node, w1, h1, s1, expected_output, d1, e1, c1], input_node, s1, expected_output, c1, [w1])

