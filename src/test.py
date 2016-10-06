from node import *
from graph import *
from functionNode import *
from random import *

input_nod = InputNode()

w1 = LearnableNode((4,1))

h1 = MultiplicationNode([w1, input_nod])

s1 = SigmoidNode([h1])

expected_output = InputNode()

d1 = SubstractionNode([s1, expected_output])

e1 = Norm2Node([d1])

c1 = ConstantGradientNode([e1])

graph = Graph([input_nod, w1, h1, s1, expected_output, d1, e1, c1], input_nod, s1, expected_output, c1, [w1])

print(graph.propagate(np.array([1,1,1,1])))

batch = np.array([[[0,0,1,1],[0]], [[0,1,1,1], [1]], [[1,0,1,1], [1]], [[1,1,1,1], [0]]])

X=[]
Y=[]
for i in range(10000):
	n = randint(0,3)
	X.append(batch[n][0])
	Y.append(batch[n][1])

print(graph.batch_descent_gradient(0.3,X,Y))

