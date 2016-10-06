from node import *
from graph import *
from functionNode import *
from random import *
import matplotlib.pyplot as plt 

input_nod = InputNode()

w1 = LearnableNode((3,2))

<<<<<<< Updated upstream
w2 = LearnableNode((2,1))
=======
w2 = LearnableNode((1,1))
>>>>>>> Stashed changes

h1 = MultiplicationNode([w1, input_nod])

s1 = SigmoidNode([h1])

h2 = MultiplicationNode([w2,s1])

s2 = SigmoidNode([h2])

expected_output = InputNode()

d1 = SubstractionNode([s2, expected_output])

e1 = Norm2Node([d1])

c1 = ConstantGradientNode([e1])

graph = Graph([input_nod, w1, h1, s1, h2, s2, expected_output, d1, e1, c1], input_nod, s2, expected_output, c1, [w1, w2])

print(graph.propagate(np.array([1,1,1])))
print(graph.learnable_nodes[0])

batch = np.array([[[0,0,1],[0]], [[0,1,1], [1]], [[1,0,1], [1]], [[1,1,1], [0]]])

X=[]
Y=[]
for i in range(100000):
	n = randint(0,3)
	X.append(batch[n][0])
	Y.append(batch[n][1])

costs = graph.batch_descent_gradient(0.01,X,Y)

print(graph.propagate(batch[0][0]))
print(graph.propagate(batch[1][0]))
print(graph.propagate(batch[2][0]))
print(graph.propagate(batch[3][0]))

x = np.linspace(-0.5, 1.5, num = 100)
plane = np.zeros((100,100))

for i in range(100):
	for j in range(100):
		plane[i][j] = graph.propagate([x[i],x[99-j],1])

print(costs[0])
plt.imshow(plane, origin = 'lower')

plt.show()

plt.plot(costs[0:100])

plt.show()
