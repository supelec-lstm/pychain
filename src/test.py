from node import *
from graph import *
from functionNode import *
from random import *
import matplotlib.pyplot as plt 

input_nod = InputNode()

w1 = LearnableNode((2,2))

w2 = LearnableNode((2,1))


h1 = MultiplicationNode([input_nod,w1 ])

s1 = SigmoidNode([h1])

h2 = MultiplicationNode([s1,w2])

s2 = SigmoidNode([h2])

expected_output = InputNode()

d1 = SubstractionNode([s2, expected_output])

e1 = Norm2Node([d1])

c1 = ConstantGradientNode([e1])

graph = Graph([input_nod, w1, w2, h1, s1, h2, s2, expected_output, d1, e1, c1], input_nod, s2, expected_output, c1, [w1, w2])

batch = [[[0,0],[0]],[[0,1],[1]],[[1,0],[1]],[[1,1],[0]]]

X=[]
Y=[]
for i in range(10000):
	n = randint(0,3)
	X.append(np.array(batch[n][0]))
	Y.append(np.array(batch[n][1]))
costs=[]
for j in range(100):
    costs.append(graph.batch_descent_gradient(0.8,X,Y))


x = np.linspace(-0.5, 1.5, num = 100)
plane = np.zeros((100,100))

for i in range(100):
	for j in range(100):
		plane[i][j] = graph.propagate([x[i],x[99-j],1])


plt.imshow(plane, origin = 'lower')

plt.show()
cost2=[]
for j in range(10000):
    cost=0
    for i in range(100):
        cost+=costs[j][i]
    cost2.append(cost/100)
plt.plot(cost2)

plt.show()
