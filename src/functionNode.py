from node import *

class FunctionNode(Node):
    """A node which applies an activation function"""

    def __init__(self, parent):
        """Parent is a Node"""

        self.parent = parent

    def evaluate(self):
        """Evaluate the output of the neuron with the f function implemented in the sub-classes"""

        if not self.y:
            self.y = self.f(self.parents[0].evaluate())
        return self.y

    def f(self,x):
        """Apply the activation function, defined for each sub-classes"""

        raise NotImplementedError()

    def gradient_f(self,x):
        """Return gradient of the activation function, also implemented in the sub-classes"""

        raise NotImplementedError()


class SigmoidNode(FunctionNode):
    """The sigmoid function node"""

    def f(self, x):
        return 1 / (1 + np.exp(x))


    def gradient_f(self, x):
        return self.f(x) * (1 - self.f(x))


class TanhNode(FunctionNode):
    """The thanh function node"""

    def f(self, x):
        return np.tanh(x)

    def gradient_f(self, x):
        return 1 / (1 + x ** 2)


class ReluNode(FunctionNode):
    """The rectified linear unit node"""

    def f(self, x):
        return x if x > 0 else 0

    def gradient_f(self, x):
        return 1 if x > 0 else 0


class SoftMaxNode(FunctionNode):
    """The softmax function node"""

    def f(self, x):
        list = []
        total = np.sum(np.exp(x))
        for i in range(len(x)):
            np.append(list,(np.exp(x[i]) / total)
        return list

    def gradient_f(self, x):
        jacob = np.zeros((len(x),len(x)))
        for i in range(len(x)):
            for j in range(i,len(x)):
                jacob[j][i], jacob[i][j] = (-a*b for a,b in zip(self.f(x[i]),self.f(x[j])))
        return jacob


class Norm2Node(FunctionNode):

    """The norm two function node"""

    def f(self, x):
        return np.linalg.norm(x)

    def gradient_f(self, x):
        return 2*x

class ScalarMultiplicationNode(FunctionNode):

    """The scalar multiplication node, needed for balancing some weights"""

    def __init__(self,parent,scalar):

        FunctionNode.__init__(self,parent)
        self.scalar = scalar


    def f(self,x):
        return self.scalar*x

    def gradient_f(self,x):
        return self.scalar