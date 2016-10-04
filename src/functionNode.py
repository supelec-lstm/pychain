from node import *

class FunctionNode(Node):
    """A node which apply an activation function"""

    def __init__(self, parent):
        """parent is a Node"""

        self.parent = parent

        def evaluate(self):
            if not self.y:
                self.y = self.f(self.parents[0].evaluate())
            return self.y

    def f(self,x):
        """apply the activation function
        the function is defined in the sub-classes"""

        raise NotImplementedError()

    def gradient_f(self,x):
        raise NotImplementedError()


class SigmoidNode(FunctionNode):

    def f(self, x):
        return 1 / (1 + np.exp(x))


    def gradient_f(self, x):
        return self.f(x) * (1 - self.f(x))


class TanhNode(FunctionNode):

    def f(self, x):
        return np.tanh(x)

    def gradient_f(self, x):
        return 1 / (1 + x ** 2)


class ReluNode(FunctionNode):

    def f(self, x):
        return x if x > 0 else 0

    def gradient_f(self, x):
        return 1 if x > 0 else 0


class SoftMaxNode(FunctionNode):

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

    def f(self, x):
        return np.linalg.norm(x)

    def gradient_f(self, x):
        return 2*x

class ScalarMultiplicationNode(FunctionNode):

    def __init__(self,parent,scalar):

        FunctionNode.__init__(self,parent)
        self.scalar = scalar


    def f(self,x):
        return self.scalar*x

    def gradient_f(self,x):
        return self.scalar