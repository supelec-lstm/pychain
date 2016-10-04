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
        liste = []
        total = np.sum(np.exp(x))
        for i in range(len(x)):
            liste[i] = np.exp(x[i]) / total
        return total

    def gradient_f(self, x):
        pass


class Norm2Node(FunctionNode):
    def f(self, x):
        return np.linalg.norm(x)

    def f(self, x):
        return np