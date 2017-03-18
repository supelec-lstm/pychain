import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import time
import numpy as np
import matplotlib.pyplot as plt
from graph import *
from node import *
from optimization_algorithm import *
from mnist import *

def init_function(shape):
    return (np.random.rand(*shape) * 0.2 - 0.1)

def shuffle_dataset(X, Y):
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]

def one_hot_encode(Y):
    return (np.dot(np.ones((Y.shape[0], 10)), np.diag(np.arange(10))) == Y).astype(float)

def normalized_dataset(X):
    mean_X = np.mean(X, axis=0)
    #stddev_X = np.std(X, axis=0)
    return (X - mean_X) / 255

def get_predicted_class(predicted_y):
    return np.argmax(predicted_y, axis=1)

def accuracy(graph, X, Y):
    predicted_y = graph.propagate([X])[0]
    predicted_class = get_predicted_class(predicted_y)
    return np.sum(Y.flatten() == predicted_class) / Y.shape[0]

def visualize(graph, X, Y, nb_samples=25):
    images = X[:nb_samples].reshape((nb_samples, 28, 28)) * 255
    labels = Y[:nb_samples]
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(5, 5, i+1)
        plt.imshow(image, cmap='Greys', vmin=0, vmax=255, interpolation='none')
        plt.title(str(get_predicted_class(graph.propagate(np.array([X[i]])))[0])+ ' ' + str(labels[i]))
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

def fully_connected(layers):
    input_node = InputNode()
    nodes = [input_node]
    learnable_nodes = []

    prev_size = 28*28+1
    cur_input_node = input_node
    for i, size in enumerate(layers):
        bias_node = AddBiasNode([cur_input_node])
        weights_node = LearnableNode(init_function((prev_size, size)))
        prod_node = MultiplicationNode([bias_node, weights_node])
        if i+1 < len(layers):
            cur_input_node = TanhNode([prod_node])
        else:
            cur_input_node = SoftmaxNode([prod_node])
        learnable_nodes += [weights_node]
        nodes += [bias_node, weights_node, prod_node, cur_input_node]
        prev_size = size+1
    
    expected_output_node = InputNode()
    #diff_node = SubstractionNode([expected_output_node, cur_input_node])
    #cost_node = Norm2Node([diff_node])
    cost_node = SoftmaxCrossEntropyNode([expected_output_node, cur_input_node])
    #cost_node = SigmoidCrossEntropyNode([expected_output_node, cur_input_node])

    nodes += [expected_output_node, cost_node]
    return Graph(nodes, [input_node], [cur_input_node], [expected_output_node], cost_node, learnable_nodes) 

 # Functions to benchmark: "one example at a time" performance

 # Need to be updated

"""def batch_gradient_descent_oe(graph, X, Y, i_start, i_end, learning_rate):
    total_cost = 0
    batch_size = i_end - i_start
    for x, y in zip(X[i_start:i_end], Y[i_start:i_end]):
        graph.propagate([np.array([x])])
        total_cost += graph.backpropagate([np.array([y])])
    graph.descend_gradient(learning_rate, batch_size)
    return total_cost / batch_size

def accuracy_oe(graph, X, Y):
    true_positive = 0
    for x, y in zip(X, Y):
        predicted_y = graph.propagate([np.array([x])])[0]
        predicted_class = get_predicted_class(predicted_y)
        true_positive += predicted_class == y
    return true_positive / Y.shape[0]

def benchmark_oe(graph, X, Y, X_test, Y_test):
    start_time = time.time()
    for i in range(0, X.shape[0], batch_size):
        print(i)
        print(batch_gradient_descent_oe(graph, X, ohe_Y, i, min(i+batch_size, 60000), learning_rate))
        if (i % 2048) == 0:
            print('ACCURACY TRAINING:', accuracy_oe(graph, X, Y))
            print('ACCURACY TEST:', accuracy_oe(graph, X_test, Y_test))
    print('ACCURACY TEST:', accuracy_oe(graph, X_test, Y_test))
    print('DURATION: ', time.time() - start_time)

def benchmark(graph, X, Y, X_test, Y_test):
    start_time = time.time()
    for i in range(0, X.shape[0], batch_size):
        print(i)
        print(graph.batch_gradient_descent([X[i:i+batch_size]], [ohe_Y[i:i+batch_size]], learning_rate) / batch_size)
        if (i % 2048) == 0:
            print('ACCURACY TRAINING:', accuracy(graph, X, Y))
            print('ACCURACY TEST:', accuracy(graph, X_test, Y_test))
    print('ACCURACY TEST:', accuracy_oe(graph, X_test, Y_test))
    print('DURATION: ', time.time() - start_time)

def propagate_oe(graph, X):
    for x in X:
        graph.propagate(np.array([x]))

def benchmark_propagate_batch(graph, X):
    batch_sizes = [1, 2, 4, 5, 8, 10, 16, 20, 32, 50, 64, 100, 128, 200, 256, 500, 512, \
    1000, 1024, 2000, 2048, 4096, 5000, 8192, 10000, 16384, 20000, 32768, 50000, 60000]
    N = 1
    durations_oe = []
    durations_ma = []
    for batch_size in batch_sizes:
        print(batch_size)

        start_time = time.time()
        for _ in range(N):
            propagate_oe(graph, X[0:batch_size])
        durations_oe.append((time.time() - start_time) / N)

        start_time = time.time()
        for _ in range(N):
            graph.propagate(X[0:batch_size])
        durations_ma.append((time.time() - start_time) / N)
    display_tab(['Durée un à la fois', 'Durée avec matrice'], [durations_oe, durations_ma])

def benchmark_with_confidence_interval():
    global graph
    start_time = time.time()
    accuracies_test = []
    for _ in range(10):
        for j in range(nb_times_datasets):
            for i in range(0, X.shape[0], batch_size):
                print(i)
                print(graph.batch_gradient_descent([X[i:i+batch_size]], [ohe_Y[i:i+batch_size]], learning_rate) / batch_size)
        accuracies_test.append(accuracy(graph, X_test, Y_test))
        graph = fully_connected(layers)
    print(time.time() - start_time)
    display_tab(['Précision sur le test set'], [accuracies_test])"""
    

def cost_plots():
    start_time = time.time()
    t = []
    accuracies_training = []
    accuracies_test = []
    c = 0
    # Optimization algorithm
    sgd = GradientDescent(graph.get_learnable_nodes(), learning_rate)
    for j in range(nb_times_datasets):
        for i in range(0, X.shape[0], batch_size):
            print(i)
            graph.propagate([X[i:i+batch_size]])
            cost = graph.backpropagate([ohe_Y[i:i+batch_size]])
            sgd.optimize(batch_size)
            print(cost / batch_size)
            if (c % 32) == 0:
                t.append(c)
                accuracies_training.append(accuracy(graph, X, Y))
                accuracies_test.append(accuracy(graph, X_test, Y_test))
            c += 1

    print('DURATION: ', time.time() - start_time)
    print(t, accuracies_training, accuracies_test)
    plt.plot(t, accuracies_training, label='apprentissage')
    plt.plot(t, accuracies_test, label='test')
    plt.xlabel("Nombre d'exemples")
    plt.ylabel('Précision')
    plt.title("Précision en fonction du nombre d'exemples vus (batch de 128, une couche softmax)")
    plt.legend(loc=4)
    plt.show()

def display_tab(headers, columns):
    print(', '.join(headers))
    for i in range(len(columns[0])):
        print(', '.join(str(column[i]) for column in columns))

def display_weights(W):
    for i, w in enumerate(W.T):
        plt.subplot(2, 5, i+1)
        print(w.shape)
        image = w[1:].reshape((28, 28))
        plt.imshow(image, cmap='Greys', interpolation='none')
        plt.title(str(i))
        #plt.colorbar()
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    layers = [10]
    batch_size = 128
    learning_rate = 1
    nb_times_datasets = 1
    graph = fully_connected(layers)

    X, (nb_rows, nb_columns), Y = get_training_set('examples/mnist')
    print(X.shape)
    X, Y = shuffle_dataset(X, Y)
    X = normalized_dataset(X)
    Y = Y.reshape((len(Y), 1))
    ohe_Y = one_hot_encode(Y)

    X_test, (_, _), Y_test = get_test_set('examples/mnist')
    print(X_test.shape)
    X_test, Y_test = shuffle_dataset(X_test, Y_test)
    X_test = normalized_dataset(X_test)
    Y_test = Y_test.reshape((len(Y_test), 1))

    cost_plots()
    #benchmark_with_confidence_interval()

    display_weights(graph.learnable_nodes[-1].w)