import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mnist


def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:].reshape((m,Y.shape[1]))

    num_complete_minibatches = int(math.floor(m/mini_batch_size))
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def predict(X, parameters):

    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})

    return prediction

def forward_propagation_for_predict(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def load_mnist_data():
    x_train = mnist.train_images()
    y_train = mnist.train_labels()
    x_test = mnist.test_images()
    y_test = mnist.test_labels()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = (x_train / 255)
    x_test = (x_test / 255)

    x_train = x_train.reshape((-1,28,28,1))
    x_test = x_test.reshape((-1,28,28,1))

    y_train = convert_to_one_hot(y_train, 10)
    y_test = convert_to_one_hot(y_test, 10)

    return x_train, y_train, x_test, y_test

def create_placeholders(n_H, n_W, n_C, n_y):
    X = tf.placeholder(tf.float32,shape=(None, n_H, n_W, n_C))
    Y = tf.placeholder(tf.float32,shape=(None, n_y))
    return X, Y

def initialize_parameters():
    W1 = tf.get_variable("W1",[3,3,1,32],initializer = tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2",[3,3,32,64],initializer = tf.contrib.layers.xavier_initializer())

    FW3 = tf.get_variable("FW3",[9216, 128],initializer = tf.contrib.layers.xavier_initializer())
    Fb3 = tf.get_variable("Fb3",[1, 128],initializer =  tf.zeros_initializer())

    FW4 = tf.get_variable("FW4",[128, 10],initializer = tf.contrib.layers.xavier_initializer())
    Fb4 = tf.get_variable("Fb4",[1, 10],initializer = tf.zeros_initializer())

    parameters = {"W1": W1, "W2": W2, "FW3": FW3, "Fb3": Fb3, "FW4": FW4, "Fb4": Fb4}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    FW3 = parameters['FW3']
    Fb3 = parameters['Fb3']
    FW4 = parameters['FW4']
    Fb4 = parameters['Fb4']

    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1], padding='VALID')
    A1 = tf.nn.relu(Z1)
    Z2 = tf.nn.conv2d(A1,W2,strides=[1,1,1,1], padding='VALID')
    A2 = tf.nn.relu(Z2)
    P1 = tf.nn.max_pool(A2,ksize= [1,2,2,1], strides=[1,2,2,1], padding= 'SAME')
    F = tf.contrib.layers.flatten(P1)

    Z3 = tf.add(tf.matmul(F, FW3), Fb3)
    A3 = tf.nn.relu(Z3)

    Z4 = tf.add(tf.matmul(A3,FW4), Fb4)

    return Z4

def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z3, labels = Y)) # axis = -1, Last dim is the class dim
    return cost


X_train, Y_train, X_test, Y_test = load_mnist_data()

(m, n_H, n_W, n_C) = X_train.shape
n_y = 10
costs = []

learning_rate = 0.009
num_epochs = 3
minibatch_size = 32
print_cost = True


tf.reset_default_graph()
X, Y = create_placeholders(n_H, n_W, n_C, n_y)
parameters = initialize_parameters()
Z4 = forward_propagation(X,parameters)
cost = compute_cost(Z4,Y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate= learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        epoch_cost = 0.
        num_minibatches = int(m / minibatch_size)
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            _ , minibatch_cost = sess.run([optimizer,cost],feed_dict= {X :minibatch_X, Y : minibatch_Y})
            epoch_cost += minibatch_cost / num_minibatches

        if print_cost == True and epoch % 1 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(epoch_cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    parameters = sess.run(parameters)
    print ("Parameters have been trained!")
    correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
