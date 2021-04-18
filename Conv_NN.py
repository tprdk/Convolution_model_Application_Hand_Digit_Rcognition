import os
import matplotlib.pyplot as plt
import numpy as np
from cnn_utils import random_mini_batches
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])  # Features
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y


def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    tf.set_random_seed(1)  # so that your "random" numbers match ours
    W1 = tf.get_variable("W1", shape=[4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", shape=[2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1,
                  "W2": W2}
    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    - Conv2D: stride 1, padding is "SAME"
    - ReLU
    - Max pool: Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
    - Conv2D: stride 1, padding is "SAME"
    - ReLU
    - Max pool: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
    - Flatten the previous output.
    - FULLYCONNECTED (FC) layer: Apply a fully connected layer without an non-linear activation function.

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    conv_layer_out = tf.nn.conv2d(input=X, filter=W1, strides=[1, 1, 1, 1], padding='SAME')
    relu_out = tf.nn.relu(conv_layer_out)
    pooling_out = tf.nn.max_pool(relu_out, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

    conv_layer_out = tf.nn.conv2d(input=pooling_out, filter=W2, strides=[1, 1, 1, 1], padding='SAME')
    relu_out = tf.nn.relu(conv_layer_out)
    pooling_out = tf.nn.max_pool(relu_out, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    flat_input = tf.contrib.layers.flatten(pooling_out)
    return tf.contrib.layers.fully_connected(flat_input, 6, activation_fn=None)


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.005,
          num_epochs=100, minibatch_size=64, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    tf.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()

    forward_prop_out = forward_propagation(X, parameters)

    cost = compute_cost(forward_prop_out, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            mini_batch_cost = 0.
            num_mini_batches = int(m / minibatch_size)
            seed += 1
            mini_batches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for mini_batch in mini_batches:
                (batch_X, batch_Y) = mini_batch
                _ , batch_cost = sess.run([optimizer, cost], feed_dict={X: batch_X, Y: batch_Y})

                mini_batch_cost += batch_cost / num_mini_batches

            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, mini_batch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(mini_batch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(forward_prop_out, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters



