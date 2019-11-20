#!/usr/bin/env python3

# tensorflow for neural networks
import tensorflow as tf
import numpy as np


class TFNetwork:
    session = tf.compat.v1.Session()

    def __init__(self, input_size, output_size, hidden_layer_size, num_hidden_layers):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers

        # initialize neural network
        self.x = tf.compat.v1.placeholder(tf.float32, [None, input_size])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, output_size])

        self.hidden_weights = [None for i in range(num_hidden_layers)]
        self.hidden_biases = [None for i in range(num_hidden_layers)]

        self.hidden_outputs = [None for i in range(num_hidden_layers)]

        for i in range(num_hidden_layers):
            if i == 0:
                num_inputs = input_size
                inputs = self.x
            else:
                num_inputs = hidden_layer_size
                inputs = self.hidden_outputs[i - 1]

            self.hidden_weights[i] = tf.Variable(tf.random.normal([num_inputs, hidden_layer_size], stddev=0.05),
                                                 name="hw%d" % i)
            self.hidden_biases[i] = tf.Variable(tf.random.normal([hidden_layer_size], stddev=0.05), name="hb%d" % i)

            self.hidden_outputs[i] = tf.nn.relu(
                tf.add(tf.matmul(inputs, self.hidden_weights[i]), self.hidden_biases[i]))

        self.output_weights = tf.Variable(tf.random.normal([hidden_layer_size, output_size], stddev=0.05), name="ow")
        self.output_biases = tf.Variable(tf.random.normal([output_size], stddev=0.05), name="ob")

        self.output = tf.nn.softmax(tf.add(tf.matmul(self.hidden_outputs[-1], self.output_weights), self.output_biases))

        # initialize global variables
        global_init_op = tf.compat.v1.global_variables_initializer()
        self.session.run(global_init_op)

    def get_copy(self):

        new_network = TFNetwork(self.input_size, self.output_size, self.hidden_layer_size, self.num_hidden_layers)

        for i in range(self.num_hidden_layers):
            copy_op = new_network.hidden_weights[i].assign(self.hidden_weights[i])

            new_network.session.run(copy_op)

            copy_op = new_network.hidden_biases[i].assign(self.hidden_biases[i])

            new_network.session.run(copy_op)

        copy_op = new_network.output_weights.assign(self.output_weights)

        new_network.session.run(copy_op)

        copy_op = new_network.output_biases.assign(self.output_biases)

        new_network.session.run(copy_op)

        return new_network

    def mutate(self, mutation_fraction, stddev):  # other args? (noise distribution, etc)

        assert (0 < mutation_fraction <= 1)

        nn_weights = self.hidden_weights + self.hidden_biases + [self.output_weights, self.output_biases]

        for weights in nn_weights:
            mask = np.random.choice([0, 1], weights.shape, replace=True, p=[(1 - mutation_fraction), mutation_fraction])

            noise = np.random.normal(scale=stddev, size=weights.shape)

            assign_op = weights.assign_add(mask * noise)

            self.session.run(assign_op)

    def train(self, x, y, epochs, learning_rate=0.1, batch_size=None, print_cost=False):

        assert(len(x) == len(y))
        assert(len(x) == len(y))

        if batch_size is None:
            batch_size = len(x)

        num_batches = int(len(x) / batch_size)

        clipped_output = tf.clip_by_value(self.output, 1e-10, 0.9999999)
        cost = -tf.reduce_mean(tf.reduce_sum(self.y * tf.math.log(clipped_output)
                                             + (1 - self.y) * tf.math.log(1 - clipped_output), axis=1))

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        for epoch in range(epochs):
            avg_cost = 0
            for i in range(num_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, len(x))
                batch_x, batch_y = x[start_index:end_index], y[start_index:end_index]
                _, c = self.session.run([optimizer, cost],
                                        feed_dict={self.x: batch_x, self.y: batch_y})
                avg_cost += c / num_batches

            if print_cost:
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))


def test():
    # runs a test of the neural network constructed in the LearningPolicy object on the MNIST dataset

    example = TFNetwork(28 * 28, 10, 100, 3)

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    learning_rate = 0.25
    epochs = 10
    batch_size = 100

    correct_prediction = tf.equal(tf.argmax(example.y, 1), tf.argmax(example.output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    x_train, y_train = mnist.train.images, mnist.train.labels

    example.train(x_train, y_train, epochs, learning_rate, batch_size, print_cost=True)

    print("MNIST accuracy: %.3f" % example.session.run(accuracy, feed_dict={example.x: mnist.test.images, example.y: mnist.test.labels}))


if __name__ == "__main__":
    test()
