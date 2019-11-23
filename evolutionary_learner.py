#!/usr/bin/env python

from tf_network import TFNetwork
import numpy as np
import random
import tensorflow as tf


class EvolutionaryLearner:

    def __init__(self, population_size, input_size, output_size, hidden_layer_size, num_hidden_layers):

        self.ready_for_selection = False

        self.population_size = population_size

        self.networks = [TFNetwork(input_size, output_size, hidden_layer_size, num_hidden_layers)
                         for _ in range(population_size)]

    def generate_mutations(self, mutation_fraction, stddev):

        mutations = [None for _ in range(self.population_size)]

        for i, network in enumerate(self.networks):

            new_network = TFNetwork.from_existing(network)

            new_network.mutate(mutation_fraction, stddev)

            mutations[i] = new_network

        self.networks = self.networks + mutations

        self.ready_for_selection = True

    def select(self, fitnesses, epsilon):

        assert (len(fitnesses) == len(self.networks))
        assert self.ready_for_selection

        selection_indices = [i for i in range(len(self.networks))]
        random.shuffle(selection_indices)

        best_network_index = np.argmax(fitnesses)

        successors = [None for _ in range(self.population_size)]

        for i in range(self.population_size):

            index_1 = selection_indices[2 * i]
            index_2 = selection_indices[2 * i + 1]

            if index_1 == best_network_index:
                successors[i] = self.networks[index_1]
            elif index_2 == best_network_index:
                successors[i] = self.networks[index_2]
            else:
                roll = random.random()
                if fitnesses[index_1] > fitnesses[index_2]:
                    better_index = index_1
                    worse_index = index_2
                else:
                    better_index = index_2
                    worse_index = index_1

                if roll > 0.1:
                    successors[i] = self.networks[better_index]
                else:
                    successors[i] = self.networks[worse_index]

        self.networks = successors

        tf.keras.backend.clear_session()

        self.ready_for_selection = False

    def evaluate(self, x, y, batch_size):

        scores = [None for _ in range(len(self.networks))]

        for i, network in enumerate(self.networks):

            results = network.model.evaluate(x, y, batch_size=batch_size, verbose=False)

            scores[i] = results[1]

        return scores

    def get_selection(self, x, network_index, get_one_hot=False):

        return self.networks[network_index].get_selection(x, one_hot=get_one_hot)


def test():

    try:
        example = EvolutionaryLearner(10, 28 * 28, 10, 10, 3)

        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        y_train = tf.one_hot(y_train, 10).numpy()
        y_test = tf.one_hot(y_test, 10).numpy()

        # Preprocess the data (these are Numpy arrays)
        x_train = x_train.reshape(60000, 784).astype('float32') / 255
        x_test = x_test.reshape(10000, 784).astype('float32') / 255

        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')

        epochs = 10
        batch_size = 10000

        print("Pre-training accuracies")
        print(example.evaluate(x_test, y_test, batch_size))

        from math import ceil
        num_batches = ceil(x_train.shape[0] / batch_size)

        for epoch in range(epochs):

            print("Epoch %d" % epoch)

            for i in range(num_batches):

                print("\tBatch %d" % i)
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, x_train.shape[0])

                example.generate_mutations(0.10, 0.10)

                scores = example.evaluate(x_train[start_index:end_index], y_train[start_index:end_index], batch_size=batch_size)

                example.select(scores, 0.1)

            print("New accuracies")
            print(example.evaluate(x_test, y_test, batch_size))

        print("Post training accuracies")
        print(example.evaluate(x_test, y_test, batch_size))

    except KeyboardInterrupt:
        print("\n\rTest interrupted")


if __name__ == "__main__":

    test()