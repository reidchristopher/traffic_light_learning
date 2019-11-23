#!/usr/bin/env python3

# tensorflow for neural networks
import tensorflow as tf
import numpy as np


class TFNetwork:

    @staticmethod
    def from_existing(network):

        return TFNetwork(None, None, None, None, network.model)

    @staticmethod
    def from_save(file_path):

        model = tf.keras.models.load_model(file_path)

        return TFNetwork(None, None, None, None, model)

    def __init__(self, input_size, output_size, hidden_layer_size, num_hidden_layers, original_model=None):

        if original_model is not None:

            self.input_size = original_model.layers[0].input_shape[-1]
            self.output_size = original_model.layers[-1].output_shape[-1]
            self.hidden_layer_size = original_model.layers[0].output_shape[-1]
            self.num_hidden_layers = len(original_model.layers) - 1

            self.model = tf.keras.models.clone_model(original_model)

            for i in range(self.num_hidden_layers + 1):

                original_layer = original_model.get_layer(index=i)

                copied_layer = self.model.get_layer(index=i)

                copied_layer.set_weights(original_layer.get_weights())

        else:
            assert (num_hidden_layers > 0)
            assert(input_size is not None)
            assert(output_size is not None)
            assert(hidden_layer_size is not None)
            assert(num_hidden_layers is not None)

            self.input_size = input_size
            self.output_size = output_size
            self.hidden_layer_size = hidden_layer_size
            self.num_hidden_layers = num_hidden_layers

            self.model = tf.keras.Sequential()

            # add hidden layers
            for i in range(num_hidden_layers):

                if i == 0:
                    input_shape = (input_size,)
                else:
                    input_shape = (hidden_layer_size,)

                self.model.add(tf.keras.layers.Dense(hidden_layer_size, input_shape=input_shape, activation='relu'))

            # add output layer
            self.model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

        self.model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

    def mutate(self, mutation_fraction, stddev):  # other args? (noise distribution, etc)

        assert (0 < mutation_fraction <= 1)

        for layer in self.model.layers:

            weights = layer.get_weights()

            for j in range(len(weights)):

                mask = np.random.choice([0, 1], weights[j].shape, replace=True, p=[(1 - mutation_fraction), mutation_fraction])

                noise = np.random.normal(scale=stddev, size=weights[j].shape)

                weights[j] += mask * noise

            layer.set_weights(weights)

    def train(self, x, y, epochs, batch_size=None):

        assert(len(x) == len(y))

        if batch_size is None:
            batch_size = len(x)

        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)

    def save(self, file_path):

        assert file_path[-3:] == '.h5'

        self.model.save(file_path)

    def get_selection(self, input_vector, one_hot=True):

        input_vector = np.ndarray( (1, len(input_vector)), dtype=np.float32)

        # inputs and outputs are 2D arrays
        output_vector = self.model(input_vector).numpy()[0]

        if one_hot:
            return output_vector == np.max(output_vector)
        else:
            return np.argmax(output_vector)


def test():
    try:
        # runs a test of the neural network and training on the MNIST dataset
        example = TFNetwork(28 * 28, 10, 100, 3)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        y_train = tf.one_hot(y_train, 10).numpy()
        y_test = tf.one_hot(y_test, 10).numpy()

        # Preprocess the data (these are Numpy arrays)
        x_train = x_train.reshape(60000, 784).astype('float32') / 255
        x_test = x_test.reshape(10000, 784).astype('float32') / 255

        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')

        epochs = 10
        batch_size = 100

        example.train(x_train, y_train, epochs, batch_size)

        result = example.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=False)

        print("Test results")
        print("Loss: %.4f" % result[0])
        print("Accuracy: %.4f" % result[1])

    except KeyboardInterrupt:
        print("Test interrupted")


if __name__ == "__main__":
    test()
