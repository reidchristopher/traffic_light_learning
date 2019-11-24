#!/usr/bin/env python

from tf_network import TFNetwork
import tensorflow as tf
import numpy as np


class ReinforcementLearner:

    def __init__(self, state_size, action_size, hidden_layer_size, num_hidden_layers, discount_rate=0.99):

        self.state_size = state_size
        self.action_size = action_size
        self.discount_rate = discount_rate

        self.actor = TFNetwork(state_size, action_size, hidden_layer_size, num_hidden_layers,
                               output_activation=tf.keras.activations.linear)

        self.critic = TFNetwork(state_size, 1, hidden_layer_size, num_hidden_layers,
                                output_activation=tf.keras.activations.linear)

        self.states = []
        self.actions = []
        self.rewards = []

        self.optimizer = tf.optimizers.RMSprop()

    def get_selection(self, x, get_max=False):

        np_x = np.reshape(x, (1, len(x)))

        logits = self.actor.model(np_x)

        if get_max:
            return np.argmax(logits)
        else:
            probs = np.array(tf.nn.softmax(logits)[0], dtype=np.float64)

            probs /= np.sum(probs)

            choice = np.random.choice(self.actor.output_size, 1, p=probs)

            return choice[0]

    def record_result(self, state, action, reward):

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def update_weights(self):

        with tf.GradientTape() as tape:
            loss = self.compute_loss()

        gradients = tape.gradient(loss, self.actor.model.trainable_weights + self.critic.model.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.actor.model.trainable_weights + self.critic.model.trainable_weights))

        self.clear_memory()

    def clear_memory(self):

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def compute_loss(self):

        discounted_rewards = []

        reward_sum = 0  # self.critic.get_output(state).numpy()[0] # TODO use current state for critic value start?

        discounted_rewards = []
        for reward in self.rewards[::-1]:
            reward_sum = reward + self.discount_rate * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits = self.actor.model(np.vstack(self.states))

        values = self.critic.model(np.vstack(self.states))

        advantage = tf.convert_to_tensor(np.array(discounted_rewards, dtype=np.float32)[:, None] - values)

        value_loss = advantage ** 2

        policy = tf.nn.softmax(logits)
        entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions,
                                                                     logits=logits)

        policy_loss *= tf.stop_gradient(advantage)

        policy_loss -= 0.01 * entropy

        total_loss = tf.reduce_mean(0.5 * value_loss + policy_loss)

        return total_loss

    def evaluate(self, x_test, y_test):

        answers = np.argmax(tf.nn.softmax(self.actor.model(x_test)), axis=1)

        return np.ones(y_test.shape)[answers == y_test].sum() / len(answers)


def test():

    try:
        # runs a test of the neural network and training on the MNIST dataset
        example = ReinforcementLearner(28 * 28, 10, 100, 3, discount_rate=0)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Preprocess the data (these are Numpy arrays)
        x_train = x_train.reshape(60000, 784).astype('float32') / 255
        x_test = x_test.reshape(10000, 784).astype('float32') / 255

        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')

        epochs = 10
        batch_size = 100

        from math import ceil
        num_batches = ceil(x_train.shape[0] / batch_size)

        print("Start Accuracy")
        print(example.evaluate(x_test, y_test))

        for epoch in range(epochs):

            print("Epoch %d" % epoch)

            for i in range(num_batches):

                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, x_train.shape[0])

                x_batch = x_train[start_index:end_index]
                y_batch = y_train[start_index:end_index]

                for x, y in zip(x_batch, y_batch):

                    action = example.get_selection(x)

                    reward = float(y == action)

                    example.record_result(x, action, reward)

                example.update_weights()

            print("New accuracy")
            print(example.evaluate(x_test, y_test))

    except KeyboardInterrupt:
        print("Test interrupted")


if __name__ == "__main__":

    test()
