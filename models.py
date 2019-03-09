import tensorflow as tf
import tensorflow.contrib.layers as layers


class Model:

    def __init__(self, name, hps):

        # settings
        self.name = name
        self._hps = hps

        # placeholders
        self.output = None
        self.tf_vars = None


class GeneratorFullyConnected(Model):

    def make_forward_pass(input_noise, depth, n_units, output_dim):

        with tf.variable_scope(self.name):

            layer = input_noise
            
            for _ in range(depth):
                layer = layers.relu(layer, n_units)

            self.output = layers.linear(layer, output_dim)

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)


class AdversaryFullyConnected(Model):

    def make_forward_pass(data, depth=3, n_units=10, n_classes=2):

        with tf.variable_scope(self.name):

            layer = data

            for _ in range(depth):
                layer = layers.relu(layer, n_units)

            self.logits = layers.linear(layer, n_classes)

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

