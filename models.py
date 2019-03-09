import tensorflow as tf
import tensorflow.contrib.layers as layers


class Model:

    def __init__(self, name, **kwargs):

        # settings
        self.name = name
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

        # placeholders
        self.output = None
        self.tf_vars = None


class GeneratorFullyConnected(Model):

    def make_forward_pass(self, input_noise):

        with tf.variable_scope(self.name):

            layer = input_noise
            
            for _ in range(self.depth):
                layer = layers.relu(layer, self.n_units)

            self.output = layers.linear(layer, self.output_dim)

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def make_loss(self, adv_loss):

        self.loss = - adv_loss

    def make_opt(self):

        self.opt = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.tf_vars)


class AdversaryFullyConnected(Model):

    def make_forward_pass(self, data):

        with tf.variable_scope(self.name):

            layer = data

            for _ in range(self.depth):
                layer = layers.relu(layer, self.n_units)

            self.logits = layers.linear(layer, self.n_classes)

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def make_loss(self, labels):

        self.labels = labels
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.logits)
        self.loss = tf.math.reduce_mean(self.loss)

    def make_opt(self):

        self.opt = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.tf_vars)





