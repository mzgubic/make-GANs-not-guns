import tensorflow as tf

class TFEnvironment:

    def __init__(self, generator, adversary, fetch_data, fetch_noise, config=tf.ConfigProto(intra_op_parallelism_threads = 32,
                                                                                         inter_op_parallelism_threads = 32,
                                                                                         allow_soft_placement = True,
                                                                                         device_count = {'CPU': 2})):

        print('--- Starting TensorFlow session')

        # store classifier and adversary
        self.generator = generator
        self.adversary = adversary
        self.fetch_data = fetch_data
        self.fetch_noise = fetch_noise

        # store the tebsorboard summaries here
        self.summaries = []

        # make a session
        self.sess = tf.Session(config=config)

    def build(self):

        print('--- Building computational graph')

        # make input placeholders
        noise_sample = self.fetch_noise(10)
        self._input_noise = tf.placeholder(tf.float32, shape=(-1, noise_sample.shape[1]), name='InputNoise')
        self._real_data = tf.placeholder(tf.float32, shape=(-1, 1), name='DataBatch')

        # create the computational graph for fake data
        self.generator.make_forward_pass(self._input_noise)
        self.summaries.append(tf.summary.histogram('GAN_data', self.generator.output))

        # create the input to the adversry
        comb_data = tf.concat([self._real_data, self.generator.output], axis=0)
        n_real = tf.shape(self._real_data, 'n_real')[0]
        n_fake = tf.shape(self.generator.output, 'n_fake')[0]
        comb_labels = tf.one_hot(tf.constant([1]*n_real + [0]*n_fake, dtype=tf.int32), depth=self.adversary.n_classes)

        # create the adversary forward pass
        self.adversary.make_forward_pass(comb_data)

        # losses
        self.adversary.make_loss(labels)
        self.generator.make_loss(self.adversary.loss)

        # create the optimisation graphs
        self.adversary.make_opt()
        self.generator.make_opt()

        # store adversary loss
        self.summaries.append(tf.summary.scalar('adv_loss', self.adversary.loss))

        # create the file writer (for tensorboard)
        self.writer = tf.summary.FileWriter(logdir='tensorboard')

    def initialise_variables(self):

        print('--- Initialising TensorFlow variables')

        self.sess.run(tf.global_variables_initializer())



