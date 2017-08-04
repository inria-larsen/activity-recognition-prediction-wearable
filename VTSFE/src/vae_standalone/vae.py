# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ..lib.useful_functions import glorot_init, interval_limiter



# Variational Autoencoder
class VAE():
    """ Variation Autoencoder (VAE) implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    HYPERPARAMS = {
        "learning_rate": 1E-3,
        #"dropout": 0.9,
        #"lambda_l2_reg": 1E-5,
        "activation_function": tf.nn.softplus,
        "output_function": tf.nn.sigmoid,
        "vae_architecture": {
            "L": 100,                  # number of Monte Carlo samples to reconstruct x (the observed value)
            "n_hidden_encoder_1": 500, # 1st layer encoder neurons
            "n_hidden_encoder_2": 500, # 2nd layer encoder neurons
            "n_hidden_decoder_1": 500, # 1st layer decoder neurons
            "n_hidden_decoder_2": 500, # 2nd layer decoder neurons
            "n_input": 784,            # dimensionality of observations
            "n_z": 20,                 # dimensionality of latent space
            "n_output": 784            # dimensionality of observations
        }
    }


    def __init__(self, hyperparams={},
        input_tensor=None,
        target_tensor=None,
        save_path=None,
        reuse_encoder_weights=False,
        reuse_decoder_weights=False,
        optimize=False,
        binary=False,
        estimate_z_derivative=False,
        z_continuity_error_coeff=None,
        apply_latent_loss=False,
        base_scope_encoder="vae",
        base_scope_decoder="vae",
        log_sigma_sq_values_limit=None
    ):

        self.__dict__.update(self.HYPERPARAMS, **hyperparams)

        self.base_scope_encoder = base_scope_encoder
        self.base_scope_decoder = base_scope_decoder
        self.save_path = save_path
        self.reuse_encoder_weights = reuse_encoder_weights
        self.reuse_decoder_weights = reuse_decoder_weights
        self.optimize = optimize
        self.binary = binary
        self.estimate_z_derivative = estimate_z_derivative
        self.z_continuity_error_coeff = z_continuity_error_coeff
        self.apply_latent_loss = apply_latent_loss
        self.log_sigma_sq_values_limit = log_sigma_sq_values_limit

        # tf Graph input
        if input_tensor == None:
            self.x = tf.placeholder(tf.float32, [None, self.vae_architecture["n_input"]], name="input_x")
            self.n_input = self.vae_architecture["n_input"]
        else:
            self.x = input_tensor
            shape = self.x.get_shape()
            self.n_input = int(shape[-1])

        if target_tensor == None:
            self.x_target = self.x
            self.n_output = self.vae_architecture["n_ouput"]
        else:
            self.x_target = target_tensor
            shape = self.x_target.get_shape()
            self.n_output = int(shape[-1])

        # Create autoencoder network
        self.create_network()
        # Defines loss function based on variational lower-bound
        self.create_loss_function()
        # and corresponding optimizer
        self.create_loss_optimizer()

        print(self.base_scope_encoder+" network created.")


    def create_network(self):
        # Initialize autoencoder network weights and biases
        self.network_weights = self.initialize_weights()

        # Use encoder network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        encoder = self.encoder_network(
            self.network_weights["weights_encoder"],
            self.network_weights["biases_encoder"]
        )
        self.z_mean = encoder[0]
        self.z_log_sigma_sq = encoder[1]

        if self.estimate_z_derivative:
            self.z_derivative = encoder[2]

        n_z = self.vae_architecture["n_z"]
        half = tf.constant(0.5, dtype=tf.float32)
        z_sigma = tf.exp(
            tf.scalar_mul(
                half,
                self.z_log_sigma_sq
            )
        )

        # Monte Carlo sampling
        # z_sample = mu + sigma*epsilon
        self.z_samples = [self.z_mean]
        for i in range(self.vae_architecture["L"]-1):
            eps = tf.random_normal(tf.shape(self.z_mean), mean=0.0, stddev=1.0, dtype=tf.float32)
            # Draw one sample z from Gaussian distribution
            # z_samples shape = [L, batch_size, n_z]
            self.z_samples.append(
                tf.add(
                    self.z_mean,
                    tf.multiply(
                        z_sigma,
                        eps
                    )
                )
            )

        # to allow direct retrieving of z
        self.z = self.z_mean

        # x_reconstr_means, x_reconstr_log_sigma_sqs shape = [L, batch_size, n_output]
        self.x_reconstr_means, self.x_reconstr_log_sigma_sqs = self.decoder_network(
            self.z_samples,
            self.network_weights["weights_decoder"],
            self.network_weights["biases_decoder"]
        )
        # to allow direct retrieving of reconstructed input x
        # x_reconstr shape = [batch_size, n_output]
        self.x_reconstr = self.x_reconstr_means[0]


    def initialize_2hlayers(self, base_scope, scope_name, reuse_weights, all_weights, n_hidden_1, n_hidden_2, n_in, n_out, estimate_z_derivative=False, use_sigma=True):

        with tf.variable_scope(base_scope+'.'+scope_name, reuse=reuse_weights) as scope:
            all_weights['weights_'+scope_name] = {
                'h1': tf.get_variable("h1", initializer=glorot_init(n_in, n_hidden_1))
            }

            all_weights['biases_'+scope_name] = {
                'b1': tf.get_variable("b1", initializer=tf.zeros([n_hidden_1]), dtype=tf.float32)
            }

            if n_hidden_2 != None:
                all_weights['weights_'+scope_name]['h2'] = tf.get_variable("h2", initializer=glorot_init(n_hidden_1, n_hidden_2))
                all_weights['biases_'+scope_name]['b2'] = tf.get_variable("b2", initializer=tf.zeros([n_hidden_2]), dtype=tf.float32)
                n_hidden = n_hidden_2
            else:
                n_hidden = n_hidden_1

            all_weights['weights_'+scope_name].update({
                'out_mean': tf.get_variable("out_mean_weights", initializer=glorot_init(n_hidden, n_out))
            })

            all_weights['biases_'+scope_name].update({
                'out_mean': tf.get_variable("out_mean_biases", initializer=tf.zeros([n_out]), dtype=tf.float32)
            })

            if use_sigma:
                all_weights['weights_'+scope_name].update({
                    'out_log_sigma': tf.get_variable("out_log_sigma_weights", initializer=glorot_init(n_hidden, n_out))
                })

                all_weights['biases_'+scope_name].update({
                    'out_log_sigma': tf.get_variable("out_log_sigma_biases", initializer=tf.zeros([n_out]), dtype=tf.float32)
                })

            if estimate_z_derivative:
                all_weights['weights_'+scope_name].update({
                    'out_derivative_mean': tf.get_variable("out_derivative_mean_weights", initializer=glorot_init(n_hidden, n_out))
                })
                all_weights['biases_'+scope_name].update({
                    'out_derivative_mean': tf.get_variable("out_derivative_mean_biases", initializer=tf.zeros([n_out]), dtype=tf.float32)
                })

            if reuse_weights:
                print(scope.name+" weights reused.")
            else:
                print(scope.name+" weights initialized.")

        return all_weights


    def initialize_weights(self):

        all_weights = {}
        all_weights = self.initialize_encoder(all_weights)
        all_weights = self.initialize_decoder(all_weights)

        return all_weights


    def initialize_encoder(self, all_weights):

        return self.initialize_2hlayers(
                    self.base_scope_encoder,
                    "encoder",
                    self.reuse_encoder_weights,
                    all_weights,
                    self.vae_architecture["n_hidden_encoder_1"],
                    self.vae_architecture["n_hidden_encoder_2"],
                    self.n_input,
                    self.vae_architecture["n_z"],
                    estimate_z_derivative=self.estimate_z_derivative
                )


    def initialize_decoder(self, all_weights):

        return self.initialize_2hlayers(
                    self.base_scope_decoder,
                    "decoder",
                    self.reuse_decoder_weights,
                    all_weights,
                    self.vae_architecture["n_hidden_decoder_1"],
                    self.vae_architecture["n_hidden_decoder_2"],
                    self.vae_architecture["n_z"],
                    self.n_output,
                    use_sigma=self.vae_architecture["use_reconstr_sigma"]
                )


    def encoder_network(self, weights, biases):
        # Generate probabilistic encoder (encoder network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.activation_function(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))

        if 'h2' in weights:
            hidden_layer = self.activation_function(tf.add(tf.matmul(layer_1, weights['h2']),
                                               biases['b2']))
        else:
            hidden_layer = layer_1

        z_mean = tf.add(tf.matmul(hidden_layer, weights['out_mean']),
                        biases['out_mean'])

        z_log_sigma_sq = tf.add(tf.matmul(hidden_layer, weights['out_log_sigma']),
                        biases['out_log_sigma'])

        if self.log_sigma_sq_values_limit != None:
            # Limiting values of log_sigma_sq to keep numerical stability
            # = keeping sigma >= exp(-limit/2) and <= exp(limit/2)
            z_log_sigma_sq = interval_limiter(z_log_sigma_sq, self.log_sigma_sq_values_limit)

        output = (z_mean, z_log_sigma_sq)

        if self.estimate_z_derivative:
            z_derivative_mean = tf.add(tf.matmul(hidden_layer, weights['out_derivative_mean']),
                            biases['out_derivative_mean'])

            # z_derivative_log_sigma_sq = tf.add(tf.matmul(hidden_layer, weights['out_derivative_log_sigma']),
            #                 biases['out_derivative_log_sigma'])

            output += (z_derivative_mean,)

        return output


    def decoder_network(self, z_samples, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.

        x_reconstr_means = []
        x_reconstr_log_sigma_sqs = []
        # creates L decoder networks to reconstruct L samples of p(x|z)
        for z in z_samples:
            layer_1 = self.activation_function(
                tf.add(
                    tf.matmul(
                        z,
                        weights['h1']
                    ),
                    biases['b1']
                )
            )

            if 'h2' in weights:
                hidden_layer = self.activation_function(
                    tf.add(
                        tf.matmul(
                            layer_1,
                            weights['h2']
                        ),
                        biases['b2']
                    )
                )
            else:
                hidden_layer = layer_1

            x_reconstr_mean = self.output_function(
                tf.add(
                    tf.matmul(
                        hidden_layer,
                        weights['out_mean']
                    ),
                    biases['out_mean']
                )
            )
            x_reconstr_log_sigma_sq = None
            if not self.binary and self.vae_architecture["use_reconstr_sigma"]:
                x_reconstr_log_sigma_sq = tf.add(
                    tf.matmul(
                        hidden_layer,
                        weights['out_log_sigma']
                    ),
                    biases['out_log_sigma']
                )

            if self.log_sigma_sq_values_limit != None and x_reconstr_log_sigma_sq != None:
                # Limiting values of log_sigma_sq to keep numerical stability
                # = keeping sigma >= exp(-limit/2) and <= exp(limit/2)
                x_reconstr_log_sigma_sq = interval_limiter(x_reconstr_log_sigma_sq, self.log_sigma_sq_values_limit)

            x_reconstr_means.append(x_reconstr_mean)
            x_reconstr_log_sigma_sqs.append(x_reconstr_log_sigma_sq)

        return x_reconstr_means, x_reconstr_log_sigma_sqs


    def create_reconstruction_loss(self, x_reconstr_means, x_reconstr_log_sigma_sqs, nb_samples):
        self.reconstr_losses = []

        if self.binary: # if data is binary
            for i in range(len(x_reconstr_means)):
                # loss sum on all dimensions
                # Adding 1e-10 to avoid evaluation of log(0.0)
                reconstr_loss = tf.scalar_mul(-1, self.x_target * tf.log(1e-10 + x_reconstr_means[i])
                                    + (1-x_target) * tf.log(1e-10 + 1 - x_reconstr_means[i]))
                self.reconstr_losses.append(reconstr_loss)
        else:
            # Multivariate gaussian negative log-likelihood with diagonal covariance matrix
            # Note that when x_reconstr_sigma < 1/sqrt(2 * pi) # 0.4, the log could compensate other errors
            # Since x_reconstr_sigma is only used to compute the following reconstruction loss term with respect to our Gaussian assummptions
            # We could arrange our hypothesis to make it disappear, saving gradient computations, correcting the error compensation
            # and removing a degree of freedom that could be used by the network to alleviate the reconstruction error otherwise

            if self.vae_architecture["use_reconstr_sigma"]:
                n_input_log_2pi = tf.scalar_mul(
                    tf.constant(self.vae_architecture["n_input"], dtype=tf.float32),
                    tf.log(2. * np.pi)
                )
                mul_and_avg = tf.constant(0.5 / nb_samples, dtype=tf.float32)

                for i in range(len(x_reconstr_means)):
                    self.reconstr_loss_divider = tf.exp(x_reconstr_log_sigma_sqs[i], name="reconstr_loss_divider")
                    self.reconstr_loss_sq_sub = tf.square(tf.subtract(self.x_target, x_reconstr_means[i]), name="reconstr_loss_sq_sub")
                    self.reconstr_loss_division = tf.divide(self.reconstr_loss_sq_sub, self.reconstr_loss_divider, name="reconstr_loss_division")
                    # reconstr_losses shape = [batch_size]
                    reconstr_loss = tf.add(
                        tf.reduce_sum(self.reconstr_loss_division, 1),
                        tf.add(
                            tf.reduce_sum(x_reconstr_log_sigma_sqs[i], 1),
                            n_input_log_2pi
                        )
                    )
                    # reconstr_losses shape = [L, batch_size]
                    self.reconstr_losses.append(reconstr_loss)
            else:
                mul_and_avg = tf.constant(np.pi / nb_samples, dtype=tf.float32)

                for i in range(len(x_reconstr_means)):
                    self.reconstr_loss_sq_sub = tf.square(tf.subtract(self.x_target, x_reconstr_means[i]), name="reconstr_loss_sq_sub")
                    # reconstr_losses shape = [batch_size]
                    reconstr_loss = tf.reduce_sum(self.reconstr_loss_sq_sub, 1)
                    # reconstr_losses shape = [L, batch_size]
                    self.reconstr_losses.append(reconstr_loss)

        # average reconstruction loss on L samples
        reconstr_loss = tf.add_n(
            self.reconstr_losses
        )

        # reconstr_loss shape = [batch_size]
        reconstr_loss = tf.scalar_mul(
            mul_and_avg,
            reconstr_loss
        )
        return reconstr_loss


    def create_latent_loss(self):
        # Kullback-Leibler divergence between two multivariate gaussians with diagonal covariance matrices
        # and identity variance for the second dstribution

        n_z = tf.constant(self.vae_architecture["n_z"], dtype=tf.float32)

        # latent_loss shape = [batch_size]
        latent_loss = tf.scalar_mul(
            tf.constant(0.5, dtype=tf.float32),
            tf.subtract(
                tf.add(
                    tf.reduce_sum(tf.square(self.z_mean), 1),
                    tf.reduce_sum(tf.exp(self.z_log_sigma_sq), 1)
                ),
                tf.add(
                    tf.reduce_sum(self.z_log_sigma_sq, 1),
                    n_z
                )
            )
        )
        return latent_loss


    def create_loss_function(self):
        """ The loss is composed of two terms:
        1.) log p(x|z) => reconstruction loss
          - log-likelihood of x given z, sampled L times then averaged (Monte Carlo)
              * in binary case: cross-entropy
              * in continuous case: log-likelihood of seeing the target x under the
                Gaussian distribution parameterized by x_reconstr_mean, sigma = sqrt(exp(x_reconstr_log_sigma_sq))

        2.) The latent loss, which is defined as the Kullback Leibler divergence
            between the system noise distribution induced by the encoder on
            the data and a unit gaussian prior. This acts as a kind of regularizer.

        N.B.: The variational lower bound represented by these two previously defined losses must be maximized. But, as tensorflow
        minimizes the network cost function, we mutiply that lower bound by -1 and minimize it.
        """
        self.reconstr_loss = self.create_reconstruction_loss(self.x_reconstr_means, self.x_reconstr_log_sigma_sqs, len(self.x_reconstr_means))

        if self.apply_latent_loss:
            self.latent_loss = self.create_latent_loss()
        else:
            self.latent_loss = tf.constant(0, dtype=tf.float32)

        self.cost_add = tf.add(self.reconstr_loss, self.latent_loss)

        if self.z_continuity_error_coeff != None:
            self.add_z_correction_cost()
            if self.estimate_z_derivative:
                self.add_z_derivative_correction_cost()

        # batch average and variance
        self.cost, self.variance = tf.nn.moments(self.cost_add, [0])
        # self.cost = tf.reduce_mean(self.cost_add)   # average over batch


    def add_z_correction_cost(self):
        self.z_target = tf.placeholder(tf.float32, [None, self.vae_architecture["n_z"]], name="z_target")
        self.z_correction = tf.square(
            tf.subtract(
                self.z,
                self.z_target
            )
        )
        self.reduced_z_correction = tf.scalar_mul(
            tf.constant(self.z_continuity_error_coeff, dtype=tf.float32),
            tf.reduce_sum(
                self.z_correction,
                1
            )
        )
        self.cost_add = tf.add_n([
            self.cost_add,
            self.reduced_z_correction
        ])


    def add_z_derivative_correction_cost(self):
        self.z_derivative_target = tf.placeholder(tf.float32, [None, self.vae_architecture["n_z"]], name="z_derivative_target")
        self.z_derivative_correction = tf.square(
            tf.subtract(
                self.z_derivative,
                self.z_derivative_target
            )
        )
        self.reduced_z_derivative_correction = tf.scalar_mul(
            tf.constant(self.z_continuity_error_coeff, dtype=tf.float32),
            tf.reduce_sum(
                self.z_derivative_correction,
                1
            )
        )
        self.cost_add = tf.add_n([
            self.cost_add,
            self.reduced_z_derivative_correction
        ])


    def create_loss_optimizer(self):
        if self.optimize == True:
            # Use ADAM optimizer
            adam_id = str(np.random.rand())
            with tf.variable_scope(adam_id):
                self.optimizer = tf.train.AdamOptimizer(name="Adam", learning_rate=self.learning_rate).minimize(self.cost)
                print(self.optimizer.name)


    def partial_fit(self, X, sess):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        #m = tf.Print(tf.shape(self.x)[0], [tf.shape(self.x)[0]])

        opt, cost = sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})

        #m_eval = m.eval(session=self.sess,feed_dict={self.x: X})

        return cost


    def transform(self, X, sess):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution

        z_mean = sess.run(self.z_mean, feed_dict={self.x: X})

        return z_mean


    def generate(self, z_mu=None, sess=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.vae_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        x_reconstr_mean = sess.run(self.x_reconstr,
                             feed_dict={self.z: z_mu})
        return x_reconstr_mean


    def reconstruct(self, X, sess):
        """ Use VAE to reconstruct given data. """
        x_reconstr_mean = sess.run(self.x_reconstr,
                             feed_dict={self.x: X})
        return x_reconstr_mean


    def init_session(self, sess):
        if self.save_path == None:
            # Initializing the tensor flow variables
            sess.run(tf.global_variables_initializer())
        else:
            # Restore variables from disk
            tf.train.Saver().restore(sess, self.save_path)
            print("Model restored.")


    def save_session(self, sess):
        # Save the variables to disk.
        if self.save_path == None:
            self.save_path = tf.train.Saver().save(sess, "./tmp/model-vae.ckpt")
        else:
            tf.train.Saver().save(sess, self.save_path)

        print("Model saved in file: %s" % self.save_path)


    def train(self, p):

        n_samples = len(p["data"])

        with tf.Session() as sess:

            self.init_session(sess)

            # Training cycle

            begin = datetime.now()
            print("\n------- Training begining: {} -------\n".format(begin.isoformat()[11:]))
            print("Sample size = "+str(n_samples))
            print("Batch size = "+str(p["batch_size"])+"\n")

            for epoch in range(p["nb_epochs"]):
                avg_cost = 0.
                total_batch = int(n_samples / p["batch_size"])

                if n_samples % p["batch_size"] != 0:
                    total_batch += 1

                w = 0.
                batches = 0
                # Loop over all batches
                for i in range(total_batch):
                    batch = p["data"][p["batch_size"]*i : p["batch_size"]*(i+1)]

                    # Fit training using batch data
                    cost = self.partial_fit(batch, sess)
                    # Compute average loss
                    batches += len(batch)
                    w = len(batch)/batches
                    avg_cost = avg_cost*(1. - w) + w*cost

                # Display logs per epoch step
                if epoch % p["display_step"] == 0:
                    print("Epoch:", '%04d' % (epoch),
                          " ----------> Average cost =", "{:.9f}".format(avg_cost))

            end = datetime.now()
            print("\n------- Training end: {} -------\n".format(end.isoformat()[11:]))
            print("Elapsed = "+str((end-begin).total_seconds())+" seconds\n")

            self.save_session(sess)




    """
        ---------------- Visual demo with a 2D VAE
    """
    def demo_2d(self):

        # Load MNIST data in a format suited for tensorflow.
        # The script input_data is available under this URL:
        # https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
        import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        TRAINING = {
            "data": mnist.train.images,
            "batch_size": 100,
            "nb_epochs": 10,
            "display_step": 2
        }

        vae_architecture_2d = \
            dict(n_hidden_encoder_1=500, # 1st layer encoder neurons
                 n_hidden_encoder_2=500, # 2nd layer encoder neurons
                 n_hidden_decoder_1=500, # 1st layer decoder neurons
                 n_hidden_decoder_2=500, # 2nd layer decoder neurons
                 n_input=784, # MNIST data input (img shape: 28*28)
                 n_z=2,       # dimensionality of latent space
                 n_output=784  # MNIST data input (img shape: 28*28)
            )

        vae_2d = VAE(vae_architecture_2d)
        vae_2d.train(TRAINING)

        """
            Latent space representation
        """
        x_sample, y_sample = mnist.test.next_batch(5000)
        z_mu = vae_2d.transform(x_sample)
        plt.figure(figsize=(8, 6))
        plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
        plt.colorbar()
        plt.grid()
        plt.show()


        """
            Latent space canvas filled with observation space equivalent
        """
        nx = ny = 20
        x_values = np.linspace(-3, 3, nx)
        y_values = np.linspace(-3, 3, ny)

        canvas = np.empty((28*ny, 28*nx))
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                z_mu = np.array([[xi, yi]]*TRAINING["batch_size"])
                x_mean = vae_2d.generate(z_mu)
                canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

        plt.figure(figsize=(8, 10))
        Xi, Yi = np.meshgrid(x_values, y_values)
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.tight_layout()
        plt.show()

        return vae_2d
