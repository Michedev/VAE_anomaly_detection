# ========== Legacy code ===============


from math import ceil

import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal


def tf_namespace(namespace):
    def wrapper(f):
        def wrapped_f(*args, **kwargs):
            with tf.name_scope(namespace):
                return f(*args, **kwargs)

        return wrapped_f

    return wrapper


class VAE:

    def __init__(self, input_shape, encode_sizes, latent_size, decode_sizes=None, mu_prior=None, sigma_prior=None,
                 lr=10e-4,  momentum=0.9, save_model=True):
        self.encode_sizes = encode_sizes
        self.latent_size = latent_size
        self.decode_sizes = decode_sizes or encode_sizes[::-1]
        self.mu_prior = mu_prior or np.zeros([latent_size], dtype='float32')
        self.sigma_prior = sigma_prior or np.ones([latent_size], 'float32')
        self.lr = lr
        self.momentum = momentum
        self.input_shape = input_shape
        self.save_model = save_model
        self._build_graph(input_shape, latent_size)

    def _build_graph(self, input_shape, latent_size):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._create_placeholders(input_shape)
            self._create_encoder(self.X)
            self._create_latent_distribution(self.encoder, latent_size)
            self._create_decoder(self.z)
            self.loss = - self.elbo(self.X, self.decoder, self.mu, self.log_sigma_square, self.sigma_square,
                                    tf.constant(self.mu_prior), tf.constant(self.sigma_prior))
            self.opt = tf.train.AdamOptimizer(self.lr, self.momentum)
            self.opt_op = self.opt.minimize(self.loss)
            self.session = tf.InteractiveSession(graph=self.graph)
        writer = tf.summary.FileWriter(logdir='logdir', graph=self.graph)
        writer.flush()

    @property
    def k_init(self):
        return {'kernel_initializer': tf.glorot_uniform_initializer()}

    def elbo(self, X_true, X_pred, mu, log_sigma, sigma, mu_prior, sigma_prior):
        epsilon = tf.constant(0.000001)
        self.mae = tf.losses.absolute_difference(X_true, X_pred, reduction=tf.losses.Reduction.NONE)
        self.mae_sum = tf.reduce_sum(self.mae, axis=1)
        log_sigma_prior = tf.log(sigma_prior + epsilon)
        mu_diff = mu - mu_prior
        self.kl = log_sigma_prior - log_sigma - 1 + (sigma + tf.multiply(mu_diff, mu_diff)) / sigma_prior
        self.kl_sum = tf.reduce_sum(self.kl, axis=1)
        return tf.reduce_mean(- self.mae_sum - self.kl_sum)

    @tf_namespace('placeholders')
    def _create_placeholders(self, input_shape):
        self.X = tf.placeholder(tf.float32, shape=[None, *input_shape], name='X')

    @tf_namespace('encoder')
    def _create_encoder(self, X):
        self.encode_layers = []
        self.encoder = X
        for i, lsize in enumerate(self.encode_sizes):
            self.encoder = tf.layers.dense(self.encoder, lsize, **self.k_init,
                                           activation=tf.nn.relu, name=f'encoder_{i + 1}')
            self.encode_layers.append(self.encoder)
            setattr(self, f'encoder_{i + 1}', self.encoder)

    @tf_namespace('latent')
    def _create_latent_distribution(self, encoder, latent_dim):
        self.mu = tf.layers.dense(encoder, latent_dim, **self.k_init, name='mu')
        self.log_sigma_square = tf.layers.dense(encoder, latent_dim,
                                                **self.k_init, name='log_sigma_square')
        self.sigma_square = tf.exp(self.log_sigma_square, 'sigma_square')
        self.z = tf.add(self.mu, self.sigma_square * tf.random.normal(tf.shape(self.sigma_square)), 'z')

    @tf_namespace('decoder')
    def _create_decoder(self, z):
        self.decoder = z
        self.decode_layers = []
        for i, lsize in enumerate(self.decode_sizes):
            self.decoder = tf.layers.dense(self.decoder, lsize, **self.k_init,
                                           activation=tf.nn.relu, name=f'decoder_{i + 1}')
            setattr(self, f'decoder_{i + 1}', self.decoder)
            self.decode_layers.append(self.decoder)
            if i == len(self.decode_sizes) - 1:
                self.mu_post = tf.layers.dense(self.decoder, self.input_shape[0], name='mu_posterior')
                self.log_sigma_post = tf.layers.dense(self.decoder, self.input_shape[0])
                self.sigma_post = tf.exp(self.log_sigma_post, 'sigma_square_posterior')
                self.decoder = tf.add(self.mu_post,
                                      self.sigma_post * tf.random.normal((self.input_shape[0],), name='eps_post'),
                                      name='decoder_output')
                setattr(self, f'decoder_{i + 2}', self.decoder)
                self.decode_layers.append(self.decoder)
        return self.decoder

    @property
    def layers(self):
        return [(f'encoder_{i}', getattr(self, f'encoder_{i}')) for i in range(1, len(self.encode_layers) + 1)] + \
               [('mu', self.mu), ('sigma', self.log_sigma_square), ('z', self.z)] + \
               [(f'decoder_{i}', getattr(self, f'decoder_{i}')) for i in range(1, len(self.decode_layers) + 1)]

    def fit(self, X, epochs, batch_size, print_every=50, save_every_epochs=5, verbose=True):
        n_batch = ceil(X.shape[0] / batch_size)
        if self.save_model:
            saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            np.random.shuffle(X)
            acc_loss = 0
            counter = 0
            for i in range(n_batch):
                slice_batch = slice(i * batch_size, (i + 1) * batch_size) if i != n_batch - 1 else slice(
                    i * batch_size,
                    None)
                X_batch = X[slice_batch, :]
                batch_loss, _ = self.session.run([self.loss, self.opt_op], {self.X: X_batch})
                acc_loss += batch_loss
                if verbose and counter % print_every == 0:
                    print(f" Epoch {epoch} - batch {i} - neg_ELBO = {batch_loss}")
                counter += 1
            if verbose:
                print(f'\nEpoch {epoch} - Avg loss = {acc_loss / n_batch}')
                print('\n' + ('-' * 70))
            if self.save_model and (epoch+1) % save_every_epochs == 0:
                saver.save(self.session, "ckpts/ad_vae.ckpt")

    def generate(self, n=1, mu_prior=None, sigma_prior=None):
        """
        Generate new examples sampling from the latent distribution
        :param n: number of examples to generate
        :param mu_prior:
        :param sigma_prior:
        :return: a matrix of size [n, p] where p is the number of variables of X_train
        """
        if mu_prior is None:
            mu_prior = self.mu_prior
        if sigma_prior is None:
            sigma_prior = self.sigma_prior
        z = np.random.multivariate_normal(mu_prior, np.diag(sigma_prior), [n])
        return self.session.run(self.decoder, feed_dict={self.z: z})

    def reconstruct(self, X):
        return self.session.run(self.decoder, feed_dict={self.X: X})

    def reconstructed_probability(self, X, L=100):
        reconstructed_prob = np.zeros((X.shape[0],), dtype='float32')
        mu_hat, sigma_hat = self.session.run([self.mu_post, self.sigma_post], {self.X: X})
        for l in range(L):
            mu_hat = mu_hat.reshape(X.shape)
            sigma_hat = sigma_hat.reshape(X.shape) + 0.00001
            for i in range(X.shape[0]):
                p_l = multivariate_normal.pdf(X[i, :], mu_hat[i, :], np.diag(sigma_hat[i, :]))
                reconstructed_prob[i] += p_l
        reconstructed_prob /= L
        return reconstructed_prob

    def is_outlier(self, X, L=100, alpha=0.05):
        p_hat = self.reconstructed_probability(X, L)
        return p_hat < alpha

    def open(self):
        if not hasattr(self, 'session') or self.session is None:
            if self.graph is None:
                self._build_graph(self.input_shape, self.latent_size)
            else:
                self.session = tf.InteractiveSession(graph=self.graph)

    def close(self):
        if hasattr(VAE, 'session') and VAE.session is not None:
            VAE.session.close()
            VAE.session = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __delete__(self, instance):
        self.close()

    def __setattr__(self, key, value):
        if key == 'session':
            if hasattr(self, 'session') and self.session is not None:
                self.close()
            VAE.session = value
        else:
            self.__dict__[key] = value

    def __delattr__(self, item):
        if item == 'session':
            self.close()
            del VAE.__dict__['session']
        else:
            del self.__dict__[item]

    def __enter__(self):
        self.open()