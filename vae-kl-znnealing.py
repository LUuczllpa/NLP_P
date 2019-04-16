# @title Imports (Do not modify!)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

sns.set(rc={"lines.linewidth": 2.8}, font_scale=2)
sns.set_style("whitegrid")

import tensorflow_probability as tfp

tfd = tfp.distributions

import warnings

warnings.filterwarnings('ignore')


def gallery(array, ncols=10, rescale=False):
    """Data visualization code."""
    if rescale:
        array = (array + 1.) / 2
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


def show_digits(axis, digits, title=''):
    axis.axis('off')
    ncols = int(np.sqrt(digits.shape[0]))
    axis.imshow(gallery(digits, ncols=ncols).squeeze(axis=2),
                cmap='gray')
    axis.set_title(title, fontsize=15)


BATCH_SIZE = 64
NUM_LATENTS = 10
TRAINING_STEPS = 10000


def show_latent_interpolations(generator, prior, sess):
    a = np.linspace(0.0, 1.0, BATCH_SIZE)
    a = np.expand_dims(a, axis=1)

    first_latents = prior.sample()[0]
    second_latents = prior.sample()[0]

    # To ensure that the interpolation is still likely under the Gaussian prior,
    # we use Gaussian interpolation - rather than linear interpolation.
    interpolations = np.sqrt(a) * first_latents + np.sqrt(1 - a) * second_latents

    ncols = int(np.sqrt(BATCH_SIZE))
    samples_from_interpolations = generator(interpolations)
    samples_from_interpolations_np = sess.run(samples_from_interpolations)
    plt.gray()
    axis = plt.gca()
    show_digits(
        axis, samples_from_interpolations_np, title='Latent space interpolations')


tf.reset_default_graph()
tf.set_random_seed(2019)

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

print(mnist.train.images.shape)
print(type(mnist.train.images))


def make_tf_data_batch(np_data, shuffle=True):
    # Reshape the data to image size.
    images = np_data.reshape((-1, 28, 28, 1))

    # Create the TF dataset.
    dataset = tf.data.Dataset.from_tensor_slices(images)

    # Shuffle and repeat the dataset for training.
    # This is required because we want to do multiple passes through the entire
    # dataset when training.
    if shuffle:
        dataset = dataset.shuffle(100000).repeat()

    # Batch the data and return the data batch.
    one_shot_iterator = dataset.batch(BATCH_SIZE).make_one_shot_iterator()
    data_batch = one_shot_iterator.get_next()
    return data_batch


real_data = make_tf_data_batch(mnist.train.images)
print(real_data.shape)

data_var = tf.Variable(
    tf.ones(shape=(BATCH_SIZE, 28, 28, 1), dtype=tf.float32),
    trainable=False)

data_assign_op = tf.assign(data_var, real_data)


def standard_decoder(z):
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        h = tf.layers.dense(z, 7 * 7 * 64, activation=tf.nn.relu)
        h = tf.reshape(h, shape=[BATCH_SIZE, 7, 7, 64])
        h = tf.layers.Conv2DTranspose(
            filters=32,
            kernel_size=5,
            strides=2,
            activation=tf.nn.relu,
            padding='same')(h)
        h = tf.layers.Conv2DTranspose(
            filters=1,
            kernel_size=5,
            strides=2,
            activation=None,  # Do not activate the last layer.
            padding='same')(h)
        return tfd.Independent(tf.distributions.Bernoulli(h))


def multi_normal(loc, log_scale):
    # We model the latent variables as independent
    return tfd.Independent(
        distribution=tfd.Normal(loc=loc, scale=tf.exp(log_scale)),
        reinterpreted_batch_ndims=1)


def make_prior():
    # Zero mean, unit variance prior.
    prior_mean = tf.zeros(shape=(BATCH_SIZE, NUM_LATENTS), dtype=tf.float32)
    prior_log_scale = tf.zeros(shape=(BATCH_SIZE, NUM_LATENTS), dtype=tf.float32)

    return multi_normal(prior_mean, prior_log_scale)


# Build the variational posterior


def get_variational_posterior(x=None):
    """x:[?, 28, 28, 1]"""
    with tf.variable_scope("variational", reuse=tf.AUTO_REUSE):
        means = tf.get_variable("means_z", shape=[BATCH_SIZE, NUM_LATENTS])
        log_scales = tf.get_variable("log_scales_z", shape=[BATCH_SIZE, NUM_LATENTS])

    return multi_normal(means, log_scales)


def encoder(x):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        h = tf.layers.Conv2D(
            filters=8,
            kernel_size=5,
            strides=2,
            activation=tf.nn.relu,
            padding='same')(x)
        h = tf.layers.Conv2D(
            filters=16,
            kernel_size=5,
            strides=2,
            activation=tf.nn.relu,
            padding='same')(h)
        h = tf.layers.Conv2D(
            filters=32,
            kernel_size=5,
            strides=1,
            activation=tf.nn.relu,
            padding='same')(h)

        out_shape = 1
        for s in h.shape.as_list()[1:]:
            out_shape *= s

        h = tf.reshape(h, shape=[BATCH_SIZE, out_shape])
        mean = tf.layers.dense(h, NUM_LATENTS, activation=None)
        scale = tf.layers.dense(h, NUM_LATENTS, activation=None)
        return multi_normal(loc=mean, log_scale=scale), mean, scale


def reparameterization(means, log_scles):
    gaussian = make_prior()
    epsilon = gaussian.sample()
    z = epsilon * tf.exp(log_scles) + means
    return z


def kl_qp(means, log_scales):
    """means: [BatchSize, NumLatent],
    log_scales: [BatchSize, NumLatent]
    """
    kl_term = -0.5 * (1 + log_scales - tf.pow(means, 2) - tf.pow(tf.exp(log_scales), 2))
    kl_term = tf.reduce_mean(kl_term, axis=1)
    return kl_term


def bound_terms(data_batch, variational_posterior_fn, decoder_fn):
    ##################
    # decoder likelihood #
    ##################
    variational_posterior, means, log_scales = variational_posterior_fn(data_batch)
    z = reparameterization(means, log_scales)
    decoder_likelihood = decoder_fn(z)

    tf.add_to_collection("reconstructions", decoder_likelihood.sample())

    likelihood_term = decoder_likelihood.log_prob(data_batch)

    # Reduce mean over the batch dimensions
    likelihood_term = tf.reduce_mean(likelihood_term)

    # Reduce over the batch dimension.
    kl_term = tf.reduce_mean(kl_qp(means, log_scales))

    # Return the terms in the optimization objective in (1.1) description
    return likelihood_term, kl_term, z, variational_posterior


def generative():
    prior = make_prior()
    z = prior.sample()
    with tf.name_scope("inference"):
        dist = standard_decoder(z)
        generated_img = dist.sample()
    return generated_img


if __name__ == '__main__':
    # Maximize the data likelihodd and minimize the KL divergence between the prior and posterior

    likelihood_term, kl_term, z, variational_posterior = \
        bound_terms(data_var, encoder, standard_decoder)

    counter = tf.Variable(initial_value=0., trainable=False, name="alpha", dtype=tf.float32)
    update_counter = tf.assign_add(counter, 1.)
    with tf.control_dependencies(control_inputs=[update_counter]):
        alpha = tf.identity(counter) / TRAINING_STEPS

    train_elbo = likelihood_term - alpha*kl_term

    ##################
    # YOUR CODE HERE #
    ##################
    loss = tf.negative(train_elbo)
    optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.9)
    update_op = optimizer.minimize(loss, var_list=tf.trainable_variables())

    generated_img = generative()
    tf.trainable_variables()

    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        # %hide_pyerr  # - uncomment to interrupt training without a stacktrace
        losses = []
        kls = []
        likelihood_terms = []

        for i in range(TRAINING_STEPS):

            # Update the data batch.
            sess.run(data_assign_op)
            alpha = (i+1) / TRAINING_STEPS
            sess.run(update_op)

            # Report the loss and the kl once in a while.
            if i % 100 == 0:
                iteration_loss, iteration_kl, iteration_likelihood = sess.run(
                    [loss, kl_term, likelihood_term])
                print('Iteration {}. Loss {}. KL {}'.format(
                    i, iteration_loss, iteration_kl))
                losses.append(iteration_loss)
                kls.append(iteration_kl)
                likelihood_terms.append(iteration_likelihood)
        real_data_examples = sess.run(data_var)
        data_reconstructions, = sess.run(tf.get_collection("reconstructions"))
        final_samples = sess.run(generated_img)
        show_latent_interpolations(lambda x: standard_decoder(x).mean(), make_prior(), sess)

    fig, axes = plt.subplots(1, 3, figsize=(3 * 8, 5))

    axes[0].plot(losses, label='Negative ELBO')
    axes[0].set_title('Time', fontsize=15)
    axes[0].legend()

    axes[1].plot(kls, label='KL')
    axes[1].set_title('Time', fontsize=15)
    axes[1].legend()

    axes[2].plot(likelihood_terms, label='Likelihood Term')
    axes[2].set_title('Time', fontsize=15)
    axes[2].legend()

    # Read data (just sample from the data set)
    # real_data_examples

    # Sample from the generative model!

    fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 4))

    show_digits(axes[0], real_data_examples, 'Data')
    show_digits(axes[1], data_reconstructions, 'Reconstructions')
    show_digits(axes[2], final_samples, 'Samples')
    plt.show()
    pass
