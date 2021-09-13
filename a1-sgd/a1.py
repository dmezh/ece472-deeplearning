#!/bin/env/python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SAMPLE_COUNT = 50

LEARNING_RATE = 0.9
ITER_COUNT = 2000
BATCH_SIZE = 15

X_START = -np.pi * (0.5 * 2.5)
X_END = np.pi * (0.5 * 2.5)
X_OFFSET = np.pi/6

NOISE_SIGMA = 0.1

class GaussianModel(tf.Module):
    # Four trainable parameters
    def __init__(self, num_gausses):
        self.w = tf.Variable(tf.constant(shape=[num_gausses], value=[-1.0, 0, 1.0]))
        self.mu = tf.Variable(tf.constant(shape=[num_gausses], value=[-2.0, 0.0, 2.0]))
        self.s = tf.Variable(tf.random.normal(shape=[num_gausses]))
        self.b = tf.Variable(tf.zeros(shape=[1]))

    # Gaussian basis functions
    def __call__(self, x):
        return tf.squeeze(tf.math.exp(- tf.math.square(tf.subtract(x, self.mu[0])) * tf.math.square(self.s[0])) * self.w[0]) + \
               tf.squeeze(tf.math.exp(- tf.math.square(tf.subtract(x, self.mu[1])) * tf.math.square(self.s[1])) * self.w[1]) + \
               tf.squeeze(tf.math.exp(- tf.math.square(tf.subtract(x, self.mu[2])) * tf.math.square(self.s[2])) * self.w[2]) + \
                   self.b

def main():

    np.random.seed(98765)

    learnrate = 0.1
    
    fuzzy_x = np.random.uniform(X_START, X_END, size=(SAMPLE_COUNT, 1))
    y_truth = np.sin(fuzzy_x - X_OFFSET)
    y_noisy = y_truth + NOISE_SIGMA * np.random.normal(size=(SAMPLE_COUNT, 1))

    params = GaussianModel(3)

    sample_locations = np.arange(SAMPLE_COUNT)

    optimizer = tf.optimizers.SGD(learnrate)

    for i in range(ITER_COUNT):
        # Perform the stochastic selection and loss calculation with a GradientTape recording
        # so we take advantage of autodiff.
        with tf.GradientTape() as tape:
            sample_group = np.random.choice(sample_locations, size=BATCH_SIZE)

            x_run = fuzzy_x[sample_group]
            y_run = y_noisy[sample_group].flatten()

            y_run_estimated = params(x_run)

            loss = 0.5 * tf.reduce_mean((y_run - y_run_estimated) ** 2)

        # Calculate gradient with autodiff using our the tape from this run.
        grads = tape.gradient(loss, params.trainable_variables)

        # # Apply SGD.
        # for j, var in enumerate(params.trainable_variables):
        #     var.assign(var - (grads[j] * learnrate)) # Bottou pg. 2
        optimizer.apply_gradients(zip(grads, params.trainable_variables))
        print(f"loss = {loss:0.5f}, iteration = {i}, learning rate = {learnrate}")

        #learnrate /= 1.002

    # Print the final parameter values.
    w_hat = np.squeeze(params.w.numpy())
    print("w_hat")
    print(w_hat)

    mu_hat = np.squeeze(params.mu.numpy())
    print("\nmu_hat")
    print(mu_hat)

    s_hat = np.squeeze(params.s.numpy())
    print("\ns_hat")
    print(s_hat)

    b_hat = np.squeeze(params.b.numpy())
    print("\nb_hat")
    print(b_hat)

    # Plot results.
    fig, axes = plt.subplots(1, 2)

    axes[0].set_title("Gaussian Fit to Noisy Sinewave")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_ylim(-2, 2)

    xs = np.linspace(X_START, X_END, SAMPLE_COUNT)
    xs = xs[:, np.newaxis]
    
    axes[0].plot(xs, np.squeeze(params(xs)), "--", np.squeeze(fuzzy_x), y_noisy, "o")
    axes[0].plot(xs, np.sin(xs - X_OFFSET), "-")

    axes[1].set_title("Trained Basis Functions")
    axes[1].set_ylim(-2, 2)
    axes[1].plot(xs, np.squeeze(tf.squeeze(tf.math.exp(- tf.math.square(tf.subtract(xs, params.mu[0])) / tf.math.square(params.s[0])) * params.w[0])))
    axes[1].plot(xs, np.squeeze(tf.squeeze(tf.math.exp(- tf.math.square(tf.subtract(xs, params.mu[1])) / tf.math.square(params.s[1])) * params.w[1])))
    axes[1].plot(xs, np.squeeze(tf.squeeze(tf.math.exp(- tf.math.square(tf.subtract(xs, params.mu[2])) / tf.math.square(params.s[2])) * params.w[2])))
    
    plt.tight_layout()
    plt.savefig("fit-2.pdf")


if __name__ == "__main__":
    main()
