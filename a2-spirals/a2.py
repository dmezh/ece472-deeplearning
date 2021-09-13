#!/bin/env/python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import pickle

THETA_START = 0
THETA_END = np.pi * 3.5
NUM_SAMPLES = 300
SPIRAL_OFFSET = 1
SPIRAL_RADIUS = 1

NOISE_FACTOR = 0.2

ITER_COUNT = 6500
BATCH_SIZE = 32

USE_PICKLED = True

class Data:
    def __init__(self):
        self.x = []
        self.y = []
        self.type = []

class MLP(tf.Module):
    def __init__(self, layers):
        self.params = {}
        self.params["w"] = []
        self.params["b"] = []

        for i, _ in enumerate(layers[1:], start=1):
            self.params["w"].append(tf.Variable(tf.random.normal(shape=[layers[i], layers[i-1]]))) # weights
            self.params["b"].append(tf.Variable(tf.random.normal(shape=[layers[i], 1]))) # biases

    def apply_layer(self, index, input):
        result = (self.params["w"][index] @ input) + self.params["b"][index]
        return tf.nn.sigmoid(result)
    
    def predict(self, input):
        for i, _ in enumerate(self.params["w"]):
            input = self.apply_layer(i, input)
        return input

def noise(n):
    return NOISE_FACTOR * np.random.normal(size=(n, 1))

def main():
    np.random.seed(98765)

    model = MLP([2, 64, 32, 16, 16, 1])

    # two random sets for each next time?
    theta_spiral = np.random.uniform(THETA_START, THETA_END, size=(NUM_SAMPLES, 1))
    r_spiral = SPIRAL_OFFSET + (theta_spiral * SPIRAL_RADIUS)

    x_spiral_a = - (r_spiral * np.cos(theta_spiral))
    y_spiral_a = r_spiral * np.sin(theta_spiral)

    x_spiral_b = -x_spiral_a + noise(NUM_SAMPLES)
    y_spiral_b = -y_spiral_a + noise(NUM_SAMPLES)

    x_spiral_a = x_spiral_a + noise(NUM_SAMPLES)
    y_spiral_a = y_spiral_a + noise(NUM_SAMPLES)

    data = Data()
    data.x = np.concatenate((x_spiral_a, x_spiral_b))
    data.y = np.concatenate((y_spiral_a, y_spiral_b))
    data.type = np.concatenate((np.zeros(NUM_SAMPLES), np.ones(NUM_SAMPLES)))

### train
    sample_locations = np.arange(NUM_SAMPLES*2)

    optimizer = tf.optimizers.Adam()

    input = [np.array(list(zip(data.x, data.y))), np.array(data.type)]

    #historical_loss = []

    if not USE_PICKLED:
        for i in range(ITER_COUNT):
            with tf.GradientTape() as tape:
                sample_group =  np.random.choice(sample_locations, size=BATCH_SIZE)

                data_run = input[0][sample_group]
                truth_run = input[1][sample_group]

                #print(sample_group)

                y_run_estimated = []
                loss = []
                for j, e in enumerate(data_run):
                    y_hat = model.predict(e)
                    y_run_estimated.append(y_hat)
                    loss.append(tf.reduce_mean((-truth_run[j] * tf.math.log(y_hat)) - ((1-truth_run[j]) * tf.math.log(1 - y_hat))))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print(f"iter: {i}, loss: {np.mean(loss)}")

        # save weights
        weightsfile = open("weights.ece472", 'wb')
        pickle.dump(model, weightsfile)
    else:
        weightsfile = open("weights.ece472", 'rb')
        model = pickle.load(weightsfile)

    # Plot results
    fig, axes = plt.subplots(1, 1, figsize=(4,4), dpi=200)

    axes.set_title("Spirals Classification")
    axes.set_xlabel("x")
    axes.set_ylabel("y")

    for i in range(NUM_SAMPLES*2):
        run = []
        run.append(data.x[i])
        run.append(data.y[i])
        predicted_type = model.predict(run)
        color = "red" if (predicted_type < 0.5) else "blue"
        axes.plot(data.x[i], data.y[i], marker="o", markersize=3, markeredgecolor=(0,0,0,1), color=color)
    #axes[1].plot(historical_loss)
    
    BOUNDARY_REGION_SIZE = 500

    region_x = np.linspace(-15, 15, BOUNDARY_REGION_SIZE)
    region_y = np.linspace(-15, 15, BOUNDARY_REGION_SIZE)

    # Plot boundary points (points, not line, unfortunately)
    for x in range(1, BOUNDARY_REGION_SIZE):
        for y in range (1, BOUNDARY_REGION_SIZE):
            if (model.predict(np.matrix([region_x[x], region_y[y]]).T) < 0.5) ^ (model.predict(np.matrix([region_x[x], region_y[y-1]]).T) < 0.5):
                axes.plot(region_x[x], region_y[y], marker="o", markersize=1, color="purple")
    
    plt.tight_layout()
    plt.savefig("fit-2.pdf")

if __name__ == "__main__":
    main()
