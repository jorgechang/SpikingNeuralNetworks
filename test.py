"""Test Artificial Neural Network Model conversion to Spiking Neural Network"""
# Standar modules
import argparse

# Third party modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist

# Local modules
from models.net import SimpleCNN, SpikingCNN


# Define and parse input arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MNIST", help="MNIST or CIFAR")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--epochs", type=int, default=8, help="epochs")
parser.add_argument("--bsz", type=int, default=64, help="batch size")
parser.add_argument("--activation", type=str, default=None, help="thReLU or ReLU")
parser.add_argument("--maxPooling", type=int, default=0, help="maxPooling")
parser.add_argument("--softmax", type=int, default=0, help="softmax")
args = parser.parse_args()

# Training hyperparameters
learning_rate = args.lr
epochs = args.epochs
batch_size = args.bsz

# Simulation parameters for Spiking Neural Network
threshold = 5
simulation_len = 64

# Dataset building and preprocessing
if args.dataset == "MNIST":
    num_classes = 10  # total classes (0-9 digits).
    input_shape = (1, 28, 28, 1)  # image shape for MNIST.
    # Prepare MNIST data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Convert to float32.
    x_train = np.array(x_train, np.float32)
    x_test = np.array(x_test, np.float32)
    # Normalize images value from [0, 255] to [0, 1].
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Expand dimensions to match network input and match torch dataloader.
    x_train, x_test = np.expand_dims(x_train, 3), np.expand_dims(x_test, 3)
else:
    num_classes = 10  # total classes.
    input_shape = (32, 32, 3)  # image shape for CIFAR10.
    # Prepare CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Convert to float32.
    x_train = np.array(x_train, np.float32)
    x_test = np.array(x_test, np.float32)
    # Normalize images value from [0, 255] to [0, 1].
    x_train, x_test = x_train / 255.0, x_test / 255.0

# Choose checkpoint path depending on dataset
checkpoint_filepath = (
    "./checkpointMNIST/" if args.dataset == "MNIST" else "./checkpointCIFAR/"
)

# Load model activations for conversion.
max_activations = np.load(
    f"./activations/{args.activation}_{args.dataset}_activations.npy"
)

# Intanciates Artificial Neural Network model
ANN = SimpleCNN(
    num_classes,
    learning_rate,
    threshold,
    simulation_len,
    args.activation,
    args.maxPooling,
    args.softmax,
    input_shape,
)

# Load Artificial Neural Network checkpoint weights
ANN.load_weights(checkpoint_filepath)

# Compile model with Crossentropy loss and Accuracy metric.
ANN.compile(
    ANN.optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    run_eagerly=True,
)

# Build Artificial Neural Network with desired input shape.
ANN.build(input_shape)

# Convert Artificial Neural Network to Spiking Neural Network
SNN = SpikingCNN(ANN, max_activations, 32, 4, args.maxPooling, args.softmax)

# Compile Spiking Neural Network
SNN.compile(
    SNN.optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    run_eagerly=True,
)

# Evaluate converted Spiking Neural Network model.
test_accuracy = SNN.evaluate(x=x_test, y=y_test, batch_size=batch_size)
