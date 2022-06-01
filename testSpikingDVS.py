"""Test Artificial Neural Network Model with N-MNIST or N-CIFAR10"""
# Standar modules
import argparse
import random

# Third party modules
import numpy as np
import tensorflow as tf
from spikingjelly.datasets import pad_sequence_collate
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_mnist import NMNIST
from torch.utils.data import DataLoader

# Local modules
from models.net import SimpleCNN, SpikingCNN
from utils.torch2tf import DataGenerator

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="NMNIST", help="NMNIST or NCIFAR")
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
thresh = 5
sim_len = 32


# Choose checkpoint path depending on dataset
checkpoint_filepath = (
    "./checkpointNMNIST/" if args.dataset == "NMNIST" else "./checkpointNCIFAR/"
)

# Load model activations for conversion.
max_activations = np.load(
    f"./activations/{args.activation}_{args.dataset}_activations.npy"
)

num_classes = 10  # total classes.
if args.dataset == "NMNIST":
    input_shape = (1, 1, 34, 34, 2)  # Data shape with time dimension.
else:
    input_shape = (1, 1, 128, 128, 2)  #Data shape with time dimension.

# Intanciates Artificial Neural Network model
ANN = SimpleCNN(
    num_classes,
    learning_rate,
    thresh,
    sim_len,
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
SNN = SpikingCNN(ANN, max_activations, 32, 4, args.maxPooling, args.softmax)

# Compile Spiking Neural Network
SNN.compile(
    SNN.optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    run_eagerly=True,
)

# Dataset build and evaluation of converted Spiking Neural Network model.
if args.dataset == "NMNIST":

    # Load N-MNIST dataset
    root_dir = "./N-MNIST"
    fixed_frames_number_set = NMNIST(
        root_dir, train=True, data_type="frame", frames_number=1, split_by="number"
    )

    # Create Pytorch DataLoader
    train_data_loader = DataLoader(
        fixed_frames_number_set,
        collate_fn=pad_sequence_collate,
        batch_size=args.bsz,
        shuffle=True,
    )

    # Convert DataLoader
    dataloader = DataGenerator(train_data_loader, num_classes)

    # Test SNN
    test_accuracy = SNN.evaluate(dataloader)

else:
    # Load N-CIFAR10 dataset
    root_dir = "./CIFAR"
    fixed_frames_number_set = CIFAR10DVS(
        root_dir, data_type="frame", frames_number=32, split_by="number"
    )

    # Split Event-Based dataset into training and testing
    x_train, y_train = [], []
    x_test, y_test = [], []

    for frame, label in fixed_frames_number_set:

        # Swap axes for Tensorflow model
        frame = np.swapaxes(frame, 1, 3)
        if random.randint(1, 10) != 1:
            x_train.append(frame)
            y_train.append(label)
        else:
            x_test.append(frame)
            y_test.append(label)

    x_train = np.array(x_train, np.float32)
    x_test = np.array(x_test, np.float32)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Test SNN
    test_accuracy = SNN.evaluate(x=x_test, y=y_test, batch_size=batch_size)

