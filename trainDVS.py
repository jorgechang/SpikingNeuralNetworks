"""Train Artificial Neural Network Model with MNIST or CIFAR10"""
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
from models.net import SimpleCNN
from utils.torch2tf import DataGenerator

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="NMNIST", help="NMNIST or NCIFAR")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--epochs", type=int, default=2, help="epochs")
parser.add_argument("--bsz", type=int, default=64, help="batch size")
parser.add_argument("--activation", type=str, default="thReLU", help="thReLU or ReLU")
parser.add_argument("--maxPooling", type=int, default=0, help="enable MaxPooling")
parser.add_argument("--softmax", type=int, default=0, help="enable Softmax")
args = parser.parse_args()

# Training hyperparameters
learning_rate = args.lr
epochs = args.epochs
batch_size = args.bsz

# Simulation parameters for Spiking Neural Network
thresh = 5
sim_len = 1

# Event-Based dataset building and preprocessing
if args.dataset == "NMNIST":

    num_classes = 10  # total classes (0-9 classes).
    input_shape = (1, 34, 34, 1)  # image shape for N-MNIST.

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

else:

    num_classes = 10  # total classes.
    input_shape = (1, 128, 128, 2)  # image shape for N-CIFAR10.

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


# Define checkpoint weights path.
checkpoint_filepath = (
    "./checkpointNMNIST/" if args.dataset == "NMNIST" else "./checkpointNCIFAR/"
)

# Define model callback to handle weights saving.
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

# Define Convolutional neural network.
model = SimpleCNN(
    num_classes,
    learning_rate,
    thresh,
    sim_len,
    args.activation,
    args.maxPooling,
    args.softmax,
    input_shape,
)

# Compile model with Crossentropy loss and Accuracy metric.
model.compile(
    model.optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    run_eagerly=True,
)

# Train model with correct parameters for N-MNIST and N-CIFAR10
if args.dataset == "NMNIST":
    history = model.fit(
        dataloader,
        epochs=args.epochs,
        verbose="auto",
        validation_split=0.0,
        validation_data=None,
        callbacks=[model_checkpoint_callback],
    )
else:
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=args.epochs,
        verbose="auto",
        validation_split=0.0,
        validation_data=None,
        callbacks=[model_checkpoint_callback],
    )


# Define model for Testing.
test_model = SimpleCNN(
    num_classes,
    learning_rate,
    thresh,
    sim_len,
    args.activation,
    args.maxPooling,
    args.softmax,
    input_shape,
)

# Load stored checkpoint.
test_model.load_weights(checkpoint_filepath)

# Compile Testing model.
test_model.compile(
    test_model.optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    run_eagerly=True,
)

# Load Event-Based data for Testing and Evaluate
if args.dataset == "NMNIST":
    fixed_frames_number_set = NMNIST(
        root_dir, train=False, data_type="frame", frames_number=1, split_by="number"
    )
    test_data_loader = DataLoader(
        fixed_frames_number_set,
        collate_fn=pad_sequence_collate,
        batch_size=args.bsz,
        shuffle=False,
    )
    dataloader = DataGenerator(test_data_loader, 10)

    test_accuracy = test_model.evaluate(dataloader)

else:

    test_accuracy = test_model.evaluate(x=x_test, y=y_test, batch_size=args.bsz)

print(f"Test Accuracy: {test_accuracy[1]:.2f}")

# Save model activations for conversion.
max_activations = np.array(
    [i.numpy() if not isinstance(i, int) else i for i in model.activations]
)
np.save(
    f"./activations/{args.activation}_{args.dataset}_activations.npy",
    max_activations,
)
