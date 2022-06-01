"""Train Artificial Neural Network Model with MNIST or CIFAR10"""
# Standar modules
import argparse

# Third party modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist

# Local modules
from models.net import SimpleCNN


# Define and parse input arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MNIST", help="MNIST or CIFAR")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--epochs", type=int, default=1, help="epochs")
parser.add_argument("--bsz", type=int, default=64, help="batch size")
parser.add_argument("--activation", type=str, default="thReLU", help="thReLU or ReLU")
parser.add_argument("--maxPooling", type=int, default=0, help="enable maxPooling")
parser.add_argument("--softmax", type=int, default=0, help="enable softmax")
args = parser.parse_args()

# Training hyperparameters
learning_rate = args.lr
epochs = args.epochs
batch_size = args.bsz

# Simulation parameters for Spiking Neural Network
threshold = 2
simulation_len = 32

# Dataset build and preprocessing
if args.dataset == "MNIST":
    num_classes = 10  # total classes (0-9 digits).
    input_shape = (28, 28, 1)  # image shape for MNIST.
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

# Define checkpoint weights path.
checkpoint_filepath = (
    "./checkpointMNIST/" if args.dataset == "MNIST" else "./checkpointCIFAR/"
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
    threshold,
    simulation_len,
    args.activation,
    args.maxPooling,
    args.softmax,
    input_shape,
)

# Compile model with Crossentropy loss and Accuracy metric.
model.compile(
    model.optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True if not args.softmax else False
    ),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    run_eagerly=True,
)

# Train model
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose="auto",
    validation_split=0.1,
    callbacks=[model_checkpoint_callback],
)

# Define model for Testing.
test_model = SimpleCNN(
    num_classes,
    learning_rate,
    threshold,
    simulation_len,
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

# Evaluate Testing model.
test_accuracy = test_model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
print(f"Test Accuracy: {test_accuracy[1]:.2f}")

# Save model activations for conversion.
max_activations = np.array(
    [i.numpy() if not isinstance(i, int) else i for i in model.activations]
)
np.save(
    f"./activations/{args.activation}_{args.dataset}_activations.npy",
    max_activations,
)
