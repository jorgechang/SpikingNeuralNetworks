"""Artificial and Spiking Neural Networks."""
# Third party modules
import tensorflow as tf

# Local modules
from models.spiking import (
        SpikeSoftMax,
        SpikingBNReLu,
        SpikingMaxPool,
        SpikingReLu
)
from models.th_relu import ThReLu

print(tf.__version__)


class SimpleCNN(tf.keras.Model):
    """Simple Convolutional Neural Network"""

    def __init__(
        self,
        num_classes,
        learning_rate,
        threshold,
        simulation_len,
        activation,
        maxPooling,
        softmax,
        input_shape,
    ):
        """Init section"""
        super(SimpleCNN, self).__init__()

        # Define Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Defines activation function
        if activation == "thReLU":
            self.relu = ThReLu(threshold, simulation_len)
            activation_layer = tf.keras.layers.Lambda(self.relu.forward)
        elif activation == "ReLU":
            activation_layer = tf.keras.layers.ReLU()

        # Build Convolutional Neural Network model
        self.model = [
            tf.keras.layers.InputLayer(input_shape),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", name="block1_conv1"),
            activation_layer,
        ]

        # MaxPooling layer
        if maxPooling:
            self.model.append(tf.keras.layers.MaxPool2D(strides=(2, 2), padding="SAME"))

        # Classification layer
        self.model.append(tf.keras.layers.Flatten())
        self.model.append(
            tf.keras.layers.Dense(
                num_classes, activation="softmax" if softmax else None, name="Dense"
            )
        )

        # Target variable for max activation storing
        self.activations = [0] * len(self.model)

    def call(self, input):
        """Performs forward propagation.

        Parameters:
        -----------
            input [tf.Tensor]:
                Batch of images to propagate.

        Returns:
        --------
            input [tf.Tensor]:
                Probabilities from Dense layer classification head.
        """

        for i, layer in enumerate(self.model):
            input = layer(input)
            self.activations[i] = tf.math.maximum(
                tf.math.reduce_max(input), self.activations[i]
            )

        return input


class SpikingCNN(tf.keras.Model):
    """Spiking Neural Network"""
    def __init__(self, ANN, thresholds, T, step, maxPooling, softmax):
        """Init section"""
        super(SpikingCNN, self).__init__()

        # Copy Artificial Neural Network parameters
        self.T = T
        self.step = step
        self.spiking_model = []
        self.maxPooling = maxPooling
        self.optimizer = ANN.optimizer

        # Copy Convolutional Layer and convert ReLU activation
        self.spiking_model.append(
            SpikingReLu(ANN.get_layer("block1_conv1"), thresholds[2], T)
        )

        # Convert MaxPooling layer if required
        if maxPooling:
            self.spiking_model.append(SpikingMaxPool((2, 2), 2, "SAME"))
        self.spiking_model.append(tf.keras.layers.Flatten())

        # Convert Softmax layer if required
        if softmax:
            self.spiking_model.append(
                SpikeSoftMax(ANN.get_layer("Dense"), thresholds[-1], T)
            )
        else:
            self.spiking_model.append(
                SpikingReLu(ANN.get_layer("Dense"), thresholds[-1], T)
            )

    def call(self, input):
        """Performs forward propagation.

        Parameters:
        -----------
            input [tf.Tensor]:
                Batch of images to propagate.

        Returns:
        --------
            result_list [list]:
                Accuracy list of size simulation_len.
        """

        # Restart Spiking layer variables after each batch propagation
        result_list = []
        out_spike_num = 0
        self.spiking_model[0].running_mem = 0
        self.spiking_model[3 if self.maxPooling else 2].running_mem = 0

        # Run Spiking Simulation
        for time in range(self.T):
            x = input
            for layer in self.spiking_model:

                x = layer(x)

            out_spike_num += x
            if (time + 1) % self.step == 0:
                sub_result = out_spike_num / (time + 1)
                result_list.append(sub_result)

        return result_list


class SimpleCNNDVS(tf.keras.Model):
    """Simple Convolutional Neural Network for Event-Based datasets"""
    def __init__(
        self,
        num_classes,
        learning_rate,
        threshold,
        simulation_len,
        activation,
        maxPooling,
        softmax,
        input_shape,
    ):
        """Init section"""
        super(SimpleCNNDVS, self).__init__()

        # Define Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Defines activation function
        if activation == "thReLU":
            self.relu = ThReLu(threshold, simulation_len)
            activation_layer = tf.keras.layers.Lambda(self.relu.forward)
        elif activation == "ReLU":
            activation_layer = tf.keras.layers.ReLU()

        # Build Convolutional Neural Network model with Batch Normalization
        self.model = [
            tf.keras.layers.InputLayer(input_shape),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", name="block1_conv1"),
            tf.keras.layers.BatchNormalization(name="block2_bn2", momentum=0.1),
            activation_layer,
        ]

        # MaxPooling layer
        if maxPooling:
            self.model.append(tf.keras.layers.MaxPool2D(strides=(2, 2), padding="SAME"))

        # Classification layer
        self.model.append(tf.keras.layers.Flatten())
        self.model.append(
            tf.keras.layers.Dense(
                num_classes, activation="softmax" if softmax else None, name="Dense"
            )
        )

        # Target variable for max activation storing
        self.activations = [0] * len(self.model)

    def call(self, input):
        """Performs forward propagation.

        Parameters:
        -----------
            input [tf.Tensor]:
                Batch of images to propagate.

        Returns:
        --------
            result_list [list]:
                Accuracy list with simulation steps.
        """

        for i, layer in enumerate(self.model):
            input = layer(input)
            self.activations[i] = tf.math.maximum(
                tf.math.reduce_max(input), self.activations[i]
            )

        return input


class SpikingCNNDVS(tf.keras.Model):
    """Spiking Neural Network for Event-Based datasets"""
    def __init__(self, ANN, thresholds, T, step, maxPooling, softmax):
        """Init section"""
        super(SpikingCNNDVS, self).__init__()

        # Copy Artificial Neural Network parameters
        self.T = T
        self.step = step
        self.spiking_model = []
        self.maxPooling = maxPooling
        self.optimizer = ANN.optimizer

        # Copy Convolutional Layer, BatchNormalization, and convert ReLU activation
        self.spiking_model.append(
            SpikingBNReLu(
                ANN.get_layer("block1_conv1"),
                ANN.get_layer("block2_bn2"),
                thresholds[3],
                T,
            )
        )

        # Convert MaxPooling layer if required
        if maxPooling:
            self.spiking_model.append(SpikingMaxPool((2, 2), 2, "SAME"))
        self.spiking_model.append(tf.keras.layers.Flatten())

        # Convert Softmax layer if required
        if softmax:
            self.spiking_model.append(
                SpikeSoftMax(ANN.get_layer("Dense"), thresholds[-1], T)
            )
        else:
            self.spiking_model.append(
                SpikingReLu(ANN.get_layer("Dense"), thresholds[-1], T)
            )

    def call(self, input):
        """Performs forward propagation.

        Parameters:
        -----------
            input [tf.Tensor]:
                Batch of images to propagate.

        Returns:
        --------
            result_list [list]:
                Accuracy list with simulation steps.
        """

        # Restart Spiking layer variables after each batch propagation
        result_list = []
        out_spike_num = 0
        self.spiking_model[0].running_mem = 0
        self.spiking_model[3 if self.maxPooling else 2].running_mem = 0

        # Run Spiking Simulation
        for time in range(self.T):
            x = input
            for layer in self.spiking_model:

                x = layer(x)

            out_spike_num += x
            if (time + 1) % self.step == 0:
                sub_result = out_spike_num / (time + 1)
                result_list.append(sub_result)

        return result_list
