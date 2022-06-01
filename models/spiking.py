"""Convert ANN layers to SNN layers based on
@inproceedings{
    deng2021optimal,
    title={Optimal Conversion of Conventional Artificial Neural Networks to Spiking Neural Networks},
    author={Shikuang Deng and Shi Gu},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=FZ1oTwcXchK}
}
https://github.com/Jackn0/snn_optimal_conversion_pipeline"""

# Third party modules
import tensorflow as tf


class SpikingReLu(tf.keras.layers.Layer):
    """Spiking ReLU implementation"""
    def __init__(self, conv_layer, thresh, T):
        """Init section"""
        super(SpikingReLu, self).__init__()
        self.conv_layer = conv_layer
        self.running_mem = 0
        self.thresh = thresh

    def call(self, inputs):
        """Runs Spiking ReLU activation"""
        x = self.conv_layer(inputs)
        self.running_mem += x
        spike = (
            tf.cast(
                tf.math.greater_equal(self.running_mem, self.thresh), dtype=tf.float32
            )
            * self.thresh
        )
        self.running_mem -= spike
        return spike


class SpikingBNReLu(tf.keras.layers.Layer):
    """Spiking Batch Normalization implementation"""
    def __init__(self, conv_layer, bn_layer, thresh, T):
        """Init section"""
        super(SpikingBNReLu, self).__init__()
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.running_mem = 0
        self.thresh = thresh

    def call(self, inputs):
        """Runs Spiking Batch Normalization"""
        x = self.conv_layer(inputs)
        x = self.bn_layer(x)
        self.running_mem += x
        spike = (
            tf.cast(
                tf.math.greater_equal(self.running_mem, self.thresh), dtype=tf.float32
            )
            * self.thresh
        )
        self.running_mem -= spike
        return spike


class SpikingMaxPool(tf.keras.layers.Layer):
    """Spiking ReLU implementation"""
    def __init__(self, pool_size, stride, padding):
        """Init section"""
        super(SpikingMaxPool, self).__init__()
        self.running_mem = 0
        self.kernal = pool_size
        self.strides = stride
        self.padding = padding
        self.maxpool = tf.keras.layers.MaxPool2D(
            self.kernal, self.strides, self.padding
        )

    def call(self, inputs):
        """Runs Spiking MaxPooling"""
        self.running_mem += inputs
        _, max_idxs = tf.nn.max_pool_with_argmax(
            self.running_mem,
            self.kernal,
            self.strides,
            self.padding,
            include_batch_in_index=True,
        )
        x_max = tf.scatter_nd(
            tf.reshape(max_idxs, (-1, 1)),
            tf.ones(tf.size(max_idxs)),
            [tf.reduce_prod(inputs.shape)],
        )

        x_max = tf.reshape(x_max, inputs.shape)
        x_masked = inputs * x_max
        return self.maxpool(x_masked)


class SpikeSoftMax(tf.keras.layers.Layer):
    """Spiking Softmax implementation"""
    def __init__(self, dense_layer, thresh, T):
        """Init section"""
        super(SpikeSoftMax, self).__init__()
        self.dense_layer = dense_layer
        self.running_mem = 0
        self.thresh = thresh

    def call(self, inputs):
        """Runs Softmax"""
        x = self.dense_layer(inputs)
        self.running_mem += x
        output_Spikes = tf.less_equal(
            tf.random.uniform(tf.shape(self.running_mem)),
            tf.nn.softmax(self.running_mem),
        )
        spike = tf.cast(output_Spikes, tf.keras.backend.floatx()) * self.thresh
        self.running_mem -= spike
        return spike
