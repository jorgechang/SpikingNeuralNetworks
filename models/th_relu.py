"""Threshold ReLU layeyr based on
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


class ThReLu:
    """Threshold ReLU implementation"""
    def __init__(self, th, sim_len):
        """Init section"""
        self.th = th
        self.sim_len = sim_len

    def forward(self, input):
        """Runs Threshold ReLU activation"""
        input = input - self.th / (self.sim_len * 2)
        aux = tf.math.abs(input - self.th / 2) <= self.th / 2
        comp = (input - self.th) > 0
        return (
            input * tf.cast(aux, dtype=tf.float32)
            + tf.cast(comp, dtype=tf.float32) * self.th
        )
