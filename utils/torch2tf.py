"""Convert Tensorflow data loader to Pytorch data loader
Code adapted from on https://github.com/SdahlSean/PytorchDataloaderForTensorflow"""

# Third party modules
import numpy as np
import tensorflow.keras as k


class DataGenerator(k.utils.Sequence):
    """
    class to be fed into model.fit_generator method of tf.keras model

    uses a pytorch dataloader object to create a new generator object that can
    be used by tf.keras dataloader in pytorch must be used to load image data
    transforms on the input image data can be done with pytorch, model fitting
    still with tf.keras

    ...

    Attributes
    ----------
    gen : torch.utils.data.dataloader.DataLoader
        pytorch dataloader object; should be able to load image data for pytorch model
    ncl : int
        number of classes of input data; equal to number of outputs of model
    """

    def __init__(self, gen, ncl):
        """
        Parameters
        ----------
        gen : torch.utils.data.dataloader.DataLoader
            pytorch dataloader object; should be able to load image data for
            pytorch model
        ncl : int
            number of classes of input data equal to number of outputs of model
        """
        self.gen = gen
        self.iter = iter(gen)
        self.ncl = ncl

    def __getitem__(self, _):
        """
        function used by model.fit_generator to get next input image batch

        Variables
        ---------
        ims : np.ndarray
            image inputs; tensor of (batch_size, height, width, channels); input of model
        lbs : np.ndarray
            labels; tensor of (batch_size, number_of_classes); correct outputs for model
        """
        # Catch when no items left in iterator
        try:
            ims, lbs, _ = next(
                self.iter
            )  # Generation of data handled by pytorch dataloader

        except StopIteration:
            self.iter = iter(self.gen)  # Reinstanciate iteator of data
            ims, lbs, _ = next(
                self.iter
            )
    
        # Swap dimensions of image data to match tf.keras dimension ordering
        ims = ims.squeeze()
        ims = np.swapaxes(ims.numpy(), 1, 3)
        lbs = lbs.numpy()

        return ims, lbs

    def __len__(self):
        """ Returns the number of batches in one epoch"""
        return len(self.gen)
