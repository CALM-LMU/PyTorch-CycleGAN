from skimage.exposure import rescale_intensity

class RescaleToZeroOne(object):
    #Converts a PyTorch Tensor with RGB Range [0, 255] to PyTorch Tensor [0,1]

    def __call__(self, pic):
        """
        Args:
            pic (tensor or numpy.ndarray): Image to be rescaled.
        Returns:
            Tensor: Converted image.
        """
        #return pic/255
        return pic/pic.max()

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RescaleToOneOne(object):
    #Converts a PyTorch Tensor with RGB Range [0, 255] to PyTorch Tensor [0,1]

    def __call__(self, pic):
        """
        Args:
            pic (tensor or numpy.ndarray): Image to be rescaled.
        Returns:
            Tensor: Converted image.
        """
        return ((pic/pic.max())*2)-1

    def __repr__(self):
        return self.__class__.__name__ + '()'