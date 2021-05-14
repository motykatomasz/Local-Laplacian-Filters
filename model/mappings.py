import numpy as np

def grayscaleR(i, g, sigma):
    """
    Remapping function for gray-scale images.
    :param i: Image pixel value.
    :param g: Gaussian pyramid coefficient.
    :param sigma: User-defined parameter
    :return:
    """
    if np.abs(i-g) <= sigma:
        return _grayscaleRd(i, g, sigma)
    else:
        return _grayscaleRe(i, g, sigma)


# TODO
def colorR(i, g, sigma):
    """
        Remapping function for color images.
    :param i:
    :param g:
    :param sigma:
    :return:
    """
    ...


# TODO
def _grayscaleRd(i, g, sigma):
    """
    Remapping function for gray-scale images.
    :return:
    """
    return g + np.sign(i-g) * sigma * _fd(np.abs(i-g)/sigma)


def _grayscaleRe(i, g, sigma):
    """
    Remapping function for gray-scale images.
    :return:
    """
    return g + np.sign(i-g) * (_fe(np.abs(i-g)-sigma) + sigma)


# TODO
def _fd(i):
    """
    Smooth function mapping between [0, 1] and [0, 1]
    :param i: Value to map
    :return: Mapped value
    """
    ...


# TODO
def _fe(i):
    """
    Smooth nonegative function defined over [0, inf)
    :param i: Value to map
    :return: Mapped value
    """
    ...
