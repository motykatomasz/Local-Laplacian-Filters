import numpy as np


def grayscaleR(i, g, sigma):
    """
    Remapping function for gray-scale images.
    :param i: Image pixel value.
    :param g: Gaussian pyramid coefficient.
    :param sigma: User-defined parameter
    :return:
    """

    i = i.astype(np.int)
    g = g.astype(np.int)

    if np.abs(i-g) <= sigma:
        return _grayscaleRd(i, g, sigma).astype(np.uint8)
    else:
        return _grayscaleRe(i, g, sigma).astype(np.uint8)




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



def _fd(i):
    """
    Smooth detail mapping for Detail Manipulation (5.2 in the paper)
    :param i: Value to map
    :return: Mapped value
    """
    _alpha = 0.8
    return np.power(i, _alpha)


# TODO
def _fe(i):
    """
    Smooth edge mapping for Detail Manipulation (5.2 in the paper)
    :param i: Value to map
    :return: Mapped value
    """
    return i
