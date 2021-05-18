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


def colorR(i, g, sigma):
    """
        Remapping function for color images.
    :param i:
    :param g:
    :param sigma:
    :return:
    """

    i = i.astype(np.int)
    g = g.astype(np.int)

    # Eq 3a, 3b:
    # details are within sphere of radius sigma centered at g
    # edges are outside sphere
    # eq for points inside sphere: sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2) <= R^2
    # (cx, cy, cz) is center
    if np.linalg.norm(i-g) <= np.sqrt(sigma):
        _colorRd(i, g, sigma).astype(np.uint8)
    else:
        _colorRe(i, g, sigma).astype(np.uint8)


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


def _colorRd(i, g, sigma):
    """
    Remapping function for color images.
    :return:
    """
    if np.all(i-g == 0):
        unit = np.zeros_like(i)
    else:
        unit = (i-g)/np.linalg.norm(i-g)

    return g + unit * sigma * _fd(np.linalg.norm(i-g)/sigma)


def _colorRe(i, g, sigma):
    """
    Remapping function for color images.
    :return:
    """
    if np.all(i-g == 0):
        unit = np.zeros_like(i)
    else:
        unit = (i-g)/np.linalg.norm(i-g)

    return g + unit * (_fe(np.linalg.norm(i-g)-sigma) + sigma)


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
