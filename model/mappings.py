import numpy as np


def grayscaleR(i, g, sigma, alpha, beta):
    """
    Remapping function for gray-scale images.
    :param i: Image pixel value.
    :param g: Gaussian pyramid coefficient.
    :param sigma: User-defined parameter
    :return:
    """

    if np.abs(i-g) <= sigma:
        return _grayscaleRd(i, g, sigma, alpha)
    else:
        return _grayscaleRe(i, g, sigma, beta)


def colorR(i, g, sigma, alpha, beta):
    """
        Remapping function for color images.
    :param i:
    :param g:
    :param sigma:
    :return:
    """
    # Eq 3a, 3b:
    # details are within sphere of radius sigma centered at g
    # edges are outside sphere
    # eq for points inside sphere: sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2) <= R
    # (cx, cy, cz) is center
    if np.linalg.norm(i-g) <= sigma:
        return _colorRd(i, g, sigma, alpha)
    else:
        return _colorRe(i, g, sigma, beta)


# TODO
def _grayscaleRd(i, g, sigma, alpha):
    """
    Remapping function for gray-scale images.
    :return:
    """
    return g + np.sign(i-g) * sigma * _fd(np.abs(i-g)/sigma, alpha)


def _grayscaleRe(i, g, sigma, beta):
    """
    Remapping function for gray-scale images.
    :return:
    """
    return g + np.sign(i-g) * (_fe(np.abs(i-g)-sigma, beta) + sigma)


def _colorRd(i, g, sigma, alpha):
    """
    Remapping function for color images.
    :return:
    """
    if np.all(i-g == 0):
        unit = np.zeros_like(i)
    else:
        unit = (i-g)/np.linalg.norm(i-g)

    return g + unit * sigma * _fd(np.linalg.norm(i-g)/sigma, alpha)


def _colorRe(i, g, sigma, beta):
    """
    Remapping function for color images.
    :return:
    """
    if np.all(i-g == 0):
        unit = np.zeros_like(i)
    else:
        unit = (i-g)/np.linalg.norm(i-g)

    return g + unit * (_fe(np.linalg.norm(i-g)-sigma, beta) + sigma)


def _fd(i, alpha):
    """
    Smooth detail mapping for Detail Manipulation (5.2 in the paper)
    :param i: Value to map
    :return: Mapped value
    """
    return np.power(i, alpha)


# TODO
def _fe(i, beta):
    """
    Smooth edge mapping for Detail Manipulation (5.2 in the paper)
    :param i: Value to map
    :return: Mapped value
    """
    return beta * i
