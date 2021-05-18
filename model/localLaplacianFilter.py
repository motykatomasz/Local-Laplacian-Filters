import cv2
import numpy as np

from .mappings import grayscaleR, colorR


class LocalLaplacianFilter:
    """
    Class representing Laplacian Local Filtering Algorithm
    """
    def __init__(self, config: dict):
        self.levels = config['levels']
        self.sigma = config['sigma']
        self.mapping = self.getMappingFunction(config['mapping_func'])          # TODO Implement mapping functions
        self.alpha = config['alpha']
        self.beta = config['beta']


    def run(self, img):
        """
        Runs the Local Laplacian Filtering algorithm.
        :param img: Input image
        :return:
        """
        gpImg = self.computeGaussianPyramid(img, self.levels)   # Gaussian Pyramid of input image
        lpOut = []                                              # Output Laplacian Pyramid

        for l, gpLayer in enumerate(gpImg):
            h, w = gpLayer.shape
            lpOutLayer = np.zeros(shape=(h, w))

            for x in range(h):
                for y in range(w):
                    g = gpLayer[x, y]
                    intermediateImg = img.copy()               # Intermediate image representation

                    for m in range(h):
                        for n in range(w):
                            intermediateImg[m, n] = self.mapping(img[m, n], g, self.sigma)          # TODO Implement method for computing subregion R

                    lpIntermediate = self.computeLaplacianPyramid(intermediateImg, self.levels)     # Intermediate Laplacian pyramid
                    lpOutLayer[x, y] = lpIntermediate[l][x, y]

            lpOut.append(lpOutLayer)

        return self.reconstructLaplacianPyramid(lpOut)

    def getMappingFunction(self, func: str):
        """
        Returns function object of the mapping function.
        :param func: Name of the mapping function
        :return: Function object
        """
        model = {
            "grayscale": lambda: grayscaleR,
            "color": lambda: colorR
        }

        return model[func]()

    def computeGaussianPyramid(self, img, l: int):
        """
        Computes and returns the Gaussian Pyramid of 'l' levels from image 'img'.
        :param img: Image.
        :param l: Number of levels.
        :return: Gaussian Pyramid.
        """
        G = img.copy()
        gpImg = [G]
        for i in range(l):
            G = cv2.pyrDown(G)
            gpImg.append(G)

        gpImg.reverse()
        return gpImg

    def computeLaplacianPyramid(self, img, l: int):
        """
        Computes and returns the Gaussian Pyramid of 'l' levels from image 'img'.
        :param img: Image.
        :param l: Number of levels.
        :return: Laplacian Pyramid.
        """
        gpImg = self.computeGaussianPyramid(img, l)
        lpImg = [gpImg[0]]
        for i in range(l):
            GE = cv2.pyrUp(gpImg[i])
            L = gpImg[i + 1] - GE
            lpImg.append(L)

        return lpImg


    def reconstructLaplacianPyramid(self, lp):
        """
        Reconstructs image from Laplacian Pyramid.
        :param lp: Laplacian Pyramid
        :param l: Number of levels
        :return: Reconstructed image
        """
        l = len(lp)
        reconstruction = lp[0]
        for i in range(1, l):
            reconstruction = cv2.pyrUp(reconstruction)
            reconstruction = reconstruction + lp[i]

        return reconstruction


    def subregion(self, img, l: int, x: int, y: int, w: int, h: int):
        """
        Computes the subregion R of original image based on level 'l' and position (x,y).
        :return: Tuple of 4 integers where:
        a -> start of R in x direction
        b -> end of R x direction
        c -> start of R in y direction
        d -> end of R in y direction

        Assumes the laplacian sub-sampling is 5x5
        """
        # define the size of sub-region according to section 4
        # try to define a kxk region around x, y
        k = 3*(2**(l+2) - 1)

        # starts and ends  in x and y direction

        a = np.floor(x - 0.5*(k-1))
        b = np.ceil(x + 0.5*(k-1))
        c = np.floor(y - 0.5*(k-1))
        d = np.ceil(y + 0.5*(k-1))

        # if start_x extends outside image
        # try to accomodate extra pixels on right side of x
        if a < 0:
            # if right side cannot accomodate extra pixels
            # set start_x and end_x to edges of image
            if b + np.abs(a) >= w:
                a = 0
                b = w - 1
            # if right side can accomodate extra pixels
            # set start_x to left edge, extend end_x
            else:
                b = b + np.abs(a)
                a = 0
        # if end_x extends outside image
        # try to accomodate extra pixels on right side of x
        elif b >= w:
            if a - (b - (w - 1)) < 0:
                a = 0
                b = w - 1
            else:
                a = a - (b - (w - 1))
                b = w - 1

        if c < 0:
            if d + np.abs(c) >= h:
                c = 0
                d = h - 1
            else:
                d = d + np.abs(c)
                c = 0
        elif d >= h:
            if c - (d - (h - 1)) < 0:
                c = 0
                d = h - 1
            else:
                c = c - (d - (h - 1))
                d = h - 1

        return img[a:b, c:d]
