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

    # TODO
    def subregion(self, img, l: int, x: int, y: int):
        """
        Computes the subregion R of original image based on level 'l' and position (x,y).
        :return: Tuple of 4 integers where:
        a -> start of R in x direction
        b -> end of R x direction
        c -> start of R in y direction
        d -> end of R in y direction
        """
        a, b, c, d = 0

        return a, b, c, d



