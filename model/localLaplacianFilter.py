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


    def run(self, img: np.ndarray):
        """
        Runs the Local Laplacian Filtering algorithm.
        :param img: Input image
        :return:
        """
        gpImg = self.computeGaussianPyramid(img, self.levels)   # Gaussian Pyramid of input image
        lpOut = []                                              # Output Laplacian Pyramid

        lp = self.computeLaplacianPyramid(img, self.levels)

        for l, gpLayer in enumerate(gpImg):
            h, w = gpLayer.shape
            lpOutLayer = np.zeros(shape=(h, w))
            for y in range(h):
                for x in range(w):
                    g = gpLayer[y, x]

                    # get sub-region R
                    a, b, c, d = self.subregion(l, y, x, img.shape[0], img.shape[1])
                    R = img[c:d+1, a:b+1]
                    
                    # make R~ same size as R
                    R_ = np.zeros_like(R)

                    # iterate through pixels of R
                    for u in range(R.shape[1]):
                        for v in range(R.shape[0]):
                            # apply remapping function on R and assign to R~
                            R_[u, v] = self.mapping(R[u, v], g, self.sigma)

                    # 9: Intermediate Laplacian pyramid
                    lpIntermediate = self.computeLaplacianPyramid(R_, self.levels)
                    # 10: Update output pyramid

                    # x,y position in original image offset by location of subregion
                    x_sub = x*2**l - a
                    y_sub = y*2**l - c
                    # adjust for the current level of the pyramid
                    x_sub_l = np.floor(x_sub/2**l).astype(np.int)
                    y_sub_l = np.floor(y_sub/2**l).astype(np.int)
                    lpOutLayer[y, x] = lpIntermediate[l][y_sub_l, x_sub_l]

            lpOut.append(lpOutLayer.astype(np.uint8))

        # 12: collapse output pyramid
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
        for _ in range(l):
            G = cv2.pyrDown(G)
            gpImg.append(G)
        
        return gpImg

    def computeLaplacianPyramid(self, img, l: int):
        """
        Computes and returns the Gaussian Pyramid of 'l' levels from image 'img'.
        :param img: Image.
        :param l: Number of levels.
        :return: Laplacian Pyramid.
        """
        gpImg = self.computeGaussianPyramid(img, l)
        gpImg.reverse()
        lpImg = [gpImg[0]]
        for i in range(l):
            GE = cv2.pyrUp(gpImg[i])
            L = gpImg[i + 1] - GE
            lpImg.append(L)
        lpImg.reverse()
        return lpImg


    def reconstructLaplacianPyramid(self, lp):
        """
        Reconstructs image from Laplacian Pyramid.
        :param lp: Laplacian Pyramid
        :param l: Number of levels
        :return: Reconstructed image
        """
        l = len(lp)
        lp.reverse()
        reconstruction = lp[0]
        for i in range(1, l):
            reconstruction = cv2.pyrUp(reconstruction)
            reconstruction = reconstruction + lp[i]

        return reconstruction


    def subregion(self, l: int, y: int, x: int, h_img: int, w_img: int):
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
        k = 3*2**(l+2) - 3

        # x,y coords corresponding to the original image
        x_img = x*2**l
        y_img = y*2**l

        # starts and ends  in x and y direction
        a = np.maximum(x_img - k, 0)
        b = np.minimum(w_img, x_img + k)
        c = np.maximum(y_img - k, 0)
        d = np.minimum(h_img, y_img + k)

        assert x >= a and x <= b and y >= c and y <= d, f"Error: sub-region ({a, c}, {b, c}, {a, d}, {b, d}) does not include the point {x, y}"

        return a, b, c, d
