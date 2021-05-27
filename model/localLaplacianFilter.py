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
        self.color = config['color_img']


    def run(self, img: np.ndarray):
        """
        Runs the Local Laplacian Filtering algorithm.
        :param img: Input image
        :return:
        """

        if self.color:
            img, colorRatios = self.computeIntensityImage(img)

        gpImg = self.computeGaussianPyramid(img, self.levels)   # Gaussian Pyramid of input image
        lpOut = []                                              # Output Laplacian Pyramid

        for l, gpLayer in enumerate(gpImg):
            h, w = gpLayer.shape
            lpOutLayer = np.zeros(shape=(h, w))
            for x in range(h):
                for y in range(w):
                    g = gpLayer[x, y]

                    # get sub-region R
                    a, b, c, d = self.subregion(l, x, y, img.shape[0], img.shape[1])
                    R = img[a:b, c:d]

                    # make R~ same size as R
                    R_ = np.zeros_like(R)

                    # iterate through pixels of R
                    for u in range(R.shape[0]):
                        for v in range(R.shape[1]):
                            # apply remapping function on R and assign to R~
                            # R_[u, v] = self.mapping(R[u, v], g, self.sigma, self.alpha, self.beta)
                            R_[u, v] = R[u, v]

                    # 9: Intermediate Laplacian pyramid
                    lpIntermediate = self.computeLaplacianPyramid(R_, self.levels)
                    # 10: Update output pyramid

                    # x,y position in original image offset by location of subregion
                    x_sub = x*2**l - a
                    y_sub = y*2**l - c
                    # adjust for the current level of the pyramid
                    x_sub_l = np.floor(x_sub/2**l).astype(np.int)
                    y_sub_l = np.floor(y_sub/2**l).astype(np.int)
                    lpOutLayer[x, y] = lpIntermediate[l][x_sub_l, y_sub_l]

            lpOut.append(lpOutLayer.astype(np.uint8))

        if self.color:
            # 12: collapse output pyramid
            reconstruction = self.reconstructLaplacianPyramid(lpOut).astype(np.int)
            # 13: reconstruct color image
            scaled = reconstruction - reconstruction.max()
            inverse = np.exp(scaled) * 255
            colored = [np.multiply(inverse, channel) for channel in colorRatios]
            colored = np.stack(colored, axis=2)

            return colored

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


    def subregion(self, l: int, x: int, y: int, h_img: int, w_img: int):
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
        k = 4*2**(l+2)

        # x,y coords corresponding to the original image
        x_img = x*2**l
        y_img = y*2**l

        # starts and ends  in x and y direction
        # always make sure subregion will be power of 2, even in border cases
        if x_img - k//2 < 0:
            a = 0
            b = k
        elif x_img + k//2 > w_img:
            a = w_img - k
            b = w_img
        else:
            a = x_img - k//2
            b = x_img + k//2

        if y_img - k//2 < 0:
            c = 0
            d = k
        elif y_img + k//2 > h_img:
            c = h_img - k
            d = h_img
        else:
            c = y_img - k//2
            d = y_img + k//2

        assert x_img >= a and x_img <= b and y_img >= c and y_img <= d, f"Error: sub-region ({a, c}, {b, c}, {a, d}, {b, d}) does not include the point {x_img, y_img}"

        return a, b, c, d

    def computeIntensityImage(self, img):
        """
        Computed the intensity image and color ration of a given color image img.
        :param img: Color image.
        :return: Tuple of Intensity image and color ratios
        """
        img64bit = img.astype(np.int)
        r = img64bit[:, :, 0]
        g = img64bit[:, :, 1]
        b = img64bit[:, :, 2]

        intensityImg = (20*r + 40*g + b) + 1    # +1 to prevent dividing by 0
        colorRatios = [r/intensityImg, g/intensityImg, b/intensityImg]

        # TODO uint8 type is needed for cv2.pyrUP/Down function but it's a problem when opearting with log(img).
        # TODO We probably have to implement cv2.pyrUP/Down ourselves.

        return np.log(intensityImg).astype(np.uint8), colorRatios
