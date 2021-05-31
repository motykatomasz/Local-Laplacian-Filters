import cv2
from PIL import Image
import numpy as np
from scipy.signal import convolve


def readImage(imgPath: str, color: bool):
    """
    Reads the image at specific path.
    :param imgPath: Path to the image.
    :return: Image
    """
    if color:
        # OpenCV reads image in BGR format. We convert it to RGB to be consistent with other libraries
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)[..., ::-1]
    else:
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not read image from path: " + imgPath)

    return img


def writeImage(img: np.array, imgPath: str, color: bool):
    """
    Reads the image at specific path.
    :param imgPath: Path to the image.
    :return: None
    """
    if color:
        img = cv2.imwrite(imgPath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img = cv2.imwrite(imgPath, np.squeeze(img))


def showImg(img):
    """
    Shows the image 'img'.
    :param img: Image
    :return: None
    """
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()

def upsample(img):
    """
    Upsamples the image 'img'.
    :param img: Image
    :return: Image
    """

    # channels = img.shape[2]
    # img_upsample = np.zeros((img.shape[0]*2, img.shape[1]*2, channels))
    img_upsample = np.zeros((img.shape[0]*2, img.shape[1]*2))
    img_upsample[::2, ::2] = img

    gaussian_filter = np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ],
        dtype=np.float32
    )/256.0

    # tile the filters for channels
    # gaussian_filter = np.tile(gaussian_filter[:, :, np.newaxis], (1, 1, channels))
    
    # kH, kW, _ = gaussian_filter.shape
    kH, kW = gaussian_filter.shape
    pad_top_bot = (kH - 1) // 2
    pad_left_right = (kW - 1) // 2
    
    img_upsample = cv2.copyMakeBorder(img_upsample, pad_top_bot, pad_top_bot, pad_left_right, pad_left_right, cv2.BORDER_REFLECT_101)
    img_blur = convolve(img_upsample, gaussian_filter, mode="valid")
    
    img_upsample = img_blur*4

    return img_upsample