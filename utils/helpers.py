import cv2
from PIL import Image
import numpy as np


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


def showImg(img):
    """
    Shows the image 'img'.
    :param img: Image
    :return: None
    """
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()
