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
        if imgPath.split('.')[1] == 'hdr':
            img = cv2.imread(imgPath, cv2.IMREAD_ANYCOLOR)[..., ::-1].astype(np.float)
        else:
            img = cv2.imread(imgPath, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float)
    else:
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE).astype(np.float)
    if img is None:
        print("Could not read image from path: " + imgPath)

    return img


def writeImage(img: np.array, imgPath: str, color: bool):
    """
    Saves the image at imgPath.
    :param imgPath: Path to the image.
    :return: None
    """
    img = np.clip(img, 0, 255).astype('uint8')
    if color:
        img = cv2.imwrite(imgPath, img[..., ::-1])
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
