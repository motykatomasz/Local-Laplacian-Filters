import cv2


def readImage(imgPath: str):
    """
    Reads the image at specific path.
    :param imgPath: Path to the image.
    :return: Image
    """
    img = cv2.imread(imgPath, cv2.CV_8UC3)
    if img is None:
        print("Could not read image from path: " + imgPath)

    return img
