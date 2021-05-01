import cv2
from .mappings import mapping1, mapping2


class LocalLaplacianFilter():
    def __init__(self, config: dict):
        ...

    def run(self, img):
        """
        Runs the algorithm
        :return: Processed image
        """
        ...

    def getMappingFunction(self, func: str):
        """
        Returns function object of the mapping function.
        :param func: Name of the mapping function
        :return: Function object
        """
        model = {
            "mapping1": lambda: mapping1,
            "mapping2": lambda: mapping2
        }

        return model[func]()




