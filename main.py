import time
from collections import OrderedDict

from utils import parse_cli_overides, readImage, showImg
from model import LocalLaplacianFilter

# Input parameters
config = OrderedDict(
    img_path='data/lena_gray_64.png',
    mapping_func='grayscale',
    levels=3,
    sigma=20.0,
    alpha=0.0,
    beta=0.0
)


def main():

    img = readImage(config['img_path'])

    algorithm = LocalLaplacianFilter(config)

    start = time.time()
    new_img = algorithm.run(img)
    end = time.time()

    print('Algorithm ran for: {:.4f} seconds.'.format(end - start), flush=True)

    showImg(new_img)


if __name__ == "__main__":
    config = parse_cli_overides(config)
    print("########## Printing Full Config ##############", flush=True)
    print(config, flush=True)
    print("########## END ###########", flush=True)

    main()
