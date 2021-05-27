import os
import sys
import time
from collections import OrderedDict

from utils import parse_cli_overides, readImage, writeImage, showImg
from model import LocalLaplacianFilter

# Input parameters
config = OrderedDict(
    img_path='data/0.png',
    out_path='data/0_reconstructed.png',
    color_img=True,
    intensity_img=False,
    mapping_func='color',   # 'color' or 'grayscale'
    levels=3,
    sigma=1.0,
    alpha=1.0,
    beta=1.0,
    num_processes=4 # Keep in mind the number of processes should be higher than number of rows in the lowest layer
)


def main():

    img = readImage(config['img_path'], config['color_img'])

    algorithm = LocalLaplacianFilter(config)

    start = time.time()
    new_img = algorithm.run(img)
    end = time.time()

    print('Algorithm ran for: {:.4f} seconds.'.format(end - start), flush=True)
    
    writeImage(new_img, config['out_path'], config['color_img'])


if __name__ == "__main__":
    config = parse_cli_overides(config)
    print("########## Printing Full Config ##############", flush=True)
    print(config, flush=True)
    print("########## END ###########", flush=True)

    main()
