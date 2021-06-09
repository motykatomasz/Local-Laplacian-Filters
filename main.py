import time
from collections import OrderedDict

from utils import parse_cli_overides, readImage, writeImage, showImg
from model import LocalLaplacianFilter

# Input parameters
config = OrderedDict(
    img_path='data/memorial_cropped_256.png',
    out_path='results/memorial_cropped_256_reconstructed.png',
    color_img=True,
    intensity_img=True,
    mapping_func='grayscale',   # 'color' or 'grayscale'
    levels=4,
    sigma=0.477,
    alpha=1.0,
    beta=1.0,
    num_processes=16
)


def main():

    img = readImage(config['img_path'], config['color_img'])/255

    algorithm = LocalLaplacianFilter(config)

    start = time.time()
    new_img = algorithm.run(img)
    end = time.time()

    print('Algorithm ran for: {:.4f} seconds.'.format(end - start), flush=True)

    path = config['out_path'].split('.')

    out_name = ''.join([path[0], '_a_' + str(config['alpha']), '_b_' + str(config['beta']),
                        '_s_' + str(config['sigma'])])
    out_path = '.'.join([out_name, path[1]])
    
    writeImage(new_img, out_path, config['color_img'])


if __name__ == "__main__":
    config = parse_cli_overides(config)
    print("########## Printing Full Config ##############", flush=True)
    print(config, flush=True)
    print("########## END ###########", flush=True)

    main()
