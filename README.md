# Local-Laplacian-Filters
Reproduction of the paper "Local Laplacian filters: edge-aware image processing with a Laplacian pyramid" for the course Advanced Digital Image Processing at TU Delft

## How to run the algorithm?

### Requirements:
* NumPy
* OpenCV
* PIL
* SharredArray


### Options:
* img_path: path to the input image
* out_path: path to save the results
* color_img: Indicates whether we process color or grayscale image.
* intensity_img: Whether to use intensity image for processing color image
* mapping_func: Type of remapping function 'color' or 'grayscale'
* levels: Number of levels for Gaussian/Laplacian Pyramid
* sigma: Algorithm Hyperparameter
* alpha: Algorithm Hyperparameter
* beta: Algorithm Hyperparameter
* num_processes: Number of processes to run the algorithm

### Example Usage:
> *python main.py --img_path data/desk_256.hdr --out_path results/result.png --color_img True --intensity_img True --mapping_func grayscale
--levels 5 --sigma 0.4 --alpha 1.0 --beta 0.5 --num_processes 16*


### Exemplary results (Detail manipulation):
![name-of-you-image](https://github.com/motykatomasz/Local-Laplacian-Filters/blob/main/results/example_detail_enhancement.png?raw=true)

