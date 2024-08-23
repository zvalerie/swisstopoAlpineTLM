# swisstopoAlpineTLM
Generate alpine land cover prediction based on swisstopo TLM labels, SwissIMAGE and SwissALTI3D raster layers.

### Study Area  : 
![Alt Text](data/shp/grid_area_above_200m/grid_area_above2000m.PNG)

## Usage

1. Install all the dependencies with conda  : 

```bash
conda env create --file environment.yml
```

2. Load the model weights for the following zenodo repository, unzip them and place them in the ``data/weights/`` folder: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13365159.svg)](https://doi.org/10.5281/zenodo.13365159)




3. Launch the script via the command line and specify the following arguments  : 

Use the following script to predict the alpine land cover for files located in the ``img_dir`` and ``dem_dir`` folders, and save the predictions as geolocated raster and as figures in the ``out_dir`` directory: 
```
python main.py --out_dir output/ --save_raster --save_png  
--dataset local 
--large_tiles  
--img_dir /YOUR/PATH/TO/swissIMAGES/ 
--dem_dir /YOUR/PATH/TO/swissALTI3D/
```

Use the following script to predict land cover with the online dataset, i.e. swissIMAGES and swissALTI3D tiles are directly downloaded from swisstopo servers. To determine which tiles to predict, update the urls in the following csv file :  ``data/url/url_to_download.csv``. Each line must contain the path to one swissIMAGE file and  one swissALTI3D file, separated by a comma. :  
```bash
python main.py --out_dir output/ --save_raster --save_png  --dataset online
```
Essential command lines options are described below :
-  img_dir : Directory to store SWISSIMAGES files (Default :  '/data/', Type :  str)
-  dem_dir : Directory to store SWISSAlti3d files (Default :  '/data/', Type :  str)
-  dataset : Choose between local dataset and online dataset (Default :  'local', Type :  str)
-  out_dir : Directory to save outputs (Default :  'output/', Type :  str)
-  save_raster :  Save predictions as raster (Default :  False, Type : bool)
-  save_png :  Save predictions as PNG (Default :  False, Type : bool)
-  large_tiles :  Use large tiles of 1x1km, swissIMAGE default raster distributed online (Default :  False, Type : bool)

Advanced command line options : 

-  num_workers :  Number of dataloader workers. Increase for multiprocessing (Default :  1, Type :  int)
-  debug :  Debugging mode. Predict only a few tiles (Default :  False, Type : bool)
-  device : Choose the device on which to run the model. Options :  "cpu" or "cuda:0" (Default :  'cuda:0', Type :  str)
-  mce_best_weights :  Choose model weights for MCE model (Default :  'data/weights/current_best.pt', Type :  str)
-  binary_best_weights :  Choose model weights for the binary classifier model (Default :  'data/weights/binary_best_weights.pt', Type :  str)
-  bs :  Batch size (Default :  1, Type :  int)

