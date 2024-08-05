# swisstopoAlpineTLM
Generate alpine land cover prediction based on swisstopo TLM labels, SwissIMAGE and SwissALTI3D raster layers.

### Study Area :
![Alt Text](data/shp/grid_area_above_200m/grid_area_above2000m.PNG)

## Usage

1. Install all the dependencies with conda :

```bash
conda env create --file environment.yml
```

2. Launch the script via the command line and specify the following arguments :

-  bs: Batch size (Default: 1, Type: int)
-  out_dir:Directory to save outputs (Default: 'output/', Type: str)
-  save_raster: Save predictions as raster (Default: False)
-  save_png: Save predictions as PNG (Default: False)
-  num_workers: Number of dataloader workers (Default: 1, Type: int)
-  debug: Debugging mode. Predict only a few tiles (Default: False)
-  device:Choose the device on which to run the model. Options: "cpu" or "cuda:0" (Default: 'cuda:0', Type: str)
-  large_tiles: Use large tiles (Default: False)
-  mce_best_weights: Choose model weights for MCE model (Default: 'data/weights/current_best.pt', Type: str)
-  binary_best_weights: Choose model weights for the binary classifier model (Default: 'data/weights/binary_best_weights.pt', Type: str)
-  img_dir:Directory to store SWISSIMAGES files (Default: '/data/', Type: str)
-  dem_dir:Directory to store SWISSAlti3d files (Default: '/data/', Type: str)
-  dataset:Choose between local dataset and online dataset (Default: 'local', Type: str)

example of command line prompt : 
```bash
python main.py --out_dir output/ --save_raster --save_png  -dataset online
```
