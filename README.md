# swisstopoAlpineTLM
Generate alpine land cover prediction based on swisstopo TLM labels, SwissIMAGE and SwissALTI3D raster layers.

![Alt Text](relative_path_to_your_image)


## How to use the script given here :

1. Install all the dependencies with conda :

```bash
conda env create --file environment.yml
```

2. Launch the script in main and specify :
- the output directory : out_dir
- the options save_raster or save_png (True,False) 
- the type of dataset (online or local). 
    - local : mention the path to the image and dem directory : img_dir and dem_dir 
    - online : will automatically SI and SA tiles from the url provided in data/url/url_to_download.csv'

more options are defined in the arg parser file : utils/argparser


example of command line prompt : 
```bash
python main.py --out_dir predictions --save_raster  --dataset online 
```