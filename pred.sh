# How to use the script given here :

# launch the script main and specify :
#   - the output directory : out_dir
#   - the options save_raster or save_png (True,False) 
#   - the type of dataset (online or local). 
#       - local : mention the path to the image and dem directory : img_dir and dem_dir 
#       - online : will automatically SI and SA tiles from the url provided in data/url/url_to_download.csv'

# more options are defined in the arg parser file : utils/argparser

python main.py --out_dir predictions --save_raster  --dataset online 