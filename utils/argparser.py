import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Predict alpine TLM on Swissimage and Swissalti3d images')
   
    parser.add_argument('--bs',
                        help='batch size',
                        default=1,
                        type=int)
    parser.add_argument('--out_dir',
                        help='directory to save outputs',
                        default='output/',
                        type=str)
    parser.add_argument('--save_raster',
                        help='save predictions as raster, True/False',
                        default=False,
                        action='store_true',)
    parser.add_argument('--save_png',
                        help='save predictions as png, True/False',
                        default=False,
                        action='store_true',)
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=1,
                        type=int)
    parser.add_argument('--debug',
                        help='is debuging mode ?, predict only a few tiles',
                        default=False,
                        action='store_true')
    parser.add_argument('--device',
                        help='choose device on which to run the model : between "cpu" and "cuda:0"',
                        default='cuda:0',
                        type=str)
    parser.add_argument('--large_tiles',
                        action='store_true',
                        default=False)
    parser.add_argument('--mce_best_weights',
                        help='choose model weights for mce model',
                        default='data/weights/current_best.pt',
                        type=str)
    parser.add_argument('--binary_best_weights',
                        help='choose model weights for binary classifier model',
                        default='data/weights/binary_best_weights.pt',
                        type=str)
    parser.add_argument('--img_dir',
                        help='directory to store SWISSIMAGES files',
                        default='',
                        type=str)
    parser.add_argument('--dem_dir',
                        help='directory to store SWISSAlti3d files',
                        default='',
                        type=str)
    parser.add_argument('--dataset',
                        help='choose between local dataset and online dataset',
                        default='local',
                        type=str)
    args = parser.parse_args()
    
    return args