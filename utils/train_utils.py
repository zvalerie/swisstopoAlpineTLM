
import torch
from torch.utils.data import DataLoader
import os
import sys
sys.path.append('.')


from utils.SwissDataset import SwissImageDataset
from utils.SwissDataset_download import SwissImageDataset_online
from model.models_utils import model_builder
from model.binaryRockClassifier import DeepLabv3


def get_models(args):

    MCEmodel = model_builder( 
                num_classes = 10, 
                pretrained_backbone = False, 
                num_experts = 3,
                aggregation = 'mean',
                )
    binaryModel = DeepLabv3(in_channels=4,nb_class=2) #2 classes for binary model

    # load_best_model_weights
    if os.path.isfile (args.mce_best_weights) :
        checkpoint = torch.load(args.mce_best_weights)
        best_weights = checkpoint['state_dict']
        MCEmodel.load_state_dict (best_weights)
        print('Weights loaded from best MCE model :', args.mce_best_weights)
    else :
        raise FileNotFoundError
    
    # load_best_model_weights
    if os.path.isfile (args.binary_best_weights) :
        best_weights = torch.load(args.binary_best_weights)
        binaryModel.load_state_dict (best_weights)
        print('Weights loaded from best binary classifier model :',args.binary_best_weights)
    else :
        raise FileNotFoundError
    
    MCEmodel.eval()
    binaryModel.eval()

    return MCEmodel,binaryModel
    


def get_dataloader(args=None):
    """     Create dataset   based on arguments from config

    Args:
        args (dict) : args from config file
    """    
   
    img_dir = args.img_dir 
    dem_dir = args.dem_dir

    
    # Create output folder if needed :
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    

    if args.dataset == 'online':
        dataset = SwissImageDataset_online(debug=args.debug)
        
        
    else :
        dataset = SwissImageDataset(
            img_dir, 
            dem_dir, 
            debug=args.debug)
        
    dataloader = DataLoader(
        dataset,
        batch_size= args.bs,
        shuffle=False,
        num_workers= 4,
        pin_memory=True
        )
        
    return dataloader


