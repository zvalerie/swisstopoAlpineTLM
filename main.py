from utils.argparser import parse_args
import os
import torch
import numpy as np
from pprint import pprint
from utils.train_utils import get_dataloader , get_models
from utils.predict import get_predictions_from_logits
from utils.plot import save_preds_to_file
from tqdm import tqdm
import time


def main(args):

    # Choose model and device :
    device =  args.device
    MCEmodel,binaryModel = get_models(args)

    MCEmodel = MCEmodel.to(device)
    binaryModel = binaryModel.to(device)


    # Get dataloaders : 
    dataloader = get_dataloader(args)
    print('Start model predictions....')
    tick= time.time()
    
    use_tiling =  not args.large_tiles  # default to True use tiling
    if use_tiling :       
        fold = dataloader.dataset.fold
    else : 
        fold =None
    
    if not args.save_png  and not args.save_raster :
        if not args.debug:
            raise 'save to png and save to raster are false ! one of them needs to be true'

    
    with torch.no_grad():        
        for i, (raw_img, mce_input, bin_input,tile_ids) in tqdm(enumerate(dataloader),ncols=30):
            
            
            # move data to device
            img =  raw_img.squeeze()  # for plotting purpose
            mce_input = mce_input.squeeze().to(device) 
            bin_input = bin_input.squeeze().to(device)       
                    
              
            B,C,W,H = mce_input.shape
            
    
            
            if use_tiling: 
                # create batch of single image and padding
                MCEoutput_exp_0 = torch.zeros([B,10,W,H])
                MCEoutput_exp_1 = torch.zeros([B,10,W,H])
                MCEoutput_exp_2 = torch.zeros([B,10,W,H])

                for idx in   np.linspace(0,361-19,19):
                    idx=int(idx)  
                    input = mce_input[idx:idx+19,:,:,:]
                    out = MCEmodel(input)
                    MCEoutput_exp_0[idx:idx+19,:,:,:] =  out ['exp_0']
                    MCEoutput_exp_1[idx:idx+19,:,:,:] =  out ['exp_1']
                    MCEoutput_exp_2[idx:idx+19,:,:,:] =  out ['exp_2']   
                MCEoutput =  {'exp_0':MCEoutput_exp_0,
                            'exp_1':MCEoutput_exp_1,
                            'exp_2':MCEoutput_exp_2,
                            }
            else : 
                MCEoutput = MCEmodel(mce_input)
                    
            
            binaryoutput = binaryModel(bin_input)


            # Get predictions and fold back to large image : 
            mce_pred,entropy = get_predictions_from_logits(MCEoutput,fold_fn=fold)
            bin_pred = get_predictions_from_logits(binaryoutput,fold_fn=fold)

            
            save_preds_to_file(img, mce_pred,bin_pred, entropy, args,tile_ids, 
                               save_to_png = args.save_png, 
                               save_to_raster = args.save_raster
                               ) 

        # measure elapsed time
        tack= time.time()
        print('Elapsed time [s]:',int(tack-tick))








if __name__ == '__main__':
    print('Predict TLM rocky classes based on swissalti and swissimage files ')
    args = parse_args()

    pprint(vars(args))
    
    
    main(args)
    

