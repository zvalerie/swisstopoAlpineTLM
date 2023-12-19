
import os
import sys
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import rasterio as rio


def   save_preds_to_file(mce_input, mce_pred,bin_pred, entropy, 
                         args=None, tile_ids =None, 
                         save_to_png =False, save_to_raster=False
                         ):

    
    # find number of sample and loop over them: 
    if len(mce_input.shape)==4:
        B,C,H,W = mce_input.shape
        mce_input = mce_input[:,:3,:,:]
    else : 
        C,H,W = mce_input.shape
        B=1
        mce_input = mce_input[:3,:,:]
    img = mce_input.squeeze().cpu().numpy()
    img = unnormalize_images(img)
    mce = mce_pred .squeeze().cpu().numpy()
    bin = bin_pred.squeeze().cpu().numpy().T
    ent = entropy.squeeze().cpu().numpy()

    # Loop 
    if B ==1 :            
        if save_to_png :
            plot_predictions_as_png(img,mce,bin,ent,tile_ids[0],args=args )
            
        if save_to_raster:
            #raise NotImplementedError
            plot_predictions_as_raster(mce,bin,ent,tile_ids[0],args=args )
    
    elif B >1 : 
        for idx in range(B) : 
        
            tile_id = tile_ids[idx]
            
            if save_to_png :
                plot_predictions_as_png(img[idx],mce[idx],bin[idx],ent[idx],tile_id,args=args )
            if save_to_raster:
                plot_predictions_as_raster(mce[idx],bin[idx],ent[idx],tile_id,args=args)


    return


def plot_predictions_as_png(img,mce,bin,ent,tile_id,args=None ):
    
    mce_classes = {'Background':0, "Bedrock" : 1, "Bedrockwith grass" : 2,
                "Large blocks" : 3, "Large blocks with grass" : 4, "Scree" : 5,
                "Scree with grass" : 6,"Water" : 7,
                "Forest" : 8, "Glacier" : 9, }
    
    final = mce*bin
    final[final>6]=0
    final[mce==2]=2
    final[mce==3]=3
    final[mce==4]=4

    
    # Loop over the samples and plot them
    colors = ['black', 'tab:grey', 'lightgrey', 'maroon', 'red', 'orange', 'yellow', 'royalblue', 'forestgreen','lightcyan',]
    cmap=ListedColormap(colors)
    colors2 = ['black',  'red']
    cmap2=ListedColormap(colors2)
    
    plt.figure(figsize =[25,5])
    plt.subplot(151)
    plt.imshow(np.moveaxis(img,0,-1))
    plt.title("RGB Image")
    plt.axis('off')

    plt.subplot(152)
    plt.imshow(mce, cmap=cmap,vmin=0,vmax=9, interpolation_stage = 'rgba')
    plt.title("MCE predictions (band 2)")
    plt.axis('off')        
    
    plt.subplot(153)
    plt.imshow(bin, cmap=cmap2,vmin=0,vmax=1, interpolation_stage = 'rgba')
    plt.title("Rock predictions (in red, band 3)")
    plt.axis('off')
                
    plt.subplot(154)
    plt.imshow(ent, cmap='viridis')
    plt.title("Entropy - Uncertainty (band 4)")
    plt.axis('off')
    
    plt.subplot(155)
    plt.imshow(final, cmap=cmap,vmin=0,vmax=9, interpolation_stage = 'rgba')
    plt.title("Final prediction (band 1)")
    plt.axis('off') 
    
    # Colorbar parameters :  
    colors = cmap.colors
    values = list(  mce_classes .values())
    txt_labels = list(  mce_classes.keys())
    patches = [ mpatches.Patch(color=colors[i], label= txt_labels[i] ) for i in values ]
    
    plt.legend(handles=patches, 
        fontsize='small',
        bbox_to_anchor=(1.05, 1), 
        loc=2, 
        frameon = False,
        borderaxespad=0. )    

        
    out_name =  args.out_dir +'/' + tile_id + '.png' if args is not None else   tile_id + '.png'
    plt.savefig(out_name,dpi = 100, bbox_inches='tight', )
    print('fig saved as ', out_name)
    plt.close()


def unnormalize_images(images, ):
    # Assuming images is a NumPy array with shape (batch_size, height, width, channels)
    # mean and std should be lists or arrays with length equal to the number of channels
    mean = [0.5585, 0.5771, 0.5543]  
    std = [0.2535, 0.2388, 0.2318] 

    unnormalized_images = np.copy(images)
    for i in range(len(mean)):
        unnormalized_images[i,  :, :] = (unnormalized_images[i,  :, :] * std[i]) + mean[i]
    
    return unnormalized_images     

def plot_predictions_as_raster(mce_preds,bin_preds,entropy,tile_id,args=None):
           
    pred_fn =  args.out_dir +'/raster/' + tile_id + '_pred.tif' if args is not None else   tile_id + '.tif'
    if not os.path.exists(args.out_dir +'/raster/') : 
        os.mkdir (args.out_dir +'/raster/')
    
    #produce final clean pred
    final = mce_preds*bin_preds
    final[final>6]=0
    final[mce_preds==2]=2
    final[mce_preds==3]=3
    final[mce_preds==4]=4
    
    # transform entropy to int32 
    # max entropy 
    data = np.stack((final,mce_preds,bin_preds,entropy),axis=0)
    data = np.uint8(data)
 
  
    # Define the geolocation information:
    
    
    C,width, height = data.shape
    west, south = tile_id.rsplit('_')
    west, south  = float(west) * 1e3, float(south) * 1e3
    
    transform = rio.transform.from_bounds(
        west = west , 
        south = south , 
        east = west + width*0.5,
        north = south + height*0.5, 
        width = width, 
        height = height
        )
                                    
    
    # Open a new raster file for writing
    with rio.open(  pred_fn, 'w', 
                    driver='GTiff',
                    width=width, 
                    height=height,
                    count=4, 
                    dtype=rio.uint8,
                    nodata=0,
                    crs='EPSG:2056',
                    transform=transform) as dst:

        # Write the data to the raster file
        dst.write(data)


        # Close the raster file
        dst.close()
    print('fig saved as ', pred_fn)
    
    
    



if __name__ == '__main__':
    mce_input =torch.rand([2,4,1920,1920])
    mce_pred = torch.randint(high = 9, size=[2,1,1920,1920])
    bin_pred = torch.randint(high=2, size=[2,1,1920,1920])
    entropy = torch.rand([2,1,1920,1920])
    tile_ids = ['a','b']
    
    save_preds_to_file(mce_input, mce_pred,bin_pred, entropy, tile_ids=tile_ids, save_to_png =True, save_to_raster=False)
