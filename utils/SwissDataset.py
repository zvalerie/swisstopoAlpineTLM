import os,sys, glob,re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import re
sys.path.append('.')
from utils.dataset_utils import  AbsoluteScaler
from random import shuffle




class SwissImageDataset(Dataset):
    '''Load Data from swissimage, swissalti3d'''
    def __init__(self, img_dir, dem_dir,   debug=False, use_tiling = True):
        
        self.img_dir = img_dir
        self.img_format0 = '{}_rgb.tif'
        self.img_format1 = 'DOP25_LV95_{}_2056.tif'  

        self.dem_dir = dem_dir
        self.dem_format = 'swissalti3d_2019_{}_2056.tif'
        self.debug = debug
        
        # List of input tiles to use : 
        if  debug:
            self.tiles_ids = ['2581_1128','2581_1084','2583_1123','2585_1095',
                              '2615_1094','2615_1095','2615_1096',
                              '2576_1087','2577_1086','2578_1087','2579_1087', # scree with grass
                              '2572_1117','2604_1097','2605_1097','2605_1098', # Large rocks
                              '2616_1094','2616_1095','2616_1096',
                              '2571_1097','2590_1099',
                              '2594_1092','2649_1146',
                              '2631_1133','2622_1101',
                              '2646_1112','2617_1102','2563_1111'
                               ]
            
        else :
           # Get the list of all files stored in img_dir and dem dir and get their common ids : 
            self.tiles_ids = get_tiles_ids_from_folder(img_dir,dem_dir)                   
            self.tiles_ids = sorted(self.tiles_ids)
            #self.tiles_ids = self.tiles_ids


        # Transform values for mce model
        self.dem_mean, self.dem_std = 41.32,19.18
        mce_mean = np.array([0.5585, 0.5771, 0.5543], dtype=np.float32)
        mce_std = np.array([0.2535, 0.2388, 0.2318], dtype=np.float32)
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mce_mean, mce_std)
        ])
        
        self.dem_transform = transforms.Compose([
            AbsoluteScaler(),
            transforms.Normalize(self.dem_mean, self.dem_std)
        ])

        # values for binary model
        bin_means =   [  0.50,   0.53,   0.46,  292.,]
        bin_stds =    [  0.16,   0.16,   0.16,  137.,] 
        self.bin_augm = transforms.Compose([   
                        transforms.Normalize(bin_means ,bin_stds),
                    ])  

        # use tiling or not
        self.use_tiling = use_tiling
        if use_tiling : 
            self.kernel_size = 200
            stride = 100
            output_size = 2000 
            self.fold = torch.nn.Fold(output_size,self.kernel_size ,stride=stride)        
            self.unfold = torch.nn.Unfold(self.kernel_size,padding =0, stride=stride) 
            
    def __len__(self):
        return len(self.tiles_ids)
    
    def __getitem__(self, idx):

        #Read images :
        tile_id = self.tiles_ids[idx]
        try : 
            img_path = self.img_dir +self.img_format0.format(tile_id)
            raw_image = Image.open(img_path) 
        except:
            img_path = self.img_dir +self.img_format1.format(tile_id)
            raw_image = Image.open(img_path)         
        raw_image = raw_image.resize((2000,2000)) ## resize rgb images from 4000x4000 to 2000x2000 to match the size of dem  
        
        # read dem 
        dem_path = self.dem_dir +self.dem_format.format(tile_id.replace('_','-'))       
        dem = Image.open(dem_path)
        
        # apply transform for mce model   
        mce_image = self.basic_transform(raw_image)  
        img=mce_image   
        mce_image = self.unfold(mce_image)
        mce_image = mce_image.moveaxis(0,1).reshape((-1,3,self.kernel_size,self.kernel_size))                                   
        
        mce_dem = torch.from_numpy(np.array(dem))
        mce_dem = self.unfold(mce_dem.unsqueeze(0))
        mce_dem = mce_dem.moveaxis(0,1).reshape((-1,1,self.kernel_size,self.kernel_size))       
        mce_dem = self.dem_transform(mce_dem)
        mce_input = torch.cat((mce_image,mce_dem),axis=1)

        # apply transform for binary classifier model  :
        bin_image = np.array(raw_image)/256
        bin_dem = np.array(dem)
        bin_image = torch.from_numpy(bin_image).swapaxes(-1,0)
        bin_dem = torch.from_numpy(bin_dem-bin_dem.min()).swapaxes(-1,0).unsqueeze(0)
        bin_input = torch.cat((bin_image, bin_dem),axis=0)
        bin_input = self.bin_augm(bin_input)   
        bin_input = self.unfold(bin_input.squeeze()  ).moveaxis(0,1).reshape((-1,4,self.kernel_size,self.kernel_size))

        return img, mce_input, bin_input.float(), tile_id
    
    
def get_tiles_ids_from_folder(img_dir,dem_dir):
    print('\nRead images and DEM folder ...' )
    print('swissimage folder :', img_dir)
    print('swissalti folder :', dem_dir)
    rgb_list = [ x for x in os.listdir(img_dir) if x[-4:]=='.tif']
    dem_list = [ x for x in os.listdir(dem_dir) if x[-4:]=='.tif']

    rgb_ids = [re.findall(r'\d{4}_\d{4}', x)[0] for x in rgb_list]
    dem_ids = [re.findall(r'\d{4}-\d{4}', x)[0].replace('-','_') for x in dem_list]
    
    common_ids = [x for x in rgb_ids if x in dem_ids]
    
    print('\nImages files found:',len(rgb_ids),'\nDem files found:',len(dem_ids), '\nCommon images - dem ids found :',len(common_ids))
    print('common ids found (max 10):', common_ids[:10],'\n')
    
    return common_ids 



if __name__ =="__main__":
    
    dataset_csv =   'data/subset.csv'
        
    img_dir = '/data/valerie/swisstopo/SI_2020_50cm/'
    dem_dir = '/data/valerie/swisstopo/ALTI_2020_50cm/'
    x =torch.rand([4,2000,2000])
    unfold = nn.Unfold(kernel_size=(200,200), stride=200,padding=0)
    
    y = unfold(x).moveaxis(0,1).reshape((-1,4,200,200))
    print(y.shape)

    
    
    ds = SwissImageDataset(img_dir,dem_dir,debug= False)
    for input,_,_ in ds:
        print(input.mean((1,2)))

    

    