import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
sys.path.append('.')
from utils.dataset_utils import  AbsoluteScaler
import requests
import uuid
import os
import re



class SwissImageDataset_online(Dataset):
    '''Load Data from swissimage, swissalti3d from url   provided in file'''
    def __init__(self, url_list='data/url/url_to_download.csv',   debug=False, use_tiling = True):
               
        # List of input tiles to use : 
        data = pd.read_csv(url_list, header = None,)  
        print('Dataset from online images')
        print('will download and predict',len(data),'images for url.')
        self.urls_SI = data[0].to_list()
        self.urls_SA = data[1].to_list()
        if debug:
            self.urls_SA = self.urls_SA[:10]
            self.urls_SI = self.urls_SI[:10]


        # Transform values for mce model
        self.dem_mean, self.dem_std = 41.32,19.18
        mce_mean = np.array([0.5585, 0.5771, 0.5543], dtype=np.float32)
        mce_std = np.array([0.2535, 0.2388, 0.2318], dtype=np.float32)
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mce_mean, mce_std)
        ])
        
        self.dem_transform = transforms.Compose([
        #    transforms.ToTensor(),
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
        if len(self.urls_SA) != len(self.urls_SI) :
            raise Exception( 'Not the same number of swissimage and swissalti url !!')
        return len(self.urls_SA)
    
    def download_file(self,url ):
        
        if not os.path.exists ('tmp/'):
            # Create the directory
            os.mkdir('tmp/')
        destination = 'tmp/'+str(uuid.uuid4())[:8]+'.tif'

        response = requests.get(url)
        with open(destination, 'wb') as file:
            file.write(response.content)
        
        data = Image.open(destination)  
        data = data.resize((2000,2000))
        
        try:
            os.remove(destination)
        except FileNotFoundError:
            print('file not deleted, pass')
            pass


        return data
    
    def get_id_from_url(self,url):
        # Define the pattern using regular expression
        pattern = re.compile(r'\d{4}-\d{4}_0.1_2056.tif')
        match = pattern.search(url)
        id = match.group()
        id = id[:9].replace('-','_')
        return id 

    def __getitem__(self, idx):
        #Read images :
        SI_url = self.urls_SI[idx]
        tile_id = self.get_id_from_url(SI_url)
        SA_url = self.urls_SA[idx]

        raw_image = self.download_file(SI_url)
        dem = self.download_file(SA_url)
        # resize rgb images from 4000x4000 to 2000x2000 to match the size of dem   
         
        # apply transform to IMG for mce model   
        mce_image = self.basic_transform(raw_image)  
        img=mce_image   
        mce_image = self.unfold(mce_image)
        mce_image = mce_image.moveaxis(0,1).reshape((-1,3,self.kernel_size,self.kernel_size))                                   
        
        # apply transform to DEM for mce model
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
    




if __name__ =="__main__":
   
    
    
    ds = SwissImageDataset_online()
    for _,input,_,_ in ds:
        print(input.mean((0,2,3)))

    

    