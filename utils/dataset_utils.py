import math
import numpy as np
import random
from PIL import Image
import itertools
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

class AbsoluteScaler(object):
    '''
        Transform the altitue values to absolute elevation per tile
    '''
    def __init__(self,):
        pass

    def __call__(self, dem):  
        min_per_c,_ = torch.min(dem.flatten(1),dim=-1)
        min_per_c = min_per_c.reshape(dem.shape[0],1,1,1)
        scaled_dem = dem - min_per_c
        return scaled_dem
    
    
