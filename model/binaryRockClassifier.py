import torch.nn as nn
import torch
import torchvision

def DeepLabv3(in_channels,nb_class):
    
    
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1') 
    
    model.classifier[4] = nn.Conv2d (256,nb_class, kernel_size=1,stride=1 )
    model.aux_classifier[4] = nn.Conv2d(256,nb_class, kernel_size=1,stride=1 )

        
    # Change the first layer (conv1) to add a 4th channel.
    weight_conv1 = model.backbone.conv1.weight.clone()
    new_conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    with torch.no_grad():
        new_conv1.weight[:, :3,:,:] = weight_conv1        # RGB channels
        new_conv1.weight[:, 3] = weight_conv1[:, 0]    # DEM 
        model.backbone.conv1 = new_conv1


    
    return model


if __name__ == '__main__':
    
    model = DeepLabv3(4,7)

    
    x = torch.rand([64,4,192,192])
    out = model(x)
    print(out['out'].shape)
    
