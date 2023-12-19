
import torch
import sys
sys.path.append('.')
import torch.nn.functional as F
from torch.nn.functional import softmax
from utils.gaussian_filter import gaussian_kernel

def get_predictions_from_logits(output, fold_fn=None):
    
    
    if 'out' in output.keys():  # for binary model
        logits = output['out']
        B,C,W,H = logits.shape
        
        if fold_fn is not None : 
            logits = fold_fn(logits.reshape(-1,C*W*H).T.float())
        else :
            logits = logits.squeeze()
        
        
        preds = torch.argmax(logits,axis=0).detach().cpu()
        
        return preds
        
    else :
        # Agregation of the 3 experts predictions 
        # Aggregation is simple average : the output logits of class c is the average among the ouputs 
        # from set of experts that trained with class c,  then softmax is applied.        
        device = output['exp_1'].device
        B,C,W,H = output['exp_1'].shape
        sum_logits =  ( output['exp_0'] + output['exp_1'] + output['exp_2'] )

        # Divide the sum of logits by the number of experts used to predict them :
        quotient =  [1,1,2,3,3,1,2,2,1,1]
        quotient = torch.tensor(quotient).unsqueeze(-1).unsqueeze(-1).to(device)
        logits = sum_logits / quotient
        logits= softmax(logits,dim=1)

        # Use a gaussian filter to give more weights to central predictions and less to those close to the border
        filter = gaussian_kernel(size=200,sigma=70,return_tensor=True).to(device)        

        logits = logits*filter 
        

        if fold_fn is not None : 
            logits = fold_fn(logits.reshape(-1,C*W*H).T.float())
        else :
            logits = logits.squeeze()


      
        # Get predictions from MCE model :    
        preds = torch.argmax(logits,axis=0).detach().cpu() 


        # Calculate entropy
        proba = F.normalize(logits, p=1,dim=0)
        
        entropy = -torch.sum(proba * torch.log2(proba + 1e-10), dim=0)
        # normalize and transform entropy to integer (indexfrom 0 to 31)
        max_entropy = -torch.log2(torch.tensor(1/10)) 
        norm_entropy = entropy/max_entropy
        norm_entropy = (norm_entropy *255).long()
   
    return preds, norm_entropy.detach().cpu()
