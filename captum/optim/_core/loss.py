import torch
import torch.nn as nn
import torch.nn.functional as F




def test_vec(target, vec):
    vec = vec.reshape((1, -1, 1, 1))
    
    target = target[:, 21, 7, 7]#y:y+1, x:x+1
    
    print(target.size(), vec.size())
    #print(target.reshape((1, -1, 1, 1)))
       

 
    #quit()
    return torch.cosine_similarity(target[None, None, None], vec)
    
    
vec = torch.rand(512) * 1000
target = torch.randn(1,512,15,15) 
out = test_vec(target, vec)
    
print(out)