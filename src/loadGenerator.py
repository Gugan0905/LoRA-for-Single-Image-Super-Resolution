"CS7180 Advanced Perception  12/13/2023   Anirudh Muthuswamy, Gugan Kathiresan, Aditya Varshney"

import torch
from model.rrdbnet_arch import RRDBNet

# This method loads the generator as RRDBNet with pretrained weights

def load_generator(weight_path = './weights/RealESRGAN_x4.pth'):
    
    generator =  RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                         num_block=23, num_grow_ch=32, scale=4
                        )
    if weight_path:
        loadnet = torch.load(weight_path)
        if 'params' in loadnet:
            generator.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            generator.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            generator.load_state_dict(loadnet, strict=True)
            
    return generator
