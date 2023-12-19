"CS7180 Advanced Perception  12/13/2023   Anirudh Muthuswamy, Gugan Kathiresan, Aditya Varshney"

import LoRAParametrization
import torch
from torch import nn
from loadGenerator import load_generator
import torch.nn.utils.parametrize as parametrize

device = torch.device('mps')

# Only add the parameterization to the weight matrix, ignore the Bias


def linear_layer_parameterization(layer, rank=4, lora_alpha=1):

    features_in, features_out = layer.weight.shape
    return LoRAParametrization(
        features_in, features_out, rank=rank, lora_alpha=lora_alpha
    )

# add lora parameterization to each convolution layer to input model

def conv_layer_parameterization(layer,rank = 4, lora_alpha = 1):

    features_in, features_out = layer.weight.view(layer.weight.shape[0], -1).shape
    return LoRAParametrization(
        features_in, features_out, rank=rank, lora_alpha=lora_alpha
    )



# Function to enable/disable LoRA
def enable_disable_lora(generator, enabled=True):
    for layer in generator.modules():
        if isinstance(layer, nn.Conv2d):
            layer.parametrizations["weight"][0].enabled = enabled

# Register parametrization for all convolutional layers

def register_parameterization_conv_layers(generator):
    for layer in generator.modules():
        if isinstance(layer, nn.Conv2d):
            parametrize.register_parametrization(
                layer, "weight", conv_layer_parameterization(layer, device)
            )

#method to display the parameter difference between the generator wwthout lor aands
        
        
def display_parameter_difference(generator):        
        
    total_parameters_original = 0

    # Iterate through all layers in the model
    for index, layer in enumerate([module for module in generator.modules() if isinstance(module, nn.Conv2d)]):
        total_parameters_original += layer.weight.nelement() + layer.bias.nelement()
        # print(f'Conv Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape}')

    print(f'Total number of convolutional parameters: {total_parameters_original:,}')
    
    print('\n\n\n')

    total_parameters_lora = 0
    total_parameters_non_lora = 0
    for index, layer in enumerate([module for module in generator.modules() if isinstance(module, nn.Conv2d)]):
        total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement() + layer.parametrizations["weight"][0].lora_B.nelement()
        total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()
        # print(
            # f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape} + Lora_A: {layer.parametrizations["weight"][0].lora_A.shape} + Lora_B: {layer.parametrizations["weight"][0].lora_B.shape}'
        # )
    # The non-LoRA parameters count must match the original network
    assert total_parameters_non_lora == total_parameters_original
    print(f'Total number of parameters (original): {total_parameters_non_lora:,}')
    print(f'Total number of parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}')
    print(f'Parameters introduced by LoRA: {total_parameters_lora:,}')
    parameters_incremment = (total_parameters_lora / total_parameters_non_lora) * 100
    print(f'Parameters incremment: {parameters_incremment:.3f}%')

#main method to register parameterization and display parameter parameterization.

    if __name__ == '__main__':

        generator = load_generator(weight_path = './weights/RealESRGAN_x4.pth')
        register_parameterization_conv_layers(generator)
        display_parameter_difference(generator)
