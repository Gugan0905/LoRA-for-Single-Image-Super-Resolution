"CS7180 Advanced Perception  12/13/2023   Anirudh Muthuswamy, Gugan Kathiresan, Aditya Varshney"

from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19

#The following class defines the conceptual loss as the output from VGG19 pretrained model

class FeatureExtractor(nn.Module):
    def __init__(self, layers=[0, 5, 10, 19, 28]):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.layers = layers
        self.net = nn.Sequential(*[vgg19_model.features[i] for i in range(max(layers) + 1)])

    def forward(self, img):
        features = []
        for i in range(len(self.net)):
            img = self.net[i](img)
            if i in self.layers:
                features.append(img)
        return features

