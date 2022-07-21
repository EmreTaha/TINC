import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models

import sys
sys.path.append('/home/optima/temre/preprocessing/Harbor')

MODELS = {
    "densenet161": (models.densenet161, 2208)
}

class GrayDenseNet(nn.Module):
    def __init__(self, in_channels=1, n_class=3, model_type="densenet161", pretrained=False, **kwargs):
        super(GrayDenseNet, self).__init__()

        self.create_function, self.input_features = MODELS[model_type]
        self.model = self.create_function(pretrained=pretrained, **kwargs)
        #self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.model.features.conv0 = nn.Conv2d(in_channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.model.classifier = nn.Sequential(nn.Linear(self.input_features, n_class, bias=True))#,
                                 #,
                                 #nn.Dropout(0.2),
                                 #nn.Linear(256, n_class))
        
        #nn.init.kaiming_normal_(self.model.features.conv0, mode='fan_out', nonlinearity='relu')        
        
    def forward(self, x):
        return self.model(x)