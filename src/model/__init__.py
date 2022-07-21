from .resnet import GrayResNet
from .densenet import GrayDenseNet
from .vicreg import VICReg, TempVICReg, TempVICRegConc
from .linear_classifier import Linear_Protocoler
from .resnet_preact import GrayPreActResNet

Models = {
    "GrayResNet": GrayResNet,
    "GrayDenseNet": GrayDenseNet,
    "GrayPreActResNet": GrayPreActResNet,
    "VICReg": VICReg,
    "Linear": Linear_Protocoler,
    "TempVICReg": TempVICReg,
    "TempVICRegConc": TempVICRegConc
}