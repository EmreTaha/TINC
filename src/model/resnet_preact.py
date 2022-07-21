import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, kc=64):
        super(PreActResNet, self).__init__()
        self.in_planes = kc

        self.conv1 = nn.Conv2d(3, kc, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, kc, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, kc*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, kc*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, kc*8, num_blocks[3], stride=2)
        self.fc = nn.Linear(8*kc*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(8*kc)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = F.relu(self.bn(out), inplace=True)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class GrayPreActResNet(nn.Module):
    def __init__(self, in_channels=1, n_class=3, model_type="resnet50"):
        super(GrayPreActResNet, self).__init__()

        self.model = PreActResNet(PreActBottleneck, [3,4,6,3],n_class)  #TODO Fix, currently it is not resnet50

        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.model.fc = nn.Linear(self.input_features, n_class)
        #self.model.fc = nn.Sequential(nn.Linear(self.input_features, n_class))#,
                                 #,
                                 #nn.Dropout(0.2),
                                 #nn.Linear(256, n_class))
        
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')        
        
    def forward(self, x):
        return self.model(x)