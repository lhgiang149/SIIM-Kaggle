import torch
import torch.nn as nn

from utils import *

base = {'json_path': 'info.json'}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False, base_width=64):
        super(BasicBlock, self).__init__()
        self.expansion = 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def forward(self, x_base):
        x = self.conv1(x_base)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.bn2(x)
        if self.downsample:
            temp_model = nn.Sequential(
                conv1x1(self.planes, self.planes*2, 2),
                nn.BatchNorm2d(self.planes *2)
            )
            x_base = temp_model(x_base)
        x += x_base
        x = self.relu(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False, groups=1,
                 base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def forward(self, x_base):
        x = self.conv1(x_base)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            temp_model = nn.Sequential(
                conv1x1(self.planes, self.planes*4 , 2),
                nn.BatchNorm2d(self.planes * 4)
            )
            x_base = temp_model(x_base)
        x += x_base
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, name, num_classes=1000):
        super(ResNet, self).__init__()
        self.__dict__.update(base)
        model_info = loadJson(self.json_path)
        if name not in model_info.keys():
            raise NameError('Wrong name! Only accept \"ResNet18\", \"ResNet34\", \"ResNet50\", \"ResNet101\" or \"ResNet152\"')
        model_info = model_info[name]
        if model_info['Basic']:
            block = BasicBlock
        else:
            block = Bottleneck

        self.inplanes = 64

        residual = model_info['residual']

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = self._residual_block(block, 64, residual[0])
        self.block2 = self._residual_block(block, 128, residual[1], stride=2)
        self.block3 = self._residual_block(block, 256, residual[2], stride=2)
        self.block4 = self._residual_block(block, 512, residual[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _residual_block(self, block, planes, residual_blocks, stride=1, dilate=False):

        layers = []
        

        if stride != 1 or self.inplanes != planes * block.expansion:
            layers.append(block(self.inplanes, planes, stride, downsample = True))
        
        self.inplanes = planes * block.expansion

        for _ in range(residual_blocks-1):
            layers.append(block(self.inplanes, planes, downsample = False))

        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x



    