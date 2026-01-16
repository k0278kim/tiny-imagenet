import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctypes   # Load the shared library
import matplotlib.pyplot as plt

class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x, batch_shape):
        return self.custom_conv2d(x, self.weight, batch_shape, self.bias, self.stride, self.padding)

    def custom_conv2d(self, input, weight, batch_shape, bias=None, stride=1, padding=0):
        return F.conv2d(input, weight, bias=bias, stride=stride, padding=padding)
    
def encrypt_data(input, batch_shape, stride):
    return input

class Bottleneck(nn.Module):
    '''
    Contains three types of convolutional layers
    conv1-Number of compression channels
    conv2-Extract features
    conv3-extended number of channels
    This structure can better extract features, deepen the network, and reduce the number of network parameters。
    inplanes - in_channels 
    planes = out_channels
    '''

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_custom_conv=False, planes_per_use_custom_planes=4, custom_conv=None):
        super(Bottleneck, self).__init__()

        self.use_custom_conv = use_custom_conv
        self.custom_conv = custom_conv
        self.stride = stride

        if use_custom_conv:
            self.conv1 = CustomConv2D(inplanes, planes, kernel_size=1, stride=stride, bias=False)   # 1x1 conv
            self.bn1 = nn.BatchNorm2d(planes)
            
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            
            self.conv3 = nn.Conv2d(planes, planes * 4 * planes_per_use_custom_planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4 * planes_per_use_custom_planes)
            
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)

            self.bn1 = nn.BatchNorm2d(planes)
            
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride


        
    def forward(self, x):
        residual = x

        enc = None
        batch_shape = [1, x.shape[1], x.shape[2], x.shape[3]]

        if self.use_custom_conv:
            enc = encrypt_data(x, batch_shape, self.stride)
            out = self.conv1(enc, batch_shape)
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)                                                                   # 3x3 conv  
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)                                                                   # 1x1 conv                                              
        out = self.bn3(out)

        if self.downsample is not None:
            if self.custom_conv is not None:
                res = self.custom_conv(enc, batch_shape)
            else:
                res = residual
            residual = self.downsample(res)


        out += residual
        out = self.relu(out)
        return out
 

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, custom_conv_layer_index=1):
        
        self.inplanes = 64
        self.custom_conv_layer_index = custom_conv_layer_index
        super(ResNet, self).__init__()

        self.custom_conv_layer_index = custom_conv_layer_index
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)                    #TODO: original conv 1x1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0], skip_planes=32, layer_index=1, use_custom_planes=16)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, skip_planes=128, layer_index=2, use_custom_planes=64)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, skip_planes=256, layer_index=3, use_custom_planes=128)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, skip_planes=512, layer_index=4, use_custom_planes=256)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, skip_planes, stride=1, layer_index=1, use_custom_planes=16):
        
        downsample = None
        custom_conv = None
        use_custom = (layer_index == self.custom_conv_layer_index)
        
        if stride != 1 or self.inplanes != planes * block.expansion:# block.expansion=4

            if use_custom:
                custom_conv = CustomConv2D(self.inplanes, skip_planes, kernel_size=1, stride=1, bias=False)
                downsample = nn.Sequential(
                    nn.BatchNorm2d(skip_planes),
                    nn.Conv2d(skip_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )
       
        layers = []

        if use_custom:
            layers.append(block(self.inplanes, use_custom_planes, stride, downsample, use_custom_conv=use_custom, planes_per_use_custom_planes=planes // use_custom_planes, custom_conv=custom_conv))
            self.inplanes = use_custom_planes * block.expansion * (planes // use_custom_planes)
        
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, use_custom_conv=use_custom))
            self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x