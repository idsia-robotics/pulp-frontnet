# ##############################################################################
#  Copyright (c) 2019–2021 IDSIA, USI and SUPSI, Switzerland                   #
#                2019-2021 University of Bologna, Italy                        #
#                2019-2021 ETH Zürich, Switzerland                             #
#  All rights reserved.                                                        #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#      http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
#                                                                              #
#  File: Frontnet.py                                                           #
# ##############################################################################

import numpy as np
import torch.nn as nn
from torchsummary import summary

np.set_printoptions(threshold=np.inf, suppress=True)


def FrontnetModel(h=96, w=160, c=32, fc_nodes=1920):
    """
    This is the model referred in the paper as Frontnet

    Possible configurations are:
        input size 160x96 (w,h) - c=32, fc_nodes=1920 (default)
        input size 160x96 (w,h) - c=16, fc_nodes=960
        input size  80x48 (w,h) - c=32, fc_nodes=768
    Where c is the number of the channels in the first convolution layer
    The model in the example is configured to handle gray-scale input
    """

    model = Frontnet(ConvBlock, [1, 1, 1], c=c, fc_nodes=fc_nodes)
    summary(model, (1, h, w), device='cpu')

    return model


FrontnetModel.configs = {
    "160x32": dict(h=96, w=160, c=32, fc_nodes=1920),
    "160x16": dict(h=96, w=160, c=16, fc_nodes=960),
    "80x32":  dict(h=48, w=80,  c=32, fc_nodes=768),
}


class Frontnet(nn.Module):
    def __init__(self, block, layers, w=160, h=96, c=32, fc_nodes=1920):
        super(Frontnet, self).__init__()

        self.name = "Frontnet"

        self.inplanes = c
        self.width = w
        self.height = h
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d

        self.groups = 1
        self.base_width = 64

        self.conv = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)

        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = ConvBlock(self.inplanes, self.inplanes, stride=2)
        self.layer2 = ConvBlock(self.inplanes, self.inplanes*2, stride=2)
        self.layer3 = ConvBlock(self.inplanes*2, self.inplanes*4, stride=2)

        self.dropout = nn.Dropout()
        self.fc = nn.Linear(fc_nodes, 4)

    def forward(self, x):
        conv5x5 = self.conv(x)
        btn = self.bn(conv5x5)
        relu1 = self.relu1(btn)
        max_pool = self.maxpool(relu1)

        l1 = self.layer1(max_pool)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        out = l3.flatten(1)

        out = self.dropout(out)
        out = self.fc(out)
        x = out[:, 0]
        y = out[:, 1]
        z = out[:, 2]
        phi = out[:, 3]
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        z = z.unsqueeze(1)
        phi = phi.unsqueeze(1)

        return [x, y, z, phi]


class ConvBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out
