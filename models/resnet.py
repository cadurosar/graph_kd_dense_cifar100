import torch
import torch.nn as nn
import torch.nn.functional as F


class RemoveChannelMeanStd(torch.nn.Module):
    def forward(self, x):
        x2 = x.view(x.size(0),x.size(1),-1)
        mean = x2.mean(dim=2).view(x.size(0),x.size(1),1,1)
        std = x2.std(dim=2).view(x.size(0),x.size(1),1,1)
        return (x-mean)/std



class Block(nn.Module):

    def __init__(self, in_planes, planes, stride=1, groups=False):
        super(Block, self).__init__()
        self.bn1 = RemoveChannelMeanStd()
        if not groups:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, groups=min(in_planes,planes))
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True, groups=planes)

        self.bn2 = RemoveChannelMeanStd()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
                RemoveChannelMeanStd()
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        relu1 = F.relu(out)
        out = self.conv2(relu1)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out += shortcut
        out = F.relu(out)
        return relu1, out


class WideResNet(nn.Module):
    def __init__(self, block, num_blocks, width, num_classes=10, groups=False):
        super(WideResNet, self).__init__()
        self.first_block = 64
        self.width = width
        self.in_planes = self.first_block

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = RemoveChannelMeanStd()
        self.layer1 = self._make_layer(block, int(self.width), num_blocks[0], stride=1, groups=groups)
        self.layer2 = self._make_layer(block, int(self.width*2), num_blocks[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(block, int(self.width*4), num_blocks[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(block, int(self.width*8), num_blocks[3], stride=2, groups=groups)

        self.linear = nn.Linear(int(self.width*8), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, groups):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, groups=groups))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def input_to_one(self,x):
        out = self.conv1(x)
        out = self.bn(out)
        rrelu1 = F.relu(out)
        relu1, out = self.layer1(rrelu1)
        return rrelu1, relu1, out

    def three_to_four(self,out):
        relu1, out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return relu1, out

    def forward(self, x):
        rrelu1, relu1, block1 = self.input_to_one(x)
        relu2, block2 = self.layer2(block1)
        relu3, block3 = self.layer3(block2)
        relu4, block4 = self.three_to_four(block3)                
        out = self.linear(block4)
        return out, [rrelu1, relu1, relu2, relu3, relu4, block1, block2, block3, block4]


def WideResNetStart(depth=18,width=64,num_classes=100, groups=False):
    n = (depth-2) //8
    print(n)
    return WideResNet(Block, [n,n,n,n],width=width,num_classes=num_classes, groups=groups)

# test()
