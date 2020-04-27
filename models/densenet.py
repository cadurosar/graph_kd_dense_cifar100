'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

groups = False

def set_groups(new_groups):
    global groups
    groups = new_groups

def get_groups():
    global groups
    return groups



class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.GroupNorm(nChannels,nChannels,affine=True)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, bottle_size=4):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.Sequential()
        self.bn1 = nn.BatchNorm2d(in_planes,affine=True)
        size = max(1,int(bottle_size*growth_rate))
        self.conv1 = nn.Conv2d(in_planes, size, kernel_size=1, bias=False)
        #self.bn2 = nn.Sequential()
        self.bn2 = nn.BatchNorm2d(size,affine=True)
        groups = get_groups()
        self.relu = nn.ReLU(inplace=True)
        if groups:
            self.conv2 = nn.Conv2d(size, growth_rate, kernel_size=3, padding=1, bias=False, groups=min(size,growth_rate))
        else:
            self.conv2 = nn.Conv2d(size, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out2 = torch.cat([out,x], 1)
        return out2


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, last=False):
        super(Transition, self).__init__()
        #self.bn = nn.Sequential()#
        self.bn = nn.BatchNorm2d(in_planes,affine=True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if last:
            self.avg_pool2d = nn.AvgPool2d(8)
        else:
            self.avg_pool2d = nn.AvgPool2d(2)

    def forward(self, x):
        relu = self.relu(self.bn(x))
        out = self.conv(relu)
        out = self.avg_pool2d(out)
        return out, relu


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, bottle_size=4):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.relu = nn.ReLU(inplace=True)
        num_planes = 24
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], bottle_size=bottle_size)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
#        out_planes = int(math.floor(num_planes))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], bottle_size=bottle_size)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
#        out_planes = int(math.floor(num_planes))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], bottle_size=bottle_size)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
#        out_planes = int(math.floor(num_planes))
        self.trans3 = Transition(num_planes, out_planes,last=True)
        num_planes = out_planes

#        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], bottle_size=bottle_size)
#        num_planes += nblocks[3]*growth_rate
#        out_planes = num_planes
#        out_planes = int(math.floor(num_planes))
#        num_planes = out_planes


        #self.bn = nn.Sequential()
#        self.bn = nn.BatchNorm2d(num_planes,affine=True)
        self.linear = nn.Linear(num_planes, num_classes)
        self.avg_pool2d = nn.AvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)



    def _make_dense_layers(self, block, in_planes, nblock, bottle_size):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, bottle_size=bottle_size))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out11 = self.dense1(out1)
        out2, relu1 = self.trans1(out11)
        out22 = self.dense2(out2)
        out3, relu2 = self.trans2(out22)
        out33 = self.dense3(out3)
        out, relu3 = self.trans3(out33)
        try:
            out = self.dense4(out)
            out = self.avg_pool2d(self.relu(self.bn(out)))
        except:
            pass
        out4 = out.view(out.size(0), -1)
        out = self.linear(out4)
        return out, [relu1,relu2,relu3]

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar(n=196,growth_rate=8,reduction=0.5, groups=False, bottle_size=4, num_classes=10): #n=196 growth_rate=8
    n = (n-4)//3
    n = n//2
    set_groups(groups)
    return DenseNet(Bottleneck, [n,n,n], growth_rate=growth_rate, reduction=reduction, bottle_size=bottle_size, num_classes=num_classes)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
