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
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.do_shortcut = stride != 1 or in_planes != out_planes
        planes = int(expansion * in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
 #       self.bn1 = nn.GroupNorm(1,planes)
 #       self.bn2 = nn.GroupNorm(1,planes)
 #       self.bn3 = nn.GroupNorm(1,out_planes)


#       self.bn1 = nn.GroupNorm(1,planes)
#       self.bn2 = nn.GroupNorm(1,planes)
#       self.bn3 = nn.GroupNorm(1,out_planes)


        self.bn1 = nn.Sequential()
        self.bn2 = nn.Sequential()
        self.bn3 = nn.Sequential()


        self.shortcut = nn.Sequential()
        if self.do_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True),
#                nn.GroupNorm(1,out_planes),
            )
        #self.alpha = nn.Parameter(torch.zeros(1)+1e-8)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
#        out = self.alpha*out + self.shortcut(x) if self.do_shortcut else out + x
        out = out + self.shortcut(x) if self.do_shortcut else out + x
        return F.relu(out)


class NearMobileNet(nn.Module):
    def __init__(self, block, num_blocks, width, num_classes=10, groups=False, expansion=1):
        super(NearMobileNet, self).__init__()
        self.first_block = 64
        self.width = width
        self.in_planes = self.first_block

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=True)
        #self.bn = nn.BatchNorm2d(self.in_planes)
        self.bn = nn.Sequential()        
        self.layer1 = self._make_layer(block, int(self.width), num_blocks[0], stride=1, expansion=expansion)
        self.layer2 = self._make_layer(block, int(self.width*2), num_blocks[1], stride=2, expansion=expansion)
        self.layer3 = self._make_layer(block, int(self.width*4), num_blocks[2], stride=2, expansion=expansion)
        self.layer4 = self._make_layer(block, int(self.width*8), num_blocks[3], stride=2, expansion=expansion)

        self.linear = nn.Linear(int(self.width*8), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, expansion):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, expansion, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def input_to_one(self,x):
        out = self.conv1(x)
        out = self.bn(out)
        rrelu1 = F.relu(out)
        out = self.layer1(rrelu1)
        return rrelu1, out

    def three_to_four(self,out):
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        rrelu1, block1 = self.input_to_one(x)
        block2 = self.layer2(block1)
        block3 = self.layer3(block2)
        block4 = self.three_to_four(block3)                
        out = self.linear(block4)
        return out, [rrelu1,block1,block2,block3,block4]


def NearMobileNetStart(depth=18,width=64,num_classes=100, expansion=1):
    n = (depth-2) //8
    return NearMobileNet(Block, [n,n,n,n],width=width,num_classes=num_classes, expansion=expansion)

# test()
