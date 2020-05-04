'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
import models
import models.densenet
from utils import progress_bar, load_data, train, test
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import argparse

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--depth', default=100, type=int, help='')
    parser.add_argument('--width', default=12, type=int, help='')
    parser.add_argument('--mult', default=4, type=float, help='')
    parser.add_argument('--seed', default=0, type=int, help='')
    
    args = parser.parse_args()
    save = "Densenet{}-{}-{}_{}".format(args.depth,args.width,args.mult,args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader, clean_trainloader = load_data(64)
    net = models.densenet.densenet_cifar(n=args.depth,growth_rate=args.width,bottle_size=args.mult,num_classes=100)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
#    scheduler = ReduceLROnPlateau(optimizer, factor=0.1,patience=5,threshold=1e-4,verbose=True)
    for epoch in range(300):
        print('Epoch: %d' % epoch)
        train(net,trainloader,scheduler, device, optimizer)
        test(net,testloader, device, save_name=save)
#        lr = optimizer.param_groups[0]['lr']
#        if lr <= 0.000001:
#            break
    for epoch in range(5):
        print('Epoch: %d' % epoch)
        train(net,clean_trainloader,scheduler, device, optimizer)
        test(net,testloader, device, save_name="Finetune"+save)
if __name__ == "__main__":
    main()



