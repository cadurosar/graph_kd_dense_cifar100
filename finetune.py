'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
import models
import models.resnet
import models.nearmobile
from utils import progress_bar, load_data, train, test
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import copy

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--teacher_depth', default=18, type=int, help='')
    parser.add_argument('--teacher_width', default=8, type=int, help='')
    parser.add_argument('--teacher_mult', default=1.0, type=float, help='')
    parser.add_argument('--depth', default=18, type=int, help='')
    parser.add_argument('--width', default=8, type=int, help='')
    parser.add_argument('--mult', default=1.0, type=float, help='')
    parser.add_argument('--hkd', default=0., type=float, help='')
    parser.add_argument('--temp', default=0., type=float, help='')
    parser.add_argument('--gkd', default=0., type=float, help='')
    parser.add_argument('--p', default=1, type=int, help='')
    parser.add_argument('--seed', default=0, type=int, help='')
    parser.add_argument('--k', default=128, type=int, help='')
    parser.add_argument('--pool3', action='store_true', help='')
    parser.add_argument('--intra_only', action='store_true', help='')
    parser.add_argument('--inter_only', action='store_true', help='')
    args = parser.parse_args()
    save = "FineTune{}-{}-{}_0.pth".format(args.teacher_depth,args.teacher_width,args.teacher_mult)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader, clean_trainloader = load_data(128)
    file = "checkpoint/NearMobileNet{}-{}-{}_0.pth".format(args.teacher_depth,args.teacher_width,args.teacher_mult)
    teacher = torch.load(file)["net"].module
    teacher = teacher.to(device)
    parameters = list()
    if device == 'cuda':
        teacher = torch.nn.DataParallel(teacher)
        cudnn.benchmark = True
#    optimizer = optim.AdamW(teacher.parameters(), lr=0.00001, weight_decay=5e-4)
    optimizer = optim.SGD(teacher.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[55, 120, 160], gamma=0.2)
    predicted_student, labels = test(teacher,testloader, device, save_name=save)
    for epoch in range(50):
        print('Epoch: %d' % epoch)
        train(teacher,clean_trainloader,scheduler, device, optimizer)
        predicted_student, labels = test(teacher,testloader, device, save_name=save)
if __name__ == "__main__":
    main()



