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
    parser.add_argument('--teacher_depth', default=100, type=int, help='')
    parser.add_argument('--teacher_width', default=12, type=int, help='')
    parser.add_argument('--teacher_mult', default=4.0, type=float, help='')
    parser.add_argument('--depth', default=100, type=int, help='')
    parser.add_argument('--width', default=11, type=int, help='')
    parser.add_argument('--mult', default=4.0, type=float, help='')
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
    save = "L2Loss_{}-{}-{}_teaches_{}-{}-{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        args.teacher_depth, args.teacher_width, args.teacher_mult, args.depth, args.width, args.mult,args.hkd,args.temp, args.gkd, args.p,args.k,args.pool3,args.intra_only,args.inter_only,args.seed)
    file = "checkpoint/L2Loss_{}-{}-{}_teaches_{}-{}-{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(
        100, 12, 4.0, 100, 11, 4.0,16.0,4.0, 25.0, 1,128,False,False,False,0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader, clean_trainloader = load_data(128)
    teacher = torch.load(file)["net"].module
    teacher = teacher.to(device)
#    net = models.nearmobile.NearMobileNetStart(depth=args.depth,width=args.width,expansion=args.mult,num_classes=10)
    net = models.densenet.densenet_cifar(n=args.depth,growth_rate=args.width,bottle_size=args.mult,num_classes=10)
    net = net.to(device)
#    net.bn = teacher.bn
#    net.linear = teacher.linear
#    net.conv1 = teacher.conv1
#    net.trans1 = copy.deepcopy(teacher.trans1)
#    net.trans2 = copy.deepcopy(teacher.trans2)
#    net.trans3 = copy.deepcopy(teacher.trans3)
    parameters = list()
#    ignore = [net.bn, net.linear, net.conv1]
    ignore = list()
    for a in net.parameters():
        for b in ignore:
            if a is b.weight or a is b.bias:
                print(a.size())
                continue
        parameters.append(a)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
#    optimizer = optim.Adam(parameters, lr=0.001)
#    optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9,nesterov=True,weight_decay=1e-4)
#    scheduler = ReduceLROnPlateau(optimizer, factor=0.2,patience=5,threshold=1e-4,verbose=True)
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
    predicted_teacher, labels = test(teacher,clean_trainloader, device, save_name="no")
    predicted_teacher, labels = test(teacher,trainloader, device, save_name="no")
    predicted_teacher, labels = test(teacher,testloader, device, save_name="no")
    for epoch in range(300):
        print('Epoch: %d' % epoch)
        train(net,trainloader,scheduler, device, optimizer,teacher=teacher,lambda_hkd=args.hkd,lambda_gkd=args.gkd,temp=args.temp,classes=10,power=args.p,pool3_only=args.pool3,k=args.k,intra_only=args.intra_only,inter_only=args.inter_only)
        predicted_student, labels = test(net,testloader, device, save_name=save)
        consistency = 100*(predicted_teacher == predicted_student).float()
        teacher_true = predicted_teacher == labels
        teacher_false = predicted_teacher != labels
        consistency_true = consistency[teacher_true].mean()
        consistency_false = consistency[teacher_false].mean()
        consistency = consistency.mean()
        print("Consistency: {:.2f} (T: {:.2f}, F: {:.2f})%".format(consistency, consistency_true, consistency_false))
        lr = optimizer.param_groups[0]['lr']
        if lr <= 0.000001:
            break


if __name__ == "__main__":
    main()



