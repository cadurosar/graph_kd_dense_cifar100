import argparse

import torch
import torch.nn as nn
import utils
import os


def main():
    trainloader, testloader, clean_trainloader = utils.load_data(128)
    for filename in sorted(os.listdir('checkpoint')):
     try:
        print(filename)
        file = "checkpoint/{}".format(filename)
        model = torch.load(file)["net"].module
        utils.test(model,testloader, "cuda", "no",show="error")
     except:
        continue
if __name__ == "__main__":
    main()
