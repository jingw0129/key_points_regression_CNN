import argparse
import json
import os
import numpy as np
import torch

import shutil
import time
import datetime
from region_detection import *

from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from config import *
from dataset import *
from lane_net import *

from utils.transforms.transforms import *
from utils.transforms.data_argumentation import *

def parse_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--train_data', type=str, default='//DESKTOP-3DNOAGH/traversable_region_detection/train.csv')
    parse.add_argument('--exp_dir', type= str, default='./runs/')
    args = parse.parse_args()
    return args

args = parse_arg()
exp_dir = args.exp_dir


#Rotation(2),
SAVE_FOLDER = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S").replace(':', '-')
if os.path.exists(exp_dir + str(SAVE_FOLDER)) is False:
    os.makedirs(exp_dir + str(SAVE_FOLDER))

print(SAVE_FOLDER)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)


# --------------------------------------------------------------------------------------------------------------------

transform_train = Compose(Resize(224), Darkness(5), Rotation(2), ToTensor(), Normalize(mean=mean, std=std))
train_dataset = MyData(Dataset_Path['my_image'], "train", transform_train)
train_loader = DataLoader(train_dataset, collate_fn=train_dataset.collate, batch_size=8, shuffle=True)

transform_val = Compose(Resize(224), Darkness(5), Rotation(2), ToTensor(), Normalize(mean=mean, std=std))
val_dataset = MyData(Val_Path['val_image'], "val", transform_val)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=val_dataset.collate, num_workers=0)
# --------------------------------------------------------------------------------------------------------------------
net = LaneNet(pretrained = False)
net = net.to(device)
# net = torch.nn.DataParallel(net)

for name, para in net.named_parameters():
    para.requires_grad = True
    # print(name, para)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = 0.001, weight_decay=0)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


best_val_loss = 1e6

criterion = nn.MSELoss()

def train(epoch):
    net.train()
    print("Train Epoch: {}".format(epoch))
    for e in range(epoch):
        print(e)
        print()
        for batch_idx, sample in enumerate(train_loader):
            img = sample['img'].to(device)
            label = sample['segLabel'].to(device)
            print(len(img), len(label))

            optimizer.zero_grad()
            output = net(img)
            # print(output,label)
            # loss does not converge when being wrappepd in variable
            # loss = Variable(torch.tensor(ems(output, label)), requires_grad=True)
            loss = criterion(output, label)
            if isinstance(net, torch.nn.DataParallel):
                loss = loss.sum()

            print(loss)
            loss.backward()
            optimizer.step()


        if e % 5 == 0:
            save_dict = {
                "epoch": e,
                "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
                "optim": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()
            }

        torch.save(save_dict, os.path.join(exp_dir, SAVE_FOLDER+'/epoch-{0}-loss.pth'.format(e)))

    lr_scheduler.step()
    print("------------------------\n")

# ____________________________________________________________________________________________________________________________________________________________



def val(epoch):
    global best_val_loss
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for id, sample in enumerate(val_loader):
            img = sample['img'].to(device)
            label = sample['segLabel'].to(device)
            output = net(img)
            loss = criterion(output, label)

            if isinstance(net, torch.nn.DataParallel):
                loss = loss.sum()
            print(loss)
            val_loss += loss.item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_name = os.path.join(exp_dir, SAVE_FOLDER+'/epoch-{0}-loss.pth'.format(e))

        copy_name = os.path.join(exp_dir, 'val_best.pth')
        shutil.copyfile(save_name, copy_name)
# ____________________________________________________________________________________________________________________________________________________________

def main():

    global best_val_loss
    epoch = 1000
    train(epoch)


if __name__ == '__main__':
    main()
