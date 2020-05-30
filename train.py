import argparse
import cv2
import os
import numpy as np
import torch

import shutil
import time
import datetime
from region_detection import *


import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import *
from lane_net import *

torch.manual_seed(42)
def parse_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--train_data', type=str, default='//DESKTOP-3DNOAGH/traversable_region_detection/train.csv')
    parse.add_argument('--exp_dir', type= str, default='./runs/')
    args = parse.parse_args()
    return args

args = parse_arg()
exp_dir = args.exp_dir


SAVE_FOLDER = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S").replace(':', '-')
if os.path.exists(exp_dir + str(SAVE_FOLDER)) is False:
    os.makedirs(exp_dir + str(SAVE_FOLDER))

print(SAVE_FOLDER)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# mean=(0.485, 0.456, 0.406)
# std=(0.229, 0.224, 0.225)


# --------------------------------------------------------------------------------------------------------------------

# transform_train = Compose(Resize(224), Darkness(5), Rotation(2), ToTensor(), Normalize(mean=mean, std=std))

train_dataset = MyData(train_coors, train_img_path)
train_loader = DataLoader(train_dataset, collate_fn=train_dataset.collate, batch_size=4, shuffle=True)

# transform_val = Compose(Resize(224), Darkness(5), Rotation(2), ToTensor(), Normalize(mean=mean, std=std))

val_dataset = MyData(val_coors, val_img_path)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=val_dataset.collate, num_workers=0)
# --------------------------------------------------------------------------------------------------------------------
# net = LaneNet(pretrained = False)
net = LaneNet()
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
    torch.manual_seed(42)
    print("Train Epoch: {}".format(epoch))
    for e in range(epoch):
        print(e)
        train_loss = 0
        for batch_idx, sample in enumerate(train_loader):
            img = sample['img'].to(device)
            # print(sample['segLabel'])
            label = torch.FloatTensor(sample['segLabel']).to(device)
            print(len(img), len(label))
            print(img.shape, label.shape)
            optimizer.zero_grad()
            output = net(img)
            print(output[0],label[0])

            # loss does not converge when being wrappepd in variable
            # loss = Variable(torch.tensor(ems(output, label)), requires_grad=True)
            loss = criterion(output, label)
            if isinstance(net, torch.nn.DataParallel):
                loss = loss.sum()

            train_loss += loss.item()
            print("Epoch {} - Training loss: {}  loss: {}".format(e, train_loss / len(train_loader), loss))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        if e % 2 == 0:
            save_dict = {
                "epoch": e,
                "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
                "optim": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()
            }

        torch.save(save_dict, os.path.join(exp_dir, SAVE_FOLDER+'/epoch-{0}-loss.pth'.format(e)))


    print("------------------------\n")

# ____________________________________________________________________________________________________________________________________________________________



def val(epoch):
    global best_val_loss
    net.eval()
    torch.manual_seed(42)
    val_loss = 0
    correct_count = 0
    total = 0
    for e in range(epoch):
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
                # --------------------------------------
                output_ = np.array(net(img).cpu())
                label_ = np.array(sample['segLabel'])
                for i in range(len(label_)):
                    if np.mean(np.sum(label_[i] - output_[i])) in (-0.0001, 0.0001):
                        correct_count +=1
                    total +=1
        print("Number Of Images Tested =", total)
        print("\nModel Accuracy =", (correct_count / total))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_name = os.path.join(exp_dir, SAVE_FOLDER+'/epoch-{0}-loss.pth'.format(e))

            copy_name = os.path.join(exp_dir, 'val_best.pth')
            shutil.copyfile(save_name, copy_name)
# ____________________________________________________________________________________________________________________________________________________________
test_dataset = MyData(test_coors, test_img_path)
test_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate, batch_size=4, shuffle=True)

#------------------------------------------
def test():
    net.eval()
    torch.manual_seed(42)
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            img = torch.tensor(sample['img']).to(device)

            label = np.array(sample['segLabel'])
            out_put = np.array(net(img).cpu())

            print(label[-1:], out_put[-1:])

            img = np.array(img.cpu())

            for i in range(len(out_put)):
                output = out_put[i]

                output[::2] = [int((l_o + 0.5) * 224) for l_o in output[::2]]
                output[1::2] = [int((l_e + 0.5) * 224) for l_e in output[1::2]]

                # print('after mapping', output)

                points = [tuple(output[i:i + 2]) for i in range(0, len(output), 2)]
                print(points)
                img_ = img[i]
                img_ = cv2.UMat(img_.transpose(1, 2, 0))

                cv2.line(img_, points[0], points[1], (124, 0, 255), 2)
                cv2.line(img_, points[0], points[2], (124, 0, 200), 2)
                # img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                cv2.imshow('result', img_)
                cv2.waitKey(0)

# ---------------------------------------------------------------------------------
def main():

    global best_val_loss
    epoch = 1000
    train(epoch)
    val(epoch)
    # test()

if __name__ == '__main__':
    main()
