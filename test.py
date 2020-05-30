import csv
import argparse
import json
import os
import numpy as np
import torch

import cv2
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from region_detection import *

from dataset import *
from lane_net import *




def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=str, default='runs/epoch-0-loss.pth')
    parse.add_argument('--save_dir', type=str, default='result_image/')
    parse.add_argument('--test_data', type= str, default='//DESKTOP-3DNOAGH/traversable_region_detection/test.csv')
    args = parse.parse_args()
    return args

args = parse_args()




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#------load data--------------------------
# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)

test_dataset = MyData(test_coors, test_img_path)
test_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate, batch_size=4, shuffle=True)

#------------------------------------------

net = LaneNet()

save_dict = args.model

net.load_state_dict(torch.load(save_dict, map_location='cpu')['net'], strict=False)
net.to(device)



for name, para in net.named_parameters():
    para.requires_grad = True
    # print(name, para)

#-----------------------------------------------------

out_path = args.save_dir
ratio_x = 224/1280
ratio_y = 224/720
with torch.no_grad():
    net.eval()
    torch.manual_seed(42)

    for batch_idx, sample in enumerate(test_loader):
        img = torch.tensor(sample['img']).to(device)
        image_name = sample['img_name']
        label = np.array(sample['segLabel'])
        out_put = np.array(net(img).cpu())
        # out_put = label

        print(label[-1:], out_put[-1:])

        img = np.array(img.cpu())

        for i in range(len(out_put)):
            output = out_put[i]

            # output[::2] = [l_o * 640 + 640 for l_o in output[::2]]
            # output[1::2] = [l_e * 360 + 360 for l_e in output[1::2]]
            # output[::2] = [int((l_o+0.5)*224)/ratio_x for l_o in output[::2]]
            # output[1::2] = [int((l_e+0.5)*224)/ratio_y for l_e in output[1::2]]
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

