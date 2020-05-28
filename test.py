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

from config import *
from dataset import *
from lane_net import *
from utils.transforms.transforms import *
from utils.transforms.data_argumentation import *



def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=str, default='runs/epoch0-loss.pth')
    parse.add_argument('--save_dir', type=str, default='result_image/')
    parse.add_argument('--test_data', type= str, default='//DESKTOP-3DNOAGH/traversable_region_detection/test.csv')
    args = parse.parse_args()
    return args

args = parse_args()

test_dataset = args.test_data
with open(test_dataset,'r') as f:
    lines = csv.reader(f)
    coors = list(lines)

test_labels = [list(map(lambda sub: int(''.join([ele for ele in sub if ele.isnumeric()])), c[1:])) for c in coors]
# print(test_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#------load data--------------------------
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform_test = Compose(Resize(224), Darkness(5), Rotation(2), ToTensor(), Normalize(mean=mean, std=std))

test_dataset = MyData(Test_Path['test'], "test", transform_test)
test_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate, batch_size=8, shuffle=True)

#------------------------------------------

net = LaneNet()

save_dict = args.model

check_point = torch.load(save_dict, map_location='cpu')
net.load_state_dict(check_point, strict=False)
net.to(device)
for name, para in net.named_parameters():
    para.requires_grad = True
    # print(name, para)
net.eval()
#
# #-----------------------------------------------------
#
out_path = args.save_dir
ratio_x = 224/1280
ratio_y = 224/720
with torch.no_grad():
    for batch_idx, sample in enumerate(test_loader):
        img = torch.tensor(sample['img']).to(device)
        image_name = sample['img_name']

        out_put = np.array(net(img).cpu())
        # print(out_put)
        for output_ in out_put:
            output = output_
            print('output', output)
            output[::2] = [l_o * 640 + 640 for l_o in output[::2]]
            output[1::2] = [l_e * 360 + 360 for l_e in output[1::2]]
            # out_put[::2] = [(l_o * 112+112)/ratio_x for l_o in out_put[::2]]
            # out_put[1::2] = [(l_e * 112+112)/ratio_y for l_e in out_put[1::2]]
            print('after mapping', output)

            points = [tuple(output[i:i + 2]) for i in range(0, len(output), 2)]
            # print(points)
            seg_img = np.zeros((720, 1280, 3))
            cv2.line(seg_img, points[0], points[1], (255, 255, 255), 15 // 2)
            cv2.line(seg_img, points[0], points[2], (200, 200, 200), 15 // 2)
            cv2.imshow('result', seg_img)
            cv2.waitKey(0)

