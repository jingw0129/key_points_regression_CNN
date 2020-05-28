import cv2
import csv
import torch
import numpy as np
import argparse

def parse_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--train_data', type=str, default='//DESKTOP-3DNOAGH/traversable_region_detection/train.csv')

    args = parse.parse_args()
    return args

args = parse_arg()

train_dataset = args.train_data

def label_normal(label):
    label[::2] = [(l_o-640)/640 for l_o in label[::2]]
    label[1::2] = [(l_e-360)/360 for l_e in label[1::2]]

    label = (torch.from_numpy(np.array(label)).type(torch.float)).type(torch.float)
    return label


with open(train_dataset,'r') as f:
    lines = csv.reader(f)
    coors = list(lines)

image = []
coors_list =[]
image_path = []
for c in coors:
    image.append(cv2.imread(c[0]))

    res = list(map(lambda sub: int(''.join([ele for ele in sub if ele.isnumeric()])), c[1:]))
    # segLabel = torch.from_numpy(np.array([res[i:i+2] for i in range(0, len(res),2)])).type(torch.long)
    # coors_list.append([res[i:i+2] for i in range(0, len(res),2)])
    res = label_normal(res)
    coors_list.append(res)
    image_path.append(c[0])

img_label = list(zip(image, coors_list, image_path))


