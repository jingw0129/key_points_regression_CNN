import cv2
import csv
import torch
import numpy as np
import argparse

# cv2.imread('//DESKTOP-ASB277M/Data_3D/Training/POS/cas101/20180816_21.png')

def parse_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--train_data', type=str, default='//DESKTOP-3DNOAGH/traversable_region_detection/train.csv')

    args = parse.parse_args()
    return args

args = parse_arg()

train_data = args.train_data

def label_normal(label):
    # label[::2] = [(l_o-640)/640 for l_o in label[::2]]
    # label[1::2] = [(l_e-360)/360 for l_e in label[1::2]]

    # label[::2] = [(l_o - 1280) / 1280 for l_o in label[::2]]
    # label[1::2] = [(l_e - 720) / 720 for l_e in label[1::2]]
    ratio_x = 224/1280
    ratio_y = 224/720
    # label[::2] = [(l_o*ratio_x - 112) / 112 for l_o in label[::2]]
    # label[1::2] = [(l_e*ratio_y - 112)/112 for l_e in label[1::2]]

    label[::2] = [l_o * ratio_x/224-0.5 for l_o in label[::2]]
    label[1::2] = [l_e * ratio_y/224-0.5 for l_e in label[1::2]]
    label = (torch.from_numpy(np.array(label)).type(torch.float)).type(torch.float)
    return label

def get_data(data_dir):
    with open(data_dir,'r') as f:
        lines = csv.reader(f)
        coors = list(lines)

    coors_list =[]
    image_path = []
    for c in coors:
        # image.append(cv2.imread(c[0]))

        res = list(map(lambda sub: int(''.join([ele for ele in sub if ele.isnumeric()])), c[1:]))
        # segLabel = torch.from_numpy(np.array([res[i:i+2] for i in range(0, len(res),2)])).type(torch.long)
        # coors_list.append([res[i:i+2] for i in range(0, len(res),2)])
        res = label_normal(res)
        coors_list.append(res)
        image_path.append(c[0])

    return coors_list, image_path

train_coors = get_data(train_data)[0]
train_img_path = get_data(train_data)[1]

val_coors = get_data('//DESKTOP-3DNOAGH/traversable_region_detection/val.csv')[0]
val_img_path = get_data('//DESKTOP-3DNOAGH/traversable_region_detection/val.csv')[1]

test_coors = get_data('//DESKTOP-3DNOAGH/traversable_region_detection/test.csv')[0]
test_img_path = get_data('//DESKTOP-3DNOAGH/traversable_region_detection/test.csv')[1]