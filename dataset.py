import json
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
# from region_detection import image_path, coors_list


def augment_data(img, points):
    rows, cols = img.shape
    new_img = np.copy(img)

    # flip the image
    for i in range(224):
        for j in range(112):
            temp = img[i][j]
            new_img[i][j] = img[i][cols - j - 1]
            new_img[i][cols - j - 1] = temp

    # flip the points
    new_points = points
    for i in range(0, 6, 2):
        new_points[i] = -points[i]
    new_points = np.array([np.array(p) for p in new_points])
    return new_img, new_points


class MyData(Dataset):
    """
    image_set is splitted into three partitions: train, val, test.
    train includes label_data_0313.json, label_data_0601.json
    val includes label_data_0531.json
    test includes test_label.json
    """

    def __init__(self, coors_list, image_path, transforms=None):

        super(MyData, self).__init__()
        self.segLabel_list = coors_list
        self.image_path = image_path
        self.transforms = transforms

    def __getitem__(self, idx):
        img = cv2.imread(self.image_path[idx], cv2.IMREAD_UNCHANGED)

        # print(img.shape)
        # img = img.transpose(2, 0, 1)
        img = cv2.resize(img, (224,224))
        dtype = torch.float

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.from_numpy(img).type(dtype) / 255.

        # print(type(self.segLabel_list))

        # aug_img = augment_data(img, self.segLabel_list)[0]
        # aug_point = augment_data(img, self.segLabel_list)[1]
        # # print(img.shape)
        img = img.reshape(1, 224, 224)
        sample = {'img': img,
                  'segLabel': self.segLabel_list[idx],
                  'img_name': self.image_path[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.image_path)



    @staticmethod
    def collate(batch):



        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]

        samples = {'img': img,
                   'segLabel': segLabel,
                   'img_name': [x['img_name'] for x in batch]}
        # print(samples)
        return samples