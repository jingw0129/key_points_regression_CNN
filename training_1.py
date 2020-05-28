import torch

import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
import shutil
import time
import datetime

from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

from train import train_loader, val_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_FOLDER = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S").replace(':', '-')
if os.path.exists('runs/' + str(SAVE_FOLDER)) is False:
    os.makedirs('runs/' + str(SAVE_FOLDER))

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):  # 构造函数
        super(VGG, self).__init__()
        # 网络结构（仅包含卷积层和池化层，不包含分类器）
        self.features = features
        # self.classifier = nn.Sequential(  # 分类器结构
        #     # fc6
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #
        #     # fc7
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #
        #     # fc8
        #     nn.Linear(4096, num_classes),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #
        #     nn.Linear(num_classes,6))
        self.layer1 = nn.Sequential(
            # nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, dilation=4, bias=True),
            # nn.BatchNorm2d(1024),
            # nn.PReLU(),
            nn.Conv2d(512, 128, 3, stride=2, bias=True),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 32, 3, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 16, 3, stride=2, bias=True),
            nn.BatchNorm2d(16),
            nn.PReLU()
        )

        self.fc1 = nn.Linear(16, 6)
        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        x = self.layer1(x)
        x = x.view(-1, 16)
        x = self.fc1(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


# 生成网络每层的信息
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 设定卷积层的输出数量
            conv2d = nn.Conv2d(in_channels, v, 3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  # 返回一个包含了网络结构的时序容器


def vgg16(**kwargs):
    model = VGG(make_layers(cfg, batch_norm=True), **kwargs)
    # model.load_state_dict(torch.load(model_path))
    return model

net = vgg16()
net = net.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
best_val_loss = 1e3
# def getData():  # 定义数据预处理
#     transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])])
#     trainset = tv.datasets.CIFAR10(root='/data/', train=True, transform=transform, download=True)
#     testset = tv.datasets.CIFAR10(root='/data/', train=False, transform=transform, download=True)
#
#
#     train_loader = DataLoader(trainset, batch_size=10, shuffle=True)
#     test_loader = DataLoader(testset, batch_size=10, shuffle=False)
#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     return train_loader, test_loader, classes

def train(epoch):
    net.train()
    print("Train Epoch: {}".format(epoch))
    # train_loader, testset_loader, _ = getData()
    testset_loader = val_loader
    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度清零
        img = sample['img'].to(device)
        label = sample['segLabel'].to(device)
        output = net(img)
        # print(output, label)
        loss = criterion(output, label)
        # print(output, label)
        print(loss)
        with open('loss_record', 'a') as f:
            f.writelines(str(loss) + str(epoch) + '\n')
        loss.backward()
        optimizer.step()


    if epoch % 5 == 0:
        save_dict = {
            "epoch": epoch,
            "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }

        torch.save(save_dict, os.path.join('runs', SAVE_FOLDER,
                                           'epoch{0}-loss.pth'.format(epoch)))
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
            # print(output, label)
            loss = criterion(output, label)
            # print(output, label)
            print(loss)

            if isinstance(net, torch.nn.DataParallel):
                loss = loss.sum()
            print(loss)
            val_loss += loss.item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_name = os.path.join('runs', SAVE_FOLDER,
                                           'epoch{0}-loss.pth'.format(epoch))
        copy_name = os.path.join('runs',  'val_best.pth')
        shutil.copyfile(save_name, copy_name)
#------------------------------------------------------------------------------------------------------

def main():
    # run function train and val
    global best_val_loss

    start_epoch = 0

    for epoch in range(start_epoch, 1000):
        train(epoch)
        if epoch % 5 == 0:
            # print("\nValidation For Experiment: ", exp_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            val(epoch)

if __name__ == '__main__':
    main()
