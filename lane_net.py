import torch
import torch.nn as nn
import torchvision.models as models


class LaneNet(nn.Module):
    def __init__(
            self,
            pretrained = True,

            **kwargs
    ):
        super(LaneNet, self).__init__()
        self.pretrained = pretrained
        self.net_init()

        for p in self.parameters():
            p.requires_grad = True


    def net_init(self):
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features
        # print(self.backbone)
        # ----------------- process backbone -----------------
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')

        # ----------------- additional conv -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size= 3, stride=2,dilation=4, bias=True),
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            nn.Conv2d(1024, 128, 3, stride=2, bias=True),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 32, 3, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 16, 3, stride=2,bias=True),
            nn.BatchNorm2d(16),
            nn.PReLU()
        )

        self.fc1 = nn.Linear(16, 6)

    def forward(self, x, labels = None):

        x = self.backbone(x)
        x = self.layer1(x)
        x = x.view(-1, 16)
        x = self.fc1(x)

        output = x

        return output
