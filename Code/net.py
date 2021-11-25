import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out      = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out      = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out      = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out+residual, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes*4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torchvision.models.resnet50(True).state_dict(), strict=False)


class MCA(nn.Module):
    def __init__(self, in_channel_left, in_channel_right):
        super(MCA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_right, 256, kernel_size=1, stride=1, padding=0)

        self.conv13 = nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv31 = nn.Conv2d(256, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)
        left = F.relu(self.bn1(self.conv2(left)), inplace=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = F.relu(self.conv31(down), inplace=True)
        down = F.relu(self.conv13(down), inplace=True)
        down = torch.sigmoid(down.mean(dim=(2, 3), keepdim=True))
        return left * down

    def initialize(self):
        weight_init(self)


class CFF(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(CFF, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_right, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256 * 3, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True)  # 256 channels

        down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        right = F.interpolate(right, size=left.size()[2:], mode='bilinear')

        x = left * down # l*h
        y = left * right # l*c
        z = right * down # h*c
        out = torch.cat([x, y, z], dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

    def initialize(self):
        weight_init(self)


class SR(nn.Module):
    def __init__(self, in_channel):
        super(SR, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]
        return F.relu(w * out1 + b, inplace=True)

    def initialize(self):
        weight_init(self)


class ACFFNet(nn.Module):
    def __init__(self, cfg):
        super(ACFFNet, self).__init__()
        self.cfg     = cfg
        self.bkbone  = ResNet()
        self.ca55 = MCA(2048, 2048)

        self.fam45 = CFF(1024,  256, 256)
        self.fam34 = CFF(512,  256, 256)
        self.fam23 = CFF(256,  256, 256)

        self.srm5 = SR(256)
        self.srm4 = SR(256)
        self.srm3 = SR(256)
        self.srm2 = SR(256)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5_ = self.bkbone(x)
        out5 = self.ca55(out5_, out5_)
        out5 = self.srm5(out5)
        out4 = self.srm4(self.fam45(out4, out5, out5))
        out3 = self.srm3(self.fam34(out3, out4, out5))
        out2 = self.srm2(self.fam23(out2, out3, out5))

        out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode='bilinear')
        out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode='bilinear')
        out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode='bilinear')
        out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode='bilinear')

        return out2, out3, out4, out5

    def initialize(self):
        if self.cfg.snapshot:
            try:
                print("load params")
                self.load_state_dict(torch.load(self.cfg.snapshot), strict=True)
            except:
                print("Warning: please check the snapshot file:", self.cfg.snapshot)
                pass
        else:
            weight_init(self)


if __name__ == '__main__':
    import pandas as pd
    x = torch.rand(1, 3, 320, 320)
    cag = pd.Series({'snapshot': False})
    f4 = ACFFNet(cag)
    f4.forward(x)
    total_params = sum(p.numel() for p in f4.parameters())
    print('total params : ', total_params)
