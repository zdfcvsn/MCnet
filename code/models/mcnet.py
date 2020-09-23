import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet
from torchvision import models
from models import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain

class PicanetL(nn.Module):
    def __init__(self, in_channel):
        super(PicanetL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)

    def forward(self, x):
        # x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        # print("Before unfold", x.shape)
        x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        # print("After unfold", x.shape)
        x = x.reshape(size[0], size[1], size[2] * size[3], -1)
        # print(x.shape, kernel.shape)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x

class MCNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='densenet201', pretrained=True, use_aux=True,
                 freeze_bn=False, **_):
        super(MCNet, self).__init__()
        self.use_aux = use_aux
        model = getattr(models, backbone)(pretrained)
        m_out_sz = model.classifier.in_features
        aux_out_sz = model.features.transition3.conv.out_channels

        if not pretrained or in_channels != 3:
            # If we're training from scratch, better to use 3x3 convs
            block0 = [nn.Conv2d(in_channels, 64, 3, stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
            block0.extend(
                [nn.Conv2d(64, 64, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)] * 2
            )
            self.block0 = nn.Sequential(
                *block0,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.block0)
        else:
            self.block0 = nn.Sequential(*list(model.features.children())[:4])

        self.block1 = model.features.denseblock1
        self.block2 = model.features.denseblock2
        self.block3 = model.features.denseblock3
        self.block4 = model.features.denseblock4

        self.conv1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv3 = nn.Conv2d(896, num_classes, kernel_size=1)
        num_filters = 7
        self.ca1 = PicanetL(2)
        self.ca2 = PicanetL(2)
        self.ca3 = PicanetL(2)
        self.covclass = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.covclass1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.covclass2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.transition1 = model.features.transition1
        # No pooling
        self.transition2 = nn.Sequential(
            *list(model.features.transition2.children())[:-1])
        self.transition3 = nn.Sequential(
            *list(model.features.transition3.children())[:-1])

        for n, m in self.block3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding = (2, 2), (2, 2)
        for n, m in self.block4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding = (4, 4), (4, 4)

        self.master_branch = nn.Sequential(
            _CAModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=nn.BatchNorm2d),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(aux_out_sz, m_out_sz // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(m_out_sz // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn: self.freeze_bn()

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        x0 = self.block0(x)
        x1 = self.block1(x0)
        x1 = self.transition1(x1)

        x2 = self.block2(x1)
        x2 = self.transition2(x2)
        a1 = self.conv1(x1)
        a2 = self.conv2(x2)
        s1 = self.ca1(a1)
        s2 = self.ca2(a2)

        x3 = self.block3(x2)
        x3 = self.transition3(x3)
        a3 = self.conv3(x3)
        s3 = self.ca3(a3)

        x4 = self.block4(x3)
        output = self.master_branch(x4)

        ss1 = s3+s2
        ss1 = self.covclass(ss1)
        ss2 = ss1+s1
        ss2 = self.covclass1(ss2)
        output = output + ss2
        output = self.covclass2(output)
        output = F.upsample(output, size=input_size, mode='bilinear', align_corners=True)
        return output

    def get_backbone_params(self):
        return chain(self.block0.parameters(), self.block1.parameters(), self.block2.parameters(),
                     self.block3.parameters(), self.transition1.parameters(), self.transition2.parameters(),
                     self.transition3.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


class _CAModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_CAModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]

        pyramids.extend([F.upsample(stage(features), size=(h, w), mode='bilinear',
                                    align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output