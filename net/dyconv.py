import torch
import torch.nn as nn

from einops import rearrange, repeat

from .Router import Router

from parameter import args

import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_planes, K, D):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        # self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)  # max和avg并行可以减少高维张量信息的丢失？CBAM

        # self.net=nn.Conv3d(in_planes, K, kernel_size=1)

        self.net = nn.Conv3d(in_planes, K, kernel_size=1, padding=0)
        self.net1 = nn.Conv3d(in_planes, K, kernel_size=3, padding=1)
        self.net2 = nn.Conv3d(in_planes, K, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 将输入特征全局池化为 [N, C * D, 1, 1]
        N, _, _, H, W = x.shape
        # xa = self.avgpool(self.net(x)) + self.avgpool(self.net1(x)) + self.avgpool(self.net2(x))
        # xa = x.view(N, -1)  # 64 1 30 11 11
        x1 = self.net(x) # 64 4 30 11 11
        x2 = self.net1(x)
        x3 = self.net2(x)

        xa = self.avgpool(x1) + self.avgpool(x2) + self.avgpool(x3)
        # xb = self.maxpool(x1) + self.maxpool(x2) + self.maxpool(x3)
        xa = xa.view(N, -1)
        # xb = xb.view(N, -1)
        return self.softmax(xa) # 64 4


class CondConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0,
                 groups=1, K=4, D=0):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.K = K
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = Attention(in_planes=in_planes, K=K, D=D)
        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, self.kernel_size[0], self.kernel_size[1],
                        self.kernel_size[2]), requires_grad=True)
        self.conv3d = nn.Conv3d(in_channels=self.in_planes, out_channels=self.out_planes, kernel_size=self.kernel_size,
                                stride=self.stride, padding=self.padding)

    def forward(self, x):
        # 计算注意力分数
        #  64 1 30 11 11
        r = self.conv3d(x)
        N, in_planels, H, W, D = x.shape
        softmax_att = self.attention(x) #64 4

        x = x.view(1, -1, H, W, D)   # 1 512 22 9 9
        weight = self.weight  # 4 8 1 9 3 3
        weight = weight.view(self.K, -1)# 4 648
        aggregate_weight = torch.mm(softmax_att, weight) # 64 648
        aggregate_weight = aggregate_weight.view(
            N * self.out_planes, self.in_planes // self.groups,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]) # 1024 8 7 3 3
        output = F.conv3d(x, weight=aggregate_weight,
                          stride=self.stride, padding=self.padding,
                          groups=self.groups * N)# 1 512 22 9 9
        # 恢复形状
        _, _, H, W, D = output.shape
        output = output.view(N, self.out_planes, H, W, D)
        return output + r




class Attention2(nn.Module):
    def __init__(self, in_planes, K, D):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        # self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)  # max和avg并行可以减少高维张量信息的丢失？CBAM

        # self.net=nn.Conv3d(in_planes, K, kernel_size=1)

        self.net = nn.Conv3d(1, K, kernel_size=1, padding=0)
        self.net1 = nn.Conv3d(1, K, kernel_size=3, padding=1)
        self.net2 = nn.Conv3d(1, K, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 将输入特征全局池化为 [N, C * D, 1, 1]
        N, _, _, H, W = x.shape
        # xa = self.avgpool(self.net(x)) + self.avgpool(self.net1(x)) + self.avgpool(self.net2(x))
        # xa = x.view(N, -1)  # 64 1 30 11 11
        x1 = self.net(x) # 64 4 30 11 11
        x2 = self.net1(x)
        x3 = self.net2(x)

        xa = self.avgpool(x1) + self.avgpool(x2) + self.avgpool(x3)
        # xb = self.maxpool(x1) + self.maxpool(x2) + self.maxpool(x3)
        xa = xa.view(N, -1)
        # xb = xb.view(N, -1)
        return self.softmax(xa) # 64 4


class CondConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0,
                 groups=1, K=4, D=0):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.K = K
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = Attention2(in_planes=in_planes, K=K, D=D)
        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, self.kernel_size, self.kernel_size), requires_grad=True)
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=self.out_planes, kernel_size=self.kernel_size,
                                stride=self.stride, padding=self.padding)

    def forward(self, x):
        # 计算注意力分数
        #  64 1 30 11 11
        r = self.conv(x)
        N, H, W, D = x.shape
        softmax_att = self.attention(x.unsqueeze(1)) #64 4

        x = x.view(1, -1, W, D)   # 1 64 30 11 11
        weight = self.weight  # 4 8 1 9 3 3
        weight = weight.view(self.K, -1)# 4 648
        aggregate_weight = torch.mm(softmax_att, weight) # 64 648
        aggregate_weight = aggregate_weight.view(
            N * self.out_planes, self.in_planes // self.groups,
            self.kernel_size, self.kernel_size) # 512 1 9 3 3
        output = F.conv2d(x, weight=aggregate_weight,
                          stride=self.stride, padding=self.padding,
                          groups=self.groups * N)# 1 512 22 9 9
        # 恢复形状
        _, H, W, D = output.shape
        output = output.view(N, self.out_planes, W, D)
        return output + r