import torch
import torch.nn as nn
import numpy as np
from parameter import args
from .dyconv import CondConv3d

class hsi_e(nn.Module):
    def __init__(self):
        super(hsi_e, self).__init__()

        self.hsi_step1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(5, 5, 5), padding=0),
            # CondConv3d(in_planes=1, out_planes=8, kernel_size=(9, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=8))

        self.hsi_step2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=16)
        )
        self.hsi_step3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=32)
        )
        self.hsi_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=args.uni_dimension, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension),
            # nn.Conv2d(in_channels=args.uni_dimension, out_channels=args.uni_dimension, kernel_size=3, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(num_features=args.uni_dimension)
        )

        self.hsi_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=args.uni_dimension, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension),
        )
        self.hsi_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=args.uni_dimension, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension),
            # nn.AdaptiveAvgPool3d((args.uni_dimension,args.hsi_windowSize,args.hsi_windowSize))
        )

    def forward(self, x):
        X = []
        # x = x.unsqueeze(1)
        x1 = self.hsi_step1(x)
        X1 = self.hsi_conv1(x1.reshape(-1, 8 * 16, x1.shape[3], x1.shape[4]))
        X.append(X1 )
        x2 = self.hsi_step2(x1)
        X2 = x2.reshape(-1, 16 * 14, x2.shape[3], x2.shape[4])
        X2 = self.hsi_conv2(X2)
        X.append(X2)
        x3 = self.hsi_step3(x2)
        X3 = x3.reshape(-1, 32 * 12, x3.shape[3], x3.shape[4])
        X3 = self.hsi_conv3(X3)
        X.append(X3)
        return X

    """
    conv1：（1, 30, 11, 11）， 8个 9x3x3 的卷积核 ==>（8, 22, 9, 9）
    conv2：（8, 22, 9, 9）， 16个 7x3x3 的卷积核 ==>（16, 16, 7, 7）
    conv3：（16, 16, 7, 7），32个 5x3x3 的卷积核 ==>（32, 12, 5, 5）

    """
        # self.hsi_step1 = nn.Sequential(
        #     nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(9, 3, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm3d(num_features=8))
        #
        # self.hsi_step2 = nn.Sequential(
        #     nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(7, 3, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm3d(num_features=16)
        # )
        # self.hsi_step3 = nn.Sequential(
        #     nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 3, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm3d(num_features=32)
        # )
        #
        # self.hsi_step4 = nn.Sequential(
        #     nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        #     , nn.BatchNorm2d(num_features=256)
        #     , nn.ReLU()
        #     , attentionBlock(256)
        # )
