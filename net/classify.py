import torch
import torch.nn as nn
import torch.nn.functional as F
from parameter import args


class classify(nn.Module):
    def __init__(self):
        super(classify, self).__init__()

        self.conv1 = torch.nn.Conv2d(kernel_size=1, in_channels=args.uni_dimension, stride=1, out_channels=512)
        self.linear1 = nn.Linear(in_features=512, out_features=args.num_classes)

    # f:B uni_dimension hsi_windowSize hsi_windowSize
    def forward(self, x):
        x = self.conv1(x)
        x = F.avg_pool2d(x, kernel_size=args.hsi_windowSize).reshape(-1, 512)
        x = self.linear1(x)
        return x.squeeze(-1).squeeze(-1)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from parameter import args
# class classify(nn.Module):
#     def __init__(self):
#         super(classify, self).__init__()
#         self.block = nn.Sequential(
#             torch.nn.Conv2d(kernel_size=3, in_channels=args.uni_dimension, out_channels=64, padding=0),
#             torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=32, padding=0),
#             torch.nn.Conv2d(kernel_size=1, in_channels=32, out_channels=args.num_classes),
#             # nn.Linear(32, args.num_classes)
#         )
# # f:B uni_dimension hsi_windowSize hsi_windowSize
#     def forward(self, x):
#         res = self.block(x)
#         return res.squeeze(-1).squeeze(-1)
