import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from torch.autograd import Variable

class CoordAttChannel(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(CoordAttChannel, self).__init__()
        # self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        # self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 平均值池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 平均值池化
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        # self.act = h_swish()
        self.act = nn.Sigmoid()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.conv_c = nn.Sequential(
        #     nn.Linear(inp, mip, bias=False),
        #     nn.BatchNorm1d(mip),
        #     h_swish(),
        #     # nn.Sigmoid(),
        #     nn.Linear(mip, oup, bias=False),
        #     nn.Sigmoid()
        # )

        # self.conv_c = nn.Sequential(
        #     nn.Linear(inp, mip, bias=False),
        #     nn.BatchNorm1d(mip),
        #     h_swish(),
        #     nn.Linear(mip, oup, bias=False),
        #     nn.BatchNorm1d(oup),
        #     nn.Sigmoid()
        # )

        # self.conv_c = nn.Sequential(
        #     nn.Linear(inp, mip, bias=False),
        #     nn.BatchNorm1d(mip),
        #     nn.Sigmoid(),
        #     nn.Linear(mip, oup, bias=False),
        #     nn.Sigmoid()
        # )
        self.conv_c = nn.Sequential(
            nn.Linear(inp, mip, bias=False),
            nn.ReLU(),
            nn.Linear(mip, oup, bias=False),
            nn.Sigmoid()
        )
        # print("att alpha", 4.0)
        # print("att belta", 1.0)
        # print("out_hw channel 6 sigmoid")
        print("senet channel att")



    def forward(self, x):
        # identity = x
        # alpha = 4.0
        # # alpha = 4.0
        belta = 1.0
        n,c,h,w = x.size()
        # x_h = self.pool_h(x)  # h pool
        # x_w = self.pool_w(x).permute(0, 1, 3, 2)  # w pool
        #
        # y = torch.cat([x_h, x_w], dim=2)
        # y = self.conv1(y)
        # y = self.bn1(y)
        # y = self.act(y)
        # x_h, x_w = torch.split(y, [h, w], dim=2)
        # x_w = x_w.permute(0, 1, 3, 2)
        #
        # a_h = self.conv_h(x_h).sigmoid()
        # a_w = self.conv_w(x_w).sigmoid()
        #
        # out_hw = identity * a_w * a_h * alpha

        out_hw = x

        y = self.avg_pool(out_hw).view(n, c)
        # y = self.avg_pool(x).view(n, c)

        y = self.conv_c(y).view(n, c, 1, 1)
        out = out_hw * y.expand_as(out_hw) * belta

        return out
