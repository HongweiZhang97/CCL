import torch
import torch.nn as nn
import math
import torch.nn.functional as F


from torch.autograd import Variable

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# class CoordAtt(nn.Module):
#
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         mip = max(8, inp // reduction)
#         self.u_pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.u_pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.u_conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.u_bn1 = nn.BatchNorm2d(mip)
#         self.u_act = h_swish()
#         self.u_conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.u_conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # self.d_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.d_pool_w = nn.AdaptiveAvgPool2d((1, None))
        # self.d_conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # self.d_bn1 = nn.BatchNorm2d(mip)
        # self.d_act = h_swish()
        # self.d_conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.d_conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    # def forward(self, x):
    #     alpha = 4.0
    #     dim = 3
    #     n, c, h, w = x.size()
    #     left_x, right_x = torch.split(x, int(w / 2), dim=dim)
    #     upper_out = self.u_out(left_x)
    #     down_out = self.u_out(right_x)
    #     out = torch.cat((upper_out, down_out), dim=dim) * alpha
    #     return out
    #
    # def u_out(self, x):
    #     identity = x
    #     n,c,h,w = x.size()
    #     x_h = self.u_pool_h(x)  # h pool
    #     x_w = self.u_pool_w(x).permute(0, 1, 3, 2)  # w pool
    #
    #     y = torch.cat([x_h, x_w], dim=2)
    #     y = self.u_conv1(y)
    #     y = self.u_bn1(y)
    #     y = self.u_act(y)
    #     x_h, x_w = torch.split(y, [h, w], dim=2)
    #     x_w = x_w.permute(0, 1, 3, 2)
    #
    #     a_h = self.u_conv_h(x_h).sigmoid()
    #     a_w = self.u_conv_w(x_w).sigmoid()
    #
    #     out = identity * a_w * a_h
    #     return out

    # def d_out(self, x):
    #     identity = x
    #     n,c,h,w = x.size()
    #     x_h = self.d_pool_h(x)  # h pool
    #     x_w = self.d_pool_w(x).permute(0, 1, 3, 2)  # w pool
    #
    #     y = torch.cat([x_h, x_w], dim=2)
    #     y = self.d_conv1(y)
    #     y = self.d_bn1(y)
    #     y = self.d_act(y)
    #     x_h, x_w = torch.split(y, [h, w], dim=2)
    #     x_w = x_w.permute(0, 1, 3, 2)
    #
    #     a_h = self.d_conv_h(x_h).sigmoid()
    #     a_w = self.d_conv_w(x_w).sigmoid()
    #
    #     out = identity * a_w * a_h
    #     return out

# class CoordAtt(nn.Module):
#
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
#         # self.pool_w = nn.AdaptiveMaxPool2d((1, None))
#         # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 平均值池化
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 平均值池化
#         mip = max(8, inp // reduction)
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#         alpha = 4.0
#         n,c,h,w = x.size()
#         x_h = self.pool_h(x)  # h pool
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)  # w pool
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#
#         out = identity * a_w * a_h * alpha
#         # print("size", torch.mean(identity[0]), torch.mean(out[0]))
#
#         return out

# class CoordAtt(nn.Module):
#
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         # self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
#         # self.pool_w = nn.AdaptiveMaxPool2d((1, None))
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 平均值池化
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 平均值池化
#         mip = max(8, inp // reduction)
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#         # self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # self.conv_c = nn.Sequential(
#         #     nn.Linear(inp, mip, bias=False),
#         #     # nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0),
#         #     nn.BatchNorm1d(mip),
#         #     h_swish(),
#         #     nn.Linear(mip, oup, bias=False),
#         #     # nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0),
#         #     nn.Sigmoid()
#         # )
#         print("alpha 4.0")
#
#     def forward(self, x):
#         identity = x
#         # alpha = 4.0
#         # belta = 2.0
#         alpha = 4.0
#         belta = 1.0
#         n,c,h,w = x.size()
#         x_h = self.pool_h(x)  # h pool
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)  # w pool
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#
#         out = identity * a_w * a_h * alpha
#
#         # y = self.avg_pool(x).view(n, c)
#         #
#         # y = self.conv_c(y).view(n, c, 1, 1)
#         # out = out_hw * y.expand_as(out_hw) * belta
#
#         return out

class ThreeDAM(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(ThreeDAM, self).__init__()
        # self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        # self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 平均值池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 平均值池化
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.conv_c = nn.Sequential(
        #     nn.Linear(inp, mip, bias=False),
        #     # nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm1d(mip),
        #     h_swish(),
        #     nn.Linear(mip, oup, bias=False),
        #     # nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0),
        #     nn.Sigmoid()
        # )

        print("att alpha", 4.0)

    def forward(self, x):
        identity = x
        # alpha = 4.0
        # belta = 2.0
        alpha = 4.0
        belta = 1.0
        n,c,h,w = x.size()
        x_h = self.pool_h(x)  # h pool
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # w pool

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h * alpha


        return out


class ThreeDAM(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(ThreeDAM, self).__init__()
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
        self.conv_c = nn.Sequential(
            nn.Linear(inp, mip, bias=False),
            nn.BatchNorm1d(mip),
            nn.Sigmoid(),
            nn.Linear(mip, oup, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        identity = x
        alpha = 4.0
        # alpha = 4.0
        belta = 1.0
        n,c,h,w = x.size()
        x_h = self.pool_h(x)  # h pool
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # w pool

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out_hw = identity * a_w * a_h * alpha

        y = self.avg_pool(out_hw).view(n, c)

        y = self.conv_c(y).view(n, c, 1, 1)
        out = out_hw * y.expand_as(out_hw) * belta

        return out