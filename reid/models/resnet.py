from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import build_pooling_layer
from .threedam import ThreeDAM


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 pooling_type='avg', using_local_feat=False, num_stripes=1, att=False):
        print('pooling_type: {}'.format(pooling_type))
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        if att:
            print("using att model")
            self.base = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,  # no relu
                resnet.layer1, ThreeDAM(inp=256, oup=256), resnet.layer2, ThreeDAM(inp=512, oup=512),
                resnet.layer3, ThreeDAM(inp=1024, oup=1024), resnet.layer4, ThreeDAM(inp=2048, oup=2048))
        else:
            print("base model")
            self.base = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)



        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            if using_local_feat:
                print("local batch norm, local norm!")
                self.num_features = out_planes
                self.feat_bn = nn.ModuleList()
                for i in range(num_stripes):
                    self.feat_bn.append(nn.BatchNorm1d(self.num_features))
                    self.feat_bn[i].bias.requires_grad_(False)

            else:
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        if using_local_feat:
            for i in range(num_stripes):
                init.constant_(self.feat_bn[i].weight, 1)
                init.constant_(self.feat_bn[i].bias, 0)
        else:
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

        # identity normalization
        self.identity_norm = nn.BatchNorm2d(self.num_features)


    def forward(self, x):
        bs = x.size(0)
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:  # default: False
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False):
            bn_x = F.normalize(bn_x)  # test
            return bn_x, bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)  # norm
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x  # train

        return prob


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


