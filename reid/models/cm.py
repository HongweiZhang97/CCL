import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd

from torch.autograd import Variable, Function


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.register_buffer('cluster_features', torch.zeros(num_samples, num_features))
        self.cam_centroids = None

    def forward(self, inputs, targets, cam_ids=None, batch_idx=None):

        inputs = F.normalize(inputs, dim=1).cuda()
        loss = 0.0
        sims = Hard_Cam_Contrast.apply(inputs, targets, cam_ids, self.dual_cluster_features, torch.Tensor([self.momentum]).cuda(), torch.Tensor([self.momentum1]).cuda())
        sims /= self.temp
        loss += F.cross_entropy(sims, targets)

        return loss


class Hard_Cam_Contrast(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, cam_ids, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, cam_ids)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        nums = len(ctx.features)//2

        inputs, targets, cam_ids = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_cam_centers = collections.defaultdict(list)
        for psid in torch.unique(targets):
            psid_ind = targets == psid
            psid_inputs = inputs[psid_ind]
            psid_cids = cam_ids[psid_ind]

            for cid in torch.unique(psid_cids):
                cid_ind = psid_cids == cid
                psid_cid_inputs = psid_inputs[cid_ind]
                psid_cid_center = torch.mean(psid_cid_inputs, dim=0)

                batch_cam_centers[psid].append(psid_cid_center)


        for psid, cam_centers in batch_cam_centers.items():
            distances = []
            for cam_center in cam_centers:
                distance = cam_center.unsqueeze(0).mm(ctx.features[psid].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[psid] = ctx.features[psid] * ctx.momentum + (1 - ctx.momentum) * cam_centers[median]
            ctx.features[psid] /= ctx.features[psid].norm()

        return grad_inputs, None, None, None, None, None





class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        # print("--------------")
        # print(inputs.size())
        # print(ctx.features.size())
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))






