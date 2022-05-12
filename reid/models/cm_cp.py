import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd

from torch.autograd import Variable, Function


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def distance_mining(dist_mat, labels, cameras):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)

    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())  # & cameras.expand(N,N).eq(cameras.expand(N,N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())  # | cameras.expand(N,N).ne(cameras.expand(N,N).t())
    d1 = torch.max(dist_mat * is_pos.float().detach(), 1, keepdim=True)[0]
    d1 = d1.squeeze(1)
    # dist_neg=dist_mat[is_neg].contiguous().view(N,-1)
    d2 = d1.new().resize_as_(d1).fill_(0)
    d3 = d1.new().resize_as_(d1).fill_(0)
    d2ind = []
    for i in range(N):
        sorted_tensor, sorted_index = torch.sort(dist_mat[i])
        cam_id = cameras[i]
        B, C = False, False
        for ind in sorted_index:
            if labels[ind] == labels[i]:
                continue
            if B == False and cam_id == cameras[ind]:
                d3[i] = dist_mat[i][ind]
                B = True
            if C == False and cam_id != cameras[ind]:
                d2[i] = dist_mat[i][ind]
                C = True
                d2ind.append(ind)
            if B and C:
                break
    return d1, d2, d3, d2ind

class DistanceLoss(object):
    """Multi-camera negative loss
        In a mini-batch,
       d1=(A,A'), A' is the hardest true positive.
       d2=(A,C), C is the hardest negative in another camera.
       d3=(A,B), B is the hardest negative in same camera as A.
       let d1<d2<d3
    """
    def __init__(self,):
        self.ranking_loss1=nn.MarginRankingLoss(margin=0.3,reduction="mean")
        self.ranking_loss2=nn.MarginRankingLoss(margin=0.3,reduction="mean")
        print("distance loss")


    def __call__(self,feat,labels,cameras, normalize_feature=False):
        if normalize_feature: # default: don't normalize , distance [0,1]
            feat=normalize(feat,axis=-1)
        dist_mat=euclidean_dist(feat,feat)
        d1,d2,d3,d2ind= distance_mining(dist_mat,labels,cameras)

        y=d1.new().resize_as_(d1).fill_(1)
        l1=self.ranking_loss1(d2,d1,y)
        l2=self.ranking_loss2(d3,d2,y)
        loss=l2+l1
        return loss


class ViewContrastiveLoss(nn.Module):
    def __init__(self, num_instance=16, T=0.1):
        super(ViewContrastiveLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.num_instance = num_instance
        self.T = T
        self.pos_ranking_loss = nn.MarginRankingLoss(margin=0.1, reduction="mean")
        self.neg_ranking_loss = nn.MarginRankingLoss(margin=0.1, reduction="mean")
        print("ecd sorted positive loss")


    # def forward(self, q, label, cam_ids):
    #     N = q.size(0)
    #     mat_sim = torch.matmul(q, q.transpose(0, 1))
    #
    #     # mat_sim = self.ecd_dist(q)
    #     mat_eq_psid = label.expand(N, N).eq(label.expand(N, N).t())
    #     mat_eq_cid = cam_ids.expand(N, N).eq(cam_ids.expand(N, N).t())
    #
    #     # batch hard positive
    #     hard_p_intra_cam_sim, hard_p_inter_cam_sim, hard_p, hard_n, hard_n_intra_cam = self.batch_hard(mat_sim, mat_eq_psid, mat_eq_cid)
    #     y = hard_p_inter_cam_sim.new().resize_as_(hard_p_inter_cam_sim).fill_(1)
    #     y1= hard_p_inter_cam_sim.new().resize_as_(hard_p_inter_cam_sim).fill_(1)
    #     # loss = self.pos_ranking_loss(hard_p_inter_cam_sim, hard_p_intra_cam_sim, y) \
    #     #        + self.neg_ranking_loss(hard_p_intra_cam_sim, hard_n, y1)
    #     # loss = self.pos_ranking_loss(hard_p_inter_cam_sim, hard_p_intra_cam_sim, y)
    #     #
    #     loss = self.pos_ranking_loss(hard_p_inter_cam_sim, hard_p_intra_cam_sim, y) \
    #            + self.neg_ranking_loss(hard_p_intra_cam_sim, hard_n_intra_cam, y1)
    #     # loss = self.pos_ranking_loss(hard_p, hard_n, y)
    #     return loss
    #
    # def batch_hard(self, mat_sim, mat_eq_psid, mat_eq_cid):
    #     N = len(mat_sim)
    #     # pos all cam
    #     hard_p, _ = torch.min(mat_sim[mat_eq_psid].contiguous().view(N, -1), 1, keepdim=True)
    #     # pos intra cam
    #     p_intra_indices = mat_eq_psid & mat_eq_cid
    #     hard_p_intra_cam = torch.ones(N, dtype=mat_sim.dtype).cuda()
    #     for idx in range(N):
    #         idx_sim = mat_sim[idx][p_intra_indices[idx]]
    #         if len(idx_sim) > 0:
    #             hard_p_intra_cam[idx] = torch.min(idx_sim)
    #
    #     # pos inter cam
    #     p_inter_indices = mat_eq_psid & (~mat_eq_cid)
    #     hard_p_inter_cam = hard_p_intra_cam.clone().detach()
    #     for idx in range(N):
    #         idx_sim = mat_sim[idx][p_inter_indices[idx]]
    #         if len(idx_sim) > 0:
    #             hard_p_inter_cam[idx] = torch.min(idx_sim)
    #     # neg all cam
    #     hard_n, _ = torch.max(mat_sim[~mat_eq_psid].contiguous().view(N, -1), 1, keepdim=True)
    #     # neg intra camera
    #     n_intra_indices = (~mat_eq_psid) & mat_eq_cid
    #     # hard_n_intra_cam = hard_p_intra_cam.clone().detach()
    #     hard_n_intra_cam = torch.ones(N, dtype=mat_sim.dtype).cuda()
    #     for idx in range(N):
    #         idx_sim = mat_sim[idx][n_intra_indices[idx]]
    #         if len(idx_sim) > 0:
    #             hard_n_intra_cam[idx] = torch.max(idx_sim)
    #
    #     return hard_p_intra_cam, hard_p_inter_cam, hard_p, hard_n, hard_n_intra_cam

    def forward(self, q, labels, cam_ids):
        N = q.size(0)
        mat_dist = self.ecd_dist(q)
        mat_eq_psid = labels.expand(N, N).eq(labels.expand(N, N).t())
        mat_eq_cid = cam_ids.expand(N, N).eq(cam_ids.expand(N, N).t())

        # batch hard positive
        hard_p_intra_cam_dist, hard_p_inter_cam_dist, hard_p, hard_n, hard_n_intra_cam = self.batch_hard_ecd(mat_dist, mat_eq_psid, mat_eq_cid)
        y = hard_p_inter_cam_dist.new().resize_as_(hard_p_inter_cam_dist).fill_(1)
        y1= hard_p_inter_cam_dist.new().resize_as_(hard_p_inter_cam_dist).fill_(1)
        # loss = self.pos_ranking_loss(hard_p_intra_cam_dist, hard_p_inter_cam_dist, y) \
        #        + self.neg_ranking_loss(hard_n, hard_p_intra_cam_dist, y1)
        # print("----------------------")
        # # print(hard_p_intra_cam_dist, hard_p_inter_cam_dist)
        loss = self.pos_ranking_loss(hard_p_intra_cam_dist, hard_p_inter_cam_dist, y)
        #
        # loss = self.pos_ranking_loss(hard_p_intra_cam_dist, hard_p_inter_cam_dist, y) \
        #        + self.neg_ranking_loss(hard_n_intra_cam, hard_p_intra_cam_dist, y1)
        # loss = self.pos_ranking_loss(hard_n, hard_p, y)
        return loss

    def batch_hard_ecd(self, mat_dist, mat_eq_psid, mat_eq_cid):
        N = len(mat_dist)
        # pos all cam
        hard_p, _ = torch.max(mat_dist[mat_eq_psid].contiguous().view(N, -1), 1, keepdim=True)

        # pos intra cam
        p_intra_indices = mat_eq_psid & mat_eq_cid
        hard_p_intra_cam = torch.ones(N, dtype=mat_dist.dtype).cuda()
        for idx in range(N):
            idx_sim = mat_dist[idx][p_intra_indices[idx]]
            if len(idx_sim) > 0:
                hard_p_intra_cam[idx] = torch.max(idx_sim)

        # pos inter cam
        p_inter_indices = mat_eq_psid & (~mat_eq_cid)
        hard_p_inter_cam = hard_p_intra_cam.clone().detach()
        for idx in range(N):
            idx_sim = mat_dist[idx][p_inter_indices[idx]]
            if len(idx_sim) > 0:
                hard_p_inter_cam[idx] = torch.max(idx_sim)

        # neg all cam
        hard_n, _ = torch.min(mat_dist[~mat_eq_psid].contiguous().view(N, -1), 1, keepdim=True)
        # neg intra camera
        n_intra_indices = (~mat_eq_psid) & mat_eq_cid
        hard_n_intra_cam = hard_p_intra_cam.clone().detach()
        for idx in range(N):
            idx_sim = mat_dist[idx][n_intra_indices[idx]]
            if len(idx_sim) > 0:
                hard_n_intra_cam[idx] = torch.min(idx_sim)

        return hard_p_intra_cam, hard_p_inter_cam, hard_p, hard_n, hard_n_intra_cam


    def ecd_dist(self, features):
        x = features
        y = x
        m = len(features)
        dists = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dists.addmm_(1, -2, x, y.t())
        dists = dists.clamp(min=1e-12).sqrt()  # for numerical stability
        return dists

    # def forward(self, q, label, cam_ids):
    #     batchSize = q.shape[0]
    #     N = q.size(0)
    #
    #     mat_sim = torch.matmul(q, q.transpose(0, 1))
    #     mat_eq_psid = label.expand(N, N).eq(label.expand(N, N).t()).float()
    #     # mat_eq_cid = cam_ids.expand(N, N).eq(cam_ids.expand(N, N).t()).float()
    #
    #
    #     # batch hard positive
    #     hard_p, hard_n, hard_p_indice, hard_n_indice = self.batch_hard(mat_sim, mat_eq_psid, True)
    #     l_pos = hard_p.view(batchSize, 1)
    #
    #     # batch hard negative
    #     mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
    #     negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
    #     out = torch.cat((l_pos, negatives), dim=1) / self.T
    #     # out = torch.cat((l_pos, l_neg, negatives), dim=1) / self.T
    #     targets = torch.zeros([batchSize]).cuda().long()
    #     triple_dist = F.log_softmax(out, dim=1)
    #     triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)
    #     # triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)*l + torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1)+1, 1) * (1-l)
    #     loss = (- triple_dist_ref * triple_dist).mean(0).sum()
    #     return loss
    #
    # def batch_hard(self, mat_sim, mat_eq, indice=False):
    #     sorted_mat_sim, positive_indices = torch.sort(mat_sim + (9999999.) * (1 - mat_eq), dim=1,
    #                                                        descending=False)
    #     hard_p = sorted_mat_sim[:, 0]
    #     hard_p_indice = positive_indices[:, 0]
    #     sorted_mat_distance, negative_indices = torch.sort(mat_sim + (-9999999.) * (mat_eq), dim=1,
    #                                                        descending=True)
    #     hard_n = sorted_mat_distance[:, 0]
    #     hard_n_indice = negative_indices[:, 0]
    #     if (indice):
    #         return hard_p, hard_n, hard_p_indice, hard_n_indice
    #     return hard_p, hard_n


# class IPClusterMemory(nn.Module, ABC):
#     def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
#         super(IPClusterMemory, self).__init__()
#         self.num_features = num_features
#         self.num_samples = num_samples
#
#         self.momentum = momentum
#         self.temp = temp
#         self.use_hard = use_hard
#         self.instance_loss = ViewContrastiveLoss(num_instance=16, T=temp)
#         # self.distance_loss = DistanceLoss()
#         self.register_buffer('features', torch.zeros(num_samples, num_features))
#         print("instance proxy clusterMemory")
#
#     def forward(self, inputs, targets, cam_ids=None):
#         loss = self.instance_loss(inputs.clone().detach(), targets, cam_ids)
#         # loss = self.distance_loss(inputs, targets, cam_ids)
#         inputs = F.normalize(inputs, dim=1).cuda()
#         outputs = cm(inputs, targets, self.features, self.momentum)
#         outputs /= self.temp
#         loss += F.cross_entropy(outputs, targets)
#
#         # if self.use_hard:
#         #     outputs = cm_hard(inputs, targets, self.features, self.momentum)
#         # else:
#         #     outputs = cm(inputs, targets, self.features, self.momentum)
#         # outputs = cm_hard(inputs, targets, self.features, self.momentum)
#         # print("loss", loss)
#         # print(loss)
#         # assert 0 > 1
#         return loss


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


class Cam_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, cam_ids, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, cam_ids)

        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
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
                if len(cid_ind) > 1:
                    psid_cid_center = torch.mean(psid_cid_inputs, dim=0)
                else:
                    psid_cid_center = psid_cid_inputs

                # if len(cid_ind) > 1:
                #     unique_psid_cid_inputs = torch.unique(psid_inputs[cid_ind], dim=0)  # remove repeat
                #     if len(unique_psid_cid_inputs) > 1:
                #         psid_cid_center = torch.mean(unique_psid_cid_inputs, dim=0)
                #     else:
                #         psid_cid_center = unique_psid_cid_inputs[0]
                # else:
                #     psid_cid_center = psid_inputs[cid_ind]

                batch_cam_centers[psid].append(psid_cid_center)

        for psid, cam_centers in batch_cam_centers.items():
            distances = []
            for cam_center in cam_centers:
                # print(cam_center.unsqueeze(0))
                distance = cam_center.unsqueeze(0).mm(ctx.features[psid].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())
            median = np.argmin(np.array(distances))
            ctx.features[psid] = ctx.features[psid] * ctx.momentum + (1 - ctx.momentum) * cam_centers[median]
            ctx.features[psid] /= ctx.features[psid].norm()

        return grad_inputs, None, None, None, None


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.view_contrast_loss = ViewContrastiveLoss(num_instance=16)

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        print("hard cam center global memory")
        print("momentum", self.momentum)

    def forward(self, inputs, targets, cam_ids=None):
        # print("targets", torch.unique(targets))
        inputs = F.normalize(inputs, dim=1).cuda()
        # if self.use_hard:
        #     outputs = cm_hard(inputs, targets, self.features, self.momentum)
        # else:
        #     outputs = cm(inputs, targets, self.features, self.momentum)
        outputs = Cam_Hard.apply(inputs, targets, cam_ids, self.features, torch.Tensor([self.momentum]).cuda())
        # outputs = cm(inputs, targets, self.features, self.momentum)
        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss


class HardClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(HardClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        print('HardClusterMemory, em_hard hard center update')

    def forward(self, inputs, targets, cam_ids=None):

        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = em_hard(inputs, targets, self.features, self.momentum)
        # if self.use_hard:
        #     outputs = em_hard(inputs, targets, self.features, self.momentum)
        # else:
        #     outputs = cm(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss


def em_hard(inputs, indexes, features, momentum=0.5):
    return EM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class EM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, proxies, momentum):
        ctx.proxies = proxies
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)

        outputs = inputs.mm(ctx.proxies.t())

        # ins_outputs = inputs.mm(inputs.t())
        # hard_ap, hard_an = hard_example_mining(ins_outputs, targets)
        # for idx in range(len(inputs)):
        #     # positive
        #     proxy_ind = targets[idx]
        #     outputs[idx][proxy_ind] = hard_ap[idx]
        #     # negtive
        #     neg_targets = torch.sort(torch.unique(targets[targets != targets[idx]])).values
        #     for jdx in range(len(neg_targets)):
        #         outputs[idx][neg_targets[jdx]] = hard_an[idx][jdx]

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):

        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.proxies)

        for idx, lbl in enumerate(torch.unique(targets)):
            lbl_feats = inputs[targets == lbl]
            lbl_feats = torch.unique(lbl_feats, dim=0)  # remove repeats
            if len(lbl_feats) > 1:
                lbl_center = torch.mean(lbl_feats, dim=0).unsqueeze(0)
            else:
                lbl_center = lbl_feats.unsqueeze(0)
            lbl_cent_sims = lbl_feats.mm(lbl_center.t())
            proxy_idx = torch.argmin(lbl_cent_sims)
            ctx.proxies[lbl] = ctx.proxies[lbl] * ctx.momentum + (1 - ctx.momentum) * lbl_feats[proxy_idx]
            ctx.proxies[lbl] /= ctx.proxies[lbl].norm()

        # batch_centers = collections.defaultdict(list)
        # for instance_feature, index in zip(inputs, targets.tolist()):
        #     batch_centers[index].append(instance_feature)
        #
        # for index, inputs in batch_centers.items():
        #     sum_feat = 0
        #     for feature in inputs:
        #         sum_feat += feature
        #     avg_feat = sum_feat / len(inputs)
        #     ctx.proxies[index] = ctx.proxies[index] * ctx.momentum + (1 - ctx.momentum) * avg_feat
        #     ctx.proxies[index] /= ctx.proxies[index].norm()

        return grad_inputs, None, None, None

    # @staticmethod
    # def backward(ctx, grad_outputs):
    #
    #     inputs, targets = ctx.saved_tensors
    #     grad_inputs = None
    #     if ctx.needs_input_grad[0]:
    #         grad_inputs = grad_outputs.mm(ctx.proxies)
    #
    #     for idx, lbl in enumerate(torch.unique(targets)):
    #         lbl_feats = inputs[targets == lbl]
    #         lbl_sims = lbl_feats.mm(lbl_feats.t())
    #
    #         lbl_center = torch.mean(lbl_feats, dim=0).unsqueeze(0)
    #         lbl_cent_sims = lbl_feats.mm(lbl_center.t())
    #         proxy_idx = torch.argmin(lbl_cent_sims)
    #         ctx.proxies[lbl] = ctx.proxies[lbl] * ctx.momentum + (1 - ctx.momentum) * lbl_feats[proxy_idx]
    #         ctx.proxies[lbl] /= ctx.proxies[lbl].norm()
    #
    #     # batch_centers = collections.defaultdict(list)
    #     # for instance_feature, index in zip(inputs, targets.tolist()):
    #     #     batch_centers[index].append(instance_feature)
    #     #
    #     # for index, inputs in batch_centers.items():
    #     #     sum_feat = 0
    #     #     for feature in inputs:
    #     #         sum_feat += feature
    #     #     avg_feat = sum_feat / len(inputs)
    #     #     ctx.proxies[index] = ctx.proxies[index] * ctx.momentum + (1 - ctx.momentum) * avg_feat
    #     #     ctx.proxies[index] /= ctx.proxies[index].norm()
    #
    #     return grad_inputs, None, None, None


# class EM_Hard(autograd.Function):
#     def __init__(self):
#         print("em_hard center update")
#
#     @staticmethod
#     def forward(ctx, inputs, targets, proxies, momentum):
#         ctx.proxies = proxies
#         ctx.momentum = momentum
#         ctx.save_for_backward(inputs, targets)
#
#         outputs = inputs.mm(ctx.proxies.t())
#
#         # ins_outputs = inputs.mm(inputs.t())
#         # hard_ap = hard_pos_mining(ins_outputs, targets)
#         # for idx in range(len(inputs)):
#         #     # positive
#         #     proxy_ind = targets[idx]
#         #     outputs[idx][proxy_ind] = hard_ap[idx]
#         #     # negtive
#         #     neg_ind = targets != targets[idx]
#         #     neg_proxy_ind = targets[neg_ind]
#         #     outputs[idx][neg_proxy_ind] = ins_outputs[idx][neg_ind]
#
#         ins_outputs = inputs.mm(inputs.t())
#         hard_ap, hard_an = hard_example_mining(ins_outputs, targets)
#         for idx in range(len(inputs)):
#             # positive
#             proxy_ind = targets[idx]
#             outputs[idx][proxy_ind] = hard_ap[idx]
#             # negtive
#             neg_targets = torch.sort(torch.unique(targets[targets != targets[idx]])).values
#             for jdx in range(len(neg_targets)):
#                 outputs[idx][neg_targets[jdx]] = hard_an[idx][jdx]
#
#         return outputs
#
#     @staticmethod
#     def backward(ctx, grad_outputs):
#         inputs, targets = ctx.saved_tensors
#         grad_inputs = None
#         if ctx.needs_input_grad[0]:
#             grad_inputs = grad_outputs.mm(ctx.proxies)
#
#         batch_centers = collections.defaultdict(list)
#         for instance_feature, index in zip(inputs, targets.tolist()):
#             batch_centers[index].append(instance_feature)
#
#         for index, features in batch_centers.items():
#             sum_feat = 0
#             for feature in features:
#                 sum_feat += feature
#             avg_feat = sum_feat / len(features)
#             ctx.proxies[index] = ctx.proxies[index] * ctx.momentum + (1 - ctx.momentum) * avg_feat
#             ctx.proxies[index] /= ctx.proxies[index].norm()
#
#         return grad_inputs, None, None, None


def hard_pos_mining(sim_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      sim_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(sim_mat.size()) == 2
    assert sim_mat.size(0) == sim_mat.size(1)
    N = sim_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())

    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.min(
        sim_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)


    return dist_ap


def hard_example_mining(sim_mat, labels):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      sim_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(sim_mat.size()) == 2
    assert sim_mat.size(0) == sim_mat.size(1)
    N = sim_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    dist_ap, relative_p_inds = torch.min(
        sim_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_ap = dist_ap.squeeze(1)  # 1 * N

    # 对每个negtive标签找一个hard
    hard_neg_len = int(len(torch.unique(labels)) - 1)
    dist_an = torch.zeros(N, hard_neg_len)
    for idx in range(len(labels)):
        neg_lbls = torch.sort(torch.unique(labels[labels != labels[idx]])).values
        for jdx in range(hard_neg_len):
            neg_idx = labels == neg_lbls[jdx]
            dist_an[idx][jdx] = torch.max(sim_mat[idx][neg_idx])

    return dist_ap, dist_an


# class IPClusterMemory(nn.Module, ABC):
#     def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, features=None, all_pseudo_labels=None, all_cams=None):
#         super(IPClusterMemory, self).__init__()
#         self.num_features = num_features
#         self.num_samples = num_samples
#
#         self.momentum = momentum
#         self.temp = temp
#         self.use_hard = use_hard
#         self.unique_cams = torch.unique(all_cams)
#         self.all_img_cams = all_cams
#         self.all_pseudo_label = all_pseudo_labels
#
#         self.features = features
#         self.percam_memory, self.memory_class_mapper, self.concate_intra_class = self.init_cam_proxy()
#         self.bg_knn = 50
#         print("intra hard camera proxy loss")
#
#     def init_cam_proxy(self):
#         print("initialize cam proxy memory")
#         percam_memory = []  # percam proxies
#         memory_class_mapper = []  # global to intra pseudo labels
#         concate_intra_class = []  # global pseudo labels
#         for cc in self.unique_cams:
#             percam_ind = torch.nonzero(self.all_img_cams == cc).squeeze(-1)
#             # obtain global labels
#             uniq_class = torch.unique(self.all_pseudo_label[percam_ind])
#             uniq_class = uniq_class[uniq_class >= 0]
#             concate_intra_class.append(uniq_class)
#             # global labels to intra camera labels
#             cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
#             memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera
#             # percam_memory
#             if len(self.features) > 0:
#                 percam_memory.append(self.features[cc].cuda().detach())
#         concate_intra_class = torch.cat(concate_intra_class)
#         return percam_memory, memory_class_mapper, concate_intra_class
#
#     def forward(self, inputs, targets, cam_ids=None):
#
#         loss = self.intra_loss(inputs, targets, cam_ids)
#
#         return loss
#
#     def intra_loss(self, inputs, targets, cam_ids):
#         targets = targets.cpu()
#         loss = torch.tensor([0.]).cuda()
#
#         for cc in torch.unique(cam_ids):
#             inds = torch.nonzero(cam_ids == cc).squeeze(-1)
#             percam_batch_targets = targets[inds]
#             # percam_batch_targets = self.all_pseudo_label[targets[inds]]
#             percam_feat = inputs[inds]
#
#             # intra-camera loss (useful?)
#             mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_batch_targets]
#             mapped_targets = torch.tensor(mapped_targets).cuda()
#
#             associate_loss = 0
#             target_inputs = ExemplarCamMemory.apply(percam_feat, mapped_targets, cc, self.percam_memory, self.momentum)
#             temp_sims = target_inputs.detach().clone()
#             target_inputs /= self.temp
#             for k in range(len(percam_feat)):  # element of a cross entropy is calculate separately
#                 ori_asso_ind = torch.nonzero(self.concate_intra_class == percam_batch_targets[k]).squeeze(-1)  # positive index
#                 h_pos_ind = torch.argmin(temp_sims[k, ori_asso_ind])  # hardest pos ind
#                 # temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive, to find the most similar negative
#                 # neg_sel_ind = torch.sort(temp_sims[k])[1][-self.bg_knn:]
#                 # concated_input = torch.cat(
#                 #     (target_inputs[k][ori_asso_ind[h_pos_ind]].unsqueeze(0), target_inputs[k, neg_sel_ind]), dim=0)
#
#                 # intra cam hard
#                 per_cam_targets = torch.from_numpy(np.array(list(self.memory_class_mapper[cc].keys())))
#                 neg_cids_ind = per_cam_targets[per_cam_targets != percam_batch_targets[k]]
#                 concated_input = torch.cat(
#                     (target_inputs[k][ori_asso_ind[h_pos_ind]].unsqueeze(0), target_inputs[k, neg_cids_ind]), dim=0)
#
#
#                 concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).cuda()
#                 concated_target[0] = 1.0
#                 associate_loss += -1 * (
#                         F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
#
#                 # ori_asso_ind = torch.nonzero(self.concate_intra_class == percam_batch_targets[k]).squeeze(-1)  # positive index
#                 #
#                 # h_pos_ind = torch.argmin(temp_sims[k, ori_asso_ind])  # hardest pos ind
#                 #
#                 # temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive, to find the most similar negative
#                 # neg_sel_ind = torch.sort(temp_sims[k])[1][-self.bg_knn:]
#                 # concated_input = torch.cat((target_inputs[k][ori_asso_ind[h_pos_ind]].unsqueeze(0), target_inputs[k, neg_sel_ind]), dim=0)
#                 # concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).cuda()
#                 # concated_target[0] = 1.0
#                 # associate_loss += -1 * (
#                 #         F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
#             loss += 0.5 * associate_loss / len(percam_feat)
#
#         return loss



# class ExemplarCamMemory(Function):
#     # def __init__(self, em, alpha=0.01):
#     #     super(ExemplarMemory, self).__init__()
#     #     self.em = em
#     #     self.alpha = alpha
#
#     @staticmethod
#     def forward(ctx, inputs, targets, cam_id, em, alpha):
#         ctx.save_for_backward(inputs, targets, cam_id, torch.cat(em, dim=0), em[cam_id], torch.tensor(alpha))
#         outputs = inputs.mm(torch.cat(em, dim=0).t())
#         return outputs
#     # def forward(self, inputs, targets):
#     #     self.save_for_backward(inputs, targets)
#     #     outputs = inputs.mm(self.em.t())
#     #     return outputs
#
#     @staticmethod
#     def backward(ctx, grad_outputs):
#         inputs, targets, cam_id, em, cam_em, alpha = ctx.saved_tensors
#         grad_inputs = None
#         if ctx.needs_input_grad[0]:
#             grad_inputs = grad_outputs.mm(em)
#
#         for x, y in zip(inputs, targets):
#             cam_em[y] = alpha * cam_em[y] + (1.0 - alpha) * x
#             cam_em[y] /= cam_em[y].norm()
#         return grad_inputs, None, None, None, None

