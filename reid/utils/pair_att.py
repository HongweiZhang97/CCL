import torch
import torch.nn as nn

def pair_att_sims(q_feats, g_feats, att_module):
    sims_mat = []
    for idx, q_f in enumerate(q_feats):
        exp_q_fs = q_f.expand(len(g_feats), len(q_f))
        con_feats = torch.cat((exp_q_fs, g_feats), dim=1)
        con_att_scores = att_module(con_feats)
        att_g_f = g_feats * con_att_scores * con_att_scores
        idx_sims = q_f.unsqueeze(0).mm(att_g_f.t())
        sims_mat.append(idx_sims.squeeze(0))
    sims_mat = torch.stack(sims_mat, dim=0)
    return sims_mat

def min_dim_sims(q_feats, g_feats):
    sims_mat = []
    for idx, q_f in enumerate(q_feats):
        dim_sims = q_f.unsqueeze(0) * g_feats
        idx_sims = torch.min(dim_sims, dim=1).values
        sims_mat.append(idx_sims.squeeze(0))
    sims_mat = torch.stack(sims_mat, dim=0)
    return sims_mat

def max_dim_dist(features, query, gallery):
    q_feats = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0).cuda()
    g_feats = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0).cuda()
    dist_mat = []
    for idx, q_f in enumerate(q_feats):
        dim_dist = torch.abs(q_f.unsqueeze(0) - g_feats)
        idx_sims = torch.max(dim_dist, dim=1).values
        dist_mat.append(idx_sims.squeeze(0))
    dist_mat = torch.stack(dist_mat, dim=0)
    return dist_mat.cpu(), q_feats.cpu(), g_feats.cpu()