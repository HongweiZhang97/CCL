import torch
import numpy as np
torch.autograd.set_detect_anomaly(True)
import itertools
import time
import faiss
import torch.nn.functional as F


def cal_scatter(features, global_labels):

    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    label_to_images = {}  # 标签：索引
    for idx, l in enumerate(global_labels):
        label_to_images[l] = label_to_images.get(l, []) + [idx]  # 标签为l的所有图片索引

    # 根据标签排序获得索引列表
    sort_image_by_label = list(
        itertools.chain.from_iterable([label_to_images[key] for key in sorted(label_to_images.keys())]))

    # 提取特征和预测值
    cluster_num = len(np.unique(global_labels))

    u_feas = features  # u_feas 2048
    u_feas_sorted = u_feas[sort_image_by_label]  # 根据类别进行排序
    feas_dist = cosine_dist(u_feas_sorted, u_feas_sorted)

    start_index = np.zeros(cluster_num, dtype=np.int)
    end_index = np.zeros(cluster_num, dtype=np.int)
    counts = 0
    i = 0

    for key in sorted(label_to_images.keys()):
        start_index[i] = counts
        end_index[i] = counts + len(label_to_images[key])  # 不包含
        counts = end_index[i]
        i = i + 1

    # class_num x 2048、class_num x 1024
    var = np.zeros((cluster_num, cluster_num))

    centroids = []
    for l in range(cluster_num):
        feas = u_feas[label_to_images[l]]
        center_feat = np.average(feas, axis=0)
        centroids.append(center_feat)

    centroids = np.array(centroids)
    centroids_dist = cosine_dist(centroids, centroids)
    # cen_2_fea_dist = cosine_dist(centroids, u_feas_sorted)

    for l in range(cluster_num):
        index_a_start = start_index[l]
        index_a_end = end_index[l]
        len_a = index_a_end - index_a_start
        feas_a = u_feas[label_to_images[l]]
        center_a = centroids[l]
        for j in range(l, cluster_num):
            index_b_start = start_index[j]
            index_b_end = end_index[j]
            len_b = index_b_end - index_b_start
            if l == j:
                dist2_a_b = 0.0
                dist2_b_a = 0.0
            else:
                if len_a==1 and len_b==1:
                    dist2_a_b = feas_dist[index_a_start:index_a_end, index_b_start:index_b_end]
                    dist2_b_a = feas_dist[index_b_start:index_b_end, index_a_start:index_a_end]
                elif len_b == 1:
                    dist2_a_b = np.sum(feas_dist[index_a_start:index_a_end, index_b_start:index_b_end])
                    dist2_b_a = centroids_dist[j][l]
                elif len_a == 1:
                    dist2_b_a = np.sum(feas_dist[index_b_start:index_b_end, index_a_start:index_a_end])
                    dist2_a_b = centroids_dist[l][j]
                else:
                    feas_b = u_feas[label_to_images[j]]
                    center_b = centroids[j]
                    dist2_a_b = np.sum(1.0 - np.dot(center_b, feas_a.T))
                    dist2_b_a = np.sum(1.0 - np.dot(center_a, feas_b.T))
                    # dist2_a_b = np.sum(cen_2_fea_dist[j, label_to_images[l]])
                    # dist2_b_a = np.sum(cen_2_fea_dist[l, label_to_images[j]])

            var[l, j] = (dist2_a_b + dist2_b_a) / (len_a + len_b)

    del centroids_dist, feas_dist
    var = var.T + var - var * np.eye(cluster_num)

    var = re_ranking(torch.from_numpy(2 * var), k1=20, k2=6, lambda_value=0.0)

    # 还原标签顺序
    instance_num = len(global_labels)
    W = []
    for i in range(instance_num):
        W.append(var[global_labels[i]][global_labels].tolist())
    W = np.array(W)
    return W


def re_ranking(dist_mat, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False, lambda_value=0.3):

    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = dist_mat.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    initial_rank = torch.argsort(dist_mat)
    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = dist_mat[i, k_reciprocal_expansion_index].unsqueeze(0)
        if use_float16:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    # 2
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank
    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1,N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    # 3
    del invIndex, V
    jaccard_dist = jaccard_dist * (1 - lambda_value) + dist_mat.numpy() * lambda_value
    print("lambda_value", lambda_value)
    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


def cosine_dist(x, y):
    return 1 - np.dot(x, y.T)