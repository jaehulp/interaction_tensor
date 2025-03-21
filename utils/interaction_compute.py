import os
import torch
import torch.distributed as dist

class features:
    pass

def compute_proj(model, model_list, dataloader, K):

    M = len(model_list)
    N = len(dataloader.dataset)

    proj_matrix = torch.zeros((M, N, K))

    for i, m in enumerate(model_list):

        model.load_state_dict(m)

        last_layer_output = []

        acc1_meter = AverageMeter()
        for j, (xs, ys) in enumerate(dataloader):

            xs.cuda()
            ys.cuda()

            _ = model(xs)
            hook_data = features.value.data.view(xs.shape[0],-1)
            last_layer_output.append(hook_data)

        last_layer_output = torch.cat(last_layer_output, dim=0)
        _, _, V = torch.linalg.svd(last_layer_output)
        proj_output = torch.matmul(last_layer_output, V[:K].T)
        proj_matrix[i] = proj_output

    return proj_matrix


def compute_cov(proj_matrix):

    mean = torch.mean(proj_matrix, dim=1, keepdim=True)
    std = torch.std(proj_matrix, dim=1, keepdim=True)

    proj_normalized = (proj_matrix - mean) / std
    proj_reshape = proj_normalized.permute(0, 2, 1)  # Shape (M, K, N)
    cov_matrix = torch.einsum('mkn,pqn->mpkq', proj_reshape, proj_reshape) / proj_matrix.shape[1]  # Shape (M, M, K, K)

    return cov_matrix


def greedy_clustering_iteration(cov_matrix, r_corr):
    M = cov_matrix.shape[0]
    K = cov_matrix.shape[2]

    assign_mat = torch.zeros((M, K))
    maximum_mat = torch.full((M, K), -1, dtype=torch.float32)

    currentFeature = 1
    r_corr = torch.quantile(cov_matrix.flatten(), r_corr)
    print(f'r_corr coefficients: {r_corr}')

    for i in range(M):
        for j in range(K):
            if assign_mat[i, j] != 0:
                continue
            
            assign_mat[i, j] = currentFeature
            for p in range(M):
                CorrMat = cov_matrix[i, p, :, :]
                FeatureRow = CorrMat[j, :]
                for q in range(K):
                    if (FeatureRow[q] > maximum_mat[p,q]) & (FeatureRow[q] > r_corr):     
                        assign_mat[p,q] = currentFeature
                        maximum_mat[p,q] = FeatureRow[q]
            currentFeature += 1
    
    cluster_num = torch.unique(assign_mat.flatten())

    assert(len(cluster_num) == torch.max(cluster_num))

    print(f'Num of clusters: {len(cluster_num)}')

    return assign_mat


def compute_data_feature(proj_matrix, r_data):

    proj_matrix_norm = torch.norm(proj_matrix, p=float('inf'), dim=(1), keepdim=True)
    normalized_proj_matrix = proj_matrix / proj_matrix_norm

    r_data = torch.quantile(normalized_proj_matrix.view(normalized_proj_matrix.shape[0], -1), r_data, dim=1)
    print(f'r_data coefficient: {r_data}')

    data_feature = normalized_proj_matrix > r_data[:, None, None] # boolean

    return data_feature


def compute_interaction_tensor(assign_mat, data_feature):

    M = data_feature.shape[0]
    N = data_feature.shape[1]
    T = len(torch.unique(assign_mat.flatten()))

    interaction_tensor = torch.zeros((M,N,T), dtype=torch.int64)
    assign_mask = assign_mat - 1

    for t in range(interaction_tensor.shape[2]):
        mask = (assign_mask == t).unsqueeze(1)
        interaction_tensor[:, :, t] = torch.sum(mask * data_feature, dim=2)

    interaction_tensor = interaction_tensor.bool().int()

    return interaction_tensor


def interaction_process(model, model_list, dataloader, output_dir, args, epoch): # args contain K, r_cor, r_data

    K = args.K

    proj_matrix = compute_proj(model, model_list, dataloader, K)
    cov_matrix = compute_cov(proj_matrix)
    
    assign_mat = greedy_clustering_iteration(cov_matrix, args.r_corr)
    data_feature = compute_data_feature(proj_matrix, args.r_data)

    interaction_tensor = compute_interaction_tensor(assign_mat, data_feature)

    save_path = os.path.join(output_dir, f'IntTensor_{epoch}.pt')
    torch.save(interaction_tensor, save_path)
