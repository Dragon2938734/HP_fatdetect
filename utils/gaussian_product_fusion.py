import numpy as np
import torch


def get_covariances(scale_list, dim):
    # return np.cov(scale_list) 只能获得一个数字而不是矩阵
    mean_vec = np.mean(scale_list, axis=0)
    deviation_mat = scale_list - mean_vec
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    deviation_mat = torch.from_numpy(deviation_mat).to(device)
    cov_mat = torch.mm(deviation_mat.T, deviation_mat) / (dim - 1)
    return cov_mat


def gaussian_product_fusion(dim, means, covariances):
    """
    高斯分布乘积融合函数，输入均值向量和协方差矩阵列表，返回新的均值向量和协方差矩阵。
    """
    # dim 随机变量的维度
    # 计算归一化因子Z
    Z = 1.0
    for cov in covariances:
        Z *= np.sqrt((2 * np.pi) ** dim * np.linalg.det(cov))

    # 计算新分布的均值向量mu
    mu = np.zeros((1, dim))
    for i, mean in enumerate(means):
        cov_inv = np.linalg.inv(covariances[i])
        mu += np.dot(cov_inv, mean)
    mu /= Z

    # 计算新分布的协方差矩阵Sigma
    Sigma_inv = np.zeros((dim, dim))
    for cov in covariances:
        Sigma_inv += np.linalg.inv(cov)
    Sigma = np.linalg.inv(Sigma_inv)

    return mu, Sigma


def gaussian_product_fusion_tensor(shape_means, shape_std):
    shape_means = shape_means.T
    shape_means = shape_means.cpu().numpy()
    shape_std = shape_std.cpu().numpy()
    num, dims = shape_std.shape
    shape_covariances_list = []
    for i in range(num):
        cov_i = np.diag(shape_std[i,:])
        cov_i_inv = np.linalg.inv(cov_i)
        shape_covariances_list.append(cov_i_inv)
    
    S = np.sum(shape_covariances_list,axis=0)
    S_inv = np.linalg.inv(S)

    covariances_mean_product_list = []
    for i in range(num):
        covariances_mean_product = np.dot(shape_covariances_list[i],shape_means[:,i])
        covariances_mean_product_list.append(covariances_mean_product)
    fusion_shape_means = np.dot(S_inv, np.sum(covariances_mean_product_list,axis=0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fusion_shape_means = torch.from_numpy(fusion_shape_means).to(device)
    return fusion_shape_means
