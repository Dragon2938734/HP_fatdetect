import numpy as np
import torch


def get_covariances(scale_list):
    return np.cov(scale_list)


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


def gaussian_product_fusion_tensor(means, covariances):
    means = means.cpu().numpy()
    covariances = covariances.cpu().numpy()
    rows, cols = means.shape
    means_fusion = []
    for i in range(rows):
        cov_i = get_covariances(covariances[i,:])
        import ipdb
        ipdb.set_trace()
        mu, _ = gaussian_product_fusion(cols, means[i,:], cov_i)
        means_fusion.append(mu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    means_fusion = torch.from_numpy(means_fusion).to(device)
    return means_fusion

