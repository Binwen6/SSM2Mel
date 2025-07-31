# PYTORCH version of the vlaai original code.
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def l1_loss(pred, target):
    """L1损失函数"""
    return torch.mean(torch.abs(pred - target))

def pearson_loss(pred, target):
    """皮尔逊相关系数损失"""
    pred_mean = torch.mean(pred, dim=0, keepdim=True)
    target_mean = torch.mean(target, dim=0, keepdim=True)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    numerator = torch.sum(pred_centered * target_centered, dim=0)
    denominator = torch.sqrt(torch.sum(pred_centered ** 2, dim=0) * torch.sum(target_centered ** 2, dim=0))
    
    # 避免除零
    denominator = torch.clamp(denominator, min=1e-8)
    
    correlation = numerator / denominator
    return -torch.mean(correlation)

def pearson_metric(pred, target):
    """皮尔逊相关系数评估指标"""
    pred_mean = torch.mean(pred, dim=0, keepdim=True)
    target_mean = torch.mean(target, dim=0, keepdim=True)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    numerator = torch.sum(pred_centered * target_centered, dim=0)
    denominator = torch.sqrt(torch.sum(pred_centered ** 2, dim=0) * torch.sum(target_centered ** 2, dim=0))
    
    # 避免除零
    denominator = torch.clamp(denominator, min=1e-8)
    
    correlation = numerator / denominator
    return torch.mean(correlation)

def band_diversity_loss(pred, target=None):
    """
    频带多样性损失函数
    鼓励不同频带学习不同的特征，减少频带间的相关性
    """
    if len(pred.shape) != 3:
        return torch.tensor(0.0, device=pred.device)
    
    # pred shape: [batch, time, bands]
    batch_size, time_steps, num_bands = pred.shape
    
    # 计算频带间的相关性矩阵
    pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)  # 中心化
    
    # 计算每个频带的方差
    band_variances = torch.var(pred_centered, dim=1)  # [batch, bands]
    
    # 计算频带间的协方差
    cov_matrix = torch.zeros(batch_size, num_bands, num_bands, device=pred.device)
    for i in range(num_bands):
        for j in range(num_bands):
            cov_matrix[:, i, j] = torch.mean(pred_centered[:, :, i] * pred_centered[:, :, j], dim=1)
    
    # 计算相关系数矩阵
    std_matrix = torch.sqrt(torch.outer(band_variances.mean(dim=0), band_variances.mean(dim=0)))
    std_matrix = torch.clamp(std_matrix, min=1e-8)
    corr_matrix = cov_matrix.mean(dim=0) / std_matrix
    
    # 计算非对角元素的平均相关性（我们希望这个值越小越好）
    mask = torch.ones_like(corr_matrix) - torch.eye(num_bands, device=pred.device)
    off_diagonal_corr = torch.sum(corr_matrix * mask) / torch.sum(mask)
    
    # 多样性损失：鼓励频带间相关性降低
    diversity_loss = torch.abs(off_diagonal_corr)
    
    return diversity_loss

def combined_loss(pred, target, alpha=0.1, beta=0.05):
    """
    组合损失函数
    alpha: 皮尔逊损失权重
    beta: 频带多样性损失权重
    """
    l1 = l1_loss(pred, target)
    pearson = pearson_loss(pred, target)
    diversity = band_diversity_loss(pred)
    
    total_loss = l1 + alpha * pearson + beta * diversity
    
    return total_loss, {
        'l1_loss': l1.item(),
        'pearson_loss': pearson.item(),
        'diversity_loss': diversity.item(),
        'total_loss': total_loss.item()
    }