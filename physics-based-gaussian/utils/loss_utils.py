#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

def circumradius(point1, point2, point3):
    """
    point1, point2, point3: Tensor, shape (N, 3)
        N 個三角形的三個頂點
    return: Tensor, shape (N,)
        每個三角形的外接圓半徑 (circumradius)
    """
    AB = point2 - point1  # 邊向量
    AC = point3 - point1

    # 向量叉積
    cross_product = torch.cross(AB, AC, dim=1)  # shape (N, 3)

    # 三角形面積 = 0.5 * ||AB × AC||
    areas = torch.norm(cross_product, dim=1) * 0.5  # shape (N,)

    # 三邊長
    a = torch.norm(point2 - point3, dim=1)  # |BC|
    b = torch.norm(point1 - point3, dim=1)  # |AC|
    c = torch.norm(point1 - point2, dim=1)  # |AB|

    # 外接圓半徑公式: R = (a * b * c) / (4 * 面積)
    radius = (a * b * c) / (4.0 * areas + 1e-8)  # 避免除零

    return radius

def mesh_restrict_loss(scale, point1, point2, point3, weight=10.0):
    """
    scale: Tensor, shape (N, M)
        每個 g 對應的 scale 值
    point1, point2, point3: Tensor, shape (N, 3)
        三角形頂點
    weight: float
        權重係數
    """
    max_s = torch.max(scale, dim=1).values  # shape (N,)
    r = circumradius(point1, point2, point3)  # shape (N,)

    loss = max_s - weight * r
    loss = torch.clamp(loss, min=0.0)  # ReLU
    
    return loss.sum()

def mesh_vertex_restrict_loss(scale, r, weight=10.0):
    """
    scale: Tensor, shape (N, M)
        每個 g 對應的 scale 值
    point1, point2, point3: Tensor, shape (N, 3)
        三角形頂點
    weight: float
        權重係數
    """
    max_s = torch.max(scale, dim=1).values  # shape (N,)

    loss = max_s - weight * r
    loss = torch.clamp(loss, min=0.0)  # ReLU
    
    return loss.mean()
