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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import StructuralGaussianModel
from scipy.linalg import polar
import numpy as np

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import trimesh

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))      

def quat_to_rotmat(quat, order='wxyz'):
    """
    quat: (N,4) (float) normalized or unnormalized
    order: 'wxyz' (default) meaning quat = [w,x,y,z], or 'xyzw' meaning [x,y,z,w]
    returns: (N,3,3)
    """
    q = quat.float()
    if order == 'xyzw':
        # convert to w,x,y,z internally
        q = torch.cat([q[:,3:4], q[:,:3]], dim=1)
    # normalize to avoid drift
    q = q / (q.norm(dim=1, keepdim=True) + 1e-12)

    w = q[:,0]; x = q[:,1]; y = q[:,2]; z = q[:,3]

    # compute elements
    ww = w*w; xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z

    # batch rot mats
    R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:,0,0] = ww + xx - yy - zz
    R[:,0,1] = 2*(xy - wz)
    R[:,0,2] = 2*(xz + wy)

    R[:,1,0] = 2*(xy + wz)
    R[:,1,1] = ww - xx + yy - zz
    R[:,1,2] = 2*(yz - wx)

    R[:,2,0] = 2*(xz - wy)
    R[:,2,1] = 2*(yz + wx)
    R[:,2,2] = ww - xx - yy + zz

    return R

def rotmat_to_quat(R, order='wxyz'):
    """
    R: (N,3,3)
    returns: (N,4) quaternion in order specified (default 'wxyz')
    Numerically stable conversion using trace.
    """
    N = R.shape[0]
    device = R.device
    quat = torch.zeros((N,4), device=device, dtype=R.dtype)

    m00 = R[:,0,0]; m11 = R[:,1,1]; m22 = R[:,2,2]
    trace = m00 + m11 + m22

    # case trace > 0
    pos = trace > 0
    if pos.any():
        t = trace[pos] + 1.0
        s = 0.5 / torch.sqrt(t)
        quat[pos,0] = 0.5 * torch.sqrt(t)            # w
        quat[pos,1] = (R[pos,2,1] - R[pos,1,2]) * s  # x
        quat[pos,2] = (R[pos,0,2] - R[pos,2,0]) * s  # y
        quat[pos,3] = (R[pos,1,0] - R[pos,0,1]) * s  # z

    # other cases: find max diagonal
    notpos = ~pos
    if notpos.any():
        # case m00 is largest
        c0 = (m00 > m11) & (m00 > m22)
        mask = notpos & c0
        if mask.any():
            t = 1.0 + m00[mask] - m11[mask] - m22[mask]
            s = 0.5 / torch.sqrt(t)
            quat[mask,1] = 0.5 * torch.sqrt(t)         # x
            quat[mask,0] = (R[mask,2,1] - R[mask,1,2]) * s
            quat[mask,2] = (R[mask,0,1] + R[mask,1,0]) * s
            quat[mask,3] = (R[mask,0,2] + R[mask,2,0]) * s

        # case m11 is largest
        c1 = (m11 > m00) & (m11 > m22)
        mask = notpos & c1
        if mask.any():
            t = 1.0 + m11[mask] - m00[mask] - m22[mask]
            s = 0.5 / torch.sqrt(t)
            quat[mask,2] = 0.5 * torch.sqrt(t)         # y
            quat[mask,0] = (R[mask,0,2] - R[mask,2,0]) * s
            quat[mask,1] = (R[mask,0,1] + R[mask,1,0]) * s
            quat[mask,3] = (R[mask,1,2] + R[mask,2,1]) * s

        # case m22 is largest or remaining
        mask = notpos & ~(c0 | c1)
        if mask.any():
            t = 1.0 + m22[mask] - m00[mask] - m11[mask]
            s = 0.5 / torch.sqrt(t)
            quat[mask,3] = 0.5 * torch.sqrt(t)         # z
            quat[mask,0] = (R[mask,1,0] - R[mask,0,1]) * s
            quat[mask,1] = (R[mask,0,2] + R[mask,2,0]) * s
            quat[mask,2] = (R[mask,1,2] + R[mask,2,1]) * s

    # normalize
    quat = quat / (quat.norm(dim=1, keepdim=True) + 1e-12)

    if order == 'xyzw':
        # convert [w,x,y,z] -> [x,y,z,w]
        quat = torch.cat([quat[:,1:4], quat[:,0:1]], dim=1)

    return quat

def apply_R_to_gaussian_quaternions(gauss_q, R_per_vertex, q_order='wxyz'):
    """
    gauss_q: (N,4) quaternion for each gaussian
    R_per_vertex: (N,3,3) rotation matrix to apply (R_new = R_per_vertex @ R_gauss)
    returns: new_quat (N,4), new_rotmat (N,3,3)
    """
    gauss_q = gauss_q.float()
    R_per_vertex = R_per_vertex.float()

    assert gauss_q.shape[0] == R_per_vertex.shape[0], "batch sizes must match"
    device = gauss_q.device

    # 1) quat -> rotmat
    R_gauss = quat_to_rotmat(gauss_q, order=q_order)  # (N,3,3)

    # 2) compose: apply R_per_vertex on left: R_new = R_per_vertex @ R_gauss
    R_new = torch.matmul(R_per_vertex, R_gauss)

    # 3) optionally convert back to quaternion
    q_new = rotmat_to_quat(R_new, order=q_order)

    return q_new, R_new

def apply_S_to_gaussian_scaling(gauss_scaling, S_per_vertex):
    """
    gauss_scaling: (N,3) 原始 Gaussian scale
    S_per_vertex: (N,3,3) stretch/shear matrix
    returns: new_scaling (N,3)
    """
    # 取對角線作為純伸縮
    S_diag = torch.diagonal(S_per_vertex, dim1=1, dim2=2)  # (N,3)
    
    # element-wise multiply 原始 scaling
    new_scaling = gauss_scaling * S_diag
    
    return new_scaling

def get_barycentric_coordinate(gaussians, p1, p2, p3):
    e1 = gaussians - p1
    e2 = gaussians - p2
    e3 = gaussians - p3

    s1 = torch.norm(torch.cross(e2, e3, dim=1), dim=1, keepdim=True)  # shape (N,1)
    s2 = torch.norm(torch.cross(e1, e3, dim=1), dim=1, keepdim=True)
    s3 = torch.norm(torch.cross(e1, e2, dim=1), dim=1, keepdim=True)

    s = s1 + s2 + s3

    # 重心座標
    coord = torch.cat([s1 / s, s2 / s, s3 / s], dim=1)  # shape (N,3)
    
    return coord

def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices

def compute_vertex_RS(ori_vertices, deform_vertices, faces):
    N = ori_vertices.shape[0]
    device = ori_vertices.device
    
    # 建立鄰居關係
    neighbors = [[] for _ in range(N)]
    for f in faces:
        f = f.tolist()
        neighbors[f[0]].extend([f[1], f[2]])
        neighbors[f[1]].extend([f[0], f[2]])
        neighbors[f[2]].extend([f[0], f[1]])
    neighbors = [list(set(n)) for n in neighbors]

    R_list, S_list = [], []
    for i in range(N):
        if len(neighbors[i]) < 3: # 至少需要三個點
            R_list.append(torch.eye(3, device=device))
            S_list.append(torch.eye(3, device=device))
            continue

        P = (ori_vertices[neighbors[i]] - ori_vertices[i]).T  # (3, k)
        Q = (deform_vertices[neighbors[i]] - deform_vertices[i]).T  # (3, k)

        A = Q @ P.T @ torch.linalg.pinv(P @ P.T)  # (3,3)

        U, sigma, Vh = torch.linalg.svd(A)
        R = U @ Vh
        S = Vh.T @ torch.diag(sigma) @ Vh

        R_list.append(R)
        S_list.append(S)

    return torch.stack(R_list), torch.stack(S_list)  # (N,3,3), (N,3,3)

def polar_decomposition(A: torch.Tensor):
    """
    Compute polar decomposition A = R @ S
    R: orthogonal (rotation/reflection)
    S: symmetric positive semidefinite
    """
    U, sigma, Vh = torch.linalg.svd(A)  # batched SVD
    R = U @ Vh
    S = Vh.transpose(-2, -1) @ torch.diag_embed(sigma) @ Vh
    return R, S

def compute_vertex_RS_Fast(ori_vertices, deform_vertices, faces):
    N = ori_vertices.shape[0]
    device = ori_vertices.device

    # -------------------------------
    # 建立鄰居關係（edge list）
    # -------------------------------
    i0, i1, i2 = faces.T
    edges = torch.cat([
        torch.stack([i0, i1], dim=1),
        torch.stack([i1, i2], dim=1),
        torch.stack([i2, i0], dim=1),
        torch.stack([i1, i0], dim=1),
        torch.stack([i2, i1], dim=1),
        torch.stack([i0, i2], dim=1),
    ], dim=0)  # (6F, 2)

    v_idx = edges[:, 0]  # center vertex
    n_idx = edges[:, 1]  # neighbor vertex

    # -------------------------------
    # P, Q 向量差
    # -------------------------------
    P = (ori_vertices[n_idx] - ori_vertices[v_idx])  # (E,3)
    Q = (deform_vertices[n_idx] - deform_vertices[v_idx])  # (E,3)

    # outer product
    PPt = P[:, :, None] @ P[:, None, :]   # (E,3,3)
    QP  = Q[:, :, None] @ P[:, None, :]   # (E,3,3)

    # -------------------------------
    # scatter 加總到每個 vertex
    # -------------------------------
    PPt_sum = torch.zeros((N, 3, 3), device=device)
    QP_sum  = torch.zeros((N, 3, 3), device=device)

    PPt_sum.index_add_(0, v_idx, PPt)
    QP_sum.index_add_(0, v_idx, QP)

    # -------------------------------
    # 求解 A
    # -------------------------------
    # (N,3,3)
    A = QP_sum @ torch.linalg.pinv(PPt_sum)

    # -------------------------------
    # Polar decomposition (比 SVD 快)
    # -------------------------------
    R, S = polar_decomposition(A)

    # 修正孤立點 (沒有足夠鄰居)
    mask = (PPt_sum.norm(dim=(1,2)) == 0)
    if mask.any():
        eye = torch.eye(3, device=device).expand(N,3,3)
        R[mask] = eye[mask]
        S[mask] = eye[mask]

    return R, S

def verify_RS_on_edges(ori_vertices, deform_vertices, faces, R, S):
    i0, i1, i2 = faces.T
    edges = torch.cat([
        torch.stack([i0, i1], 1),
        torch.stack([i1, i2], 1),
        torch.stack([i2, i0], 1),
        torch.stack([i1, i0], 1),
        torch.stack([i2, i1], 1),
        torch.stack([i0, i2], 1),
    ], dim=0)

    v_idx = edges[:,0]
    n_idx = edges[:,1]

    P = ori_vertices[n_idx] - ori_vertices[v_idx]      # (E,3)
    Q = deform_vertices[n_idx] - deform_vertices[v_idx]# (E,3)

    A = torch.bmm(R, S)

    P_pred = torch.bmm(A[v_idx], P.unsqueeze(-1)).squeeze(-1)

    errors = torch.norm(P_pred - Q, dim=1)

    print("Edge-based RS Verification:")
    print("Mean error:", errors.mean().item())
    print("Median:", errors.median().item())
    print("Max:", errors.max().item())

    return errors

def update_gaussians_MBG(gaussians : StructuralGaussianModel, ori_vert, deform_vert, f):
    R, S = compute_vertex_RS_Fast(ori_vert, deform_vert, f)

    #verify_RS_on_edges(ori_vert, deform_vert, f, R, S)

    if not isinstance(deform_vert, torch.Tensor):
        vertices = torch.tensor(deform_vert, dtype=torch.float32, device="cuda")
    else:
        vertices = deform_vert.detach().clone().float()

    if not isinstance(f, torch.Tensor):
        faces = torch.tensor(f, dtype=torch.long, device="cuda")
    else:
        faces = f.detach().clone().long()

    v0 = deform_vert[f[:, 0]]
    v1 = deform_vert[f[:, 1]]
    v2 = deform_vert[f[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    normals = torch.cross(e1, e2, dim=1)
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-9)

    coord = gaussians.get_bc
    face_idx = gaussians.get_fid.squeeze(1)
    v_idx = faces[face_idx]  # (N,3)

    gaussians.vertices = deform_vert
    gaussians.normals = normals
    
    cur_rot = R[v_idx]
    weight_g_rs = coord.unsqueeze(2).unsqueeze(3)
    g_delta_rot = torch.sum(weight_g_rs * cur_rot, dim=1)

    cur_scale = S[v_idx]
    g_delta_s = torch.sum(weight_g_rs * cur_scale, dim=1)

    q_new, R_new = apply_R_to_gaussian_quaternions(gaussians.get_app_rotation, g_delta_rot, q_order='wxyz')
    s_new = apply_S_to_gaussian_scaling(gaussians.get_app_scaling, g_delta_s)

    gaussians._added_rotation = q_new
    gaussians._added_scaling = torch.log(s_new)

    q_geo_new, R_geo_new = apply_R_to_gaussian_quaternions(gaussians.get_geo_rotation, R, q_order='wxyz')
    s_geo_new = apply_S_to_gaussian_scaling(gaussians.get_geo_scaling, S)

    gaussians._xyz = deform_vert
    gaussians._rotation = q_geo_new
    gaussians._scaling = torch.log(s_geo_new)

def save_ply(scene: Scene, name):
    print("\nSaving Deformed Gaussians")
    scene.save_deform(name)

def deformation(dataset : ModelParams, iteration : int, pipeline : PipelineParams, deform_path : str, separate_sh: bool):
    with torch.no_grad():
        gaussians = StructuralGaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        deform_mesh = trimesh.load(deform_path, force='mesh')
        deform_vertices = torch.tensor(deform_mesh.vertices).float().cuda()

        faces = gaussians.face_idx
        verties = gaussians.vertices

        update_gaussians_MBG(gaussians, verties, deform_vertices, faces)

        filename = os.path.basename(deform_path) 
        name = os.path.splitext(filename)[0] 
        number = name.split('_')[-1]

        save_ply(scene, number)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, name, scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform_path", type=str, default="")
    args = get_combined_args(parser)
    print("Deforming " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    deformation(model.extract(args), args.iteration, pipeline.extract(args), args.deform_path, SPARSE_ADAM_AVAILABLE)