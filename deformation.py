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
from scene import DualGaussianModel
import numpy as np

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import trimesh
import pyACAP
import tempfile

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "deform")

    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))      

def update_gaussians_MBG(gaussians : DualGaussianModel, ori_vert, deform_vert, f):

    vertices_np = ori_vert.cpu().numpy()
    faces_np = f.cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)

    with tempfile.NamedTemporaryFile(suffix=".obj") as tmpfile:
        mesh.export(tmpfile.name)
        ACAPtool = pyACAP.pyACAP(tmpfile.name)

    # ---- ACAP ----
    ori_vert_np = ori_vert.detach().cpu().numpy().astype(np.float64)
    deform_vert_np = deform_vert.detach().cpu().numpy().astype(np.float64)

    R_np, S_np = ACAPtool.GetRS(ori_vert_np, deform_vert_np, True, os.cpu_count()//2)

    R = torch.tensor(R_np.reshape((-1, 3, 3)), device='cuda', dtype=torch.float32)[:ori_vert.shape[0]]
    S = torch.tensor(S_np.reshape((-1, 3, 3)), device='cuda', dtype=torch.float32)[:ori_vert.shape[0]]

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
    
    cur_rot = R[v_idx]
    weight_g_rs = coord.unsqueeze(2).unsqueeze(3)
    g_delta_rot = torch.sum(weight_g_rs * cur_rot, dim=1)

    cur_scale = S[v_idx]
    g_delta_s = torch.sum(weight_g_rs * cur_scale, dim=1)

    deform_rot = torch.cat([R, g_delta_rot], dim=0).transpose(1,2)
    deform_s = torch.cat([S, g_delta_s], dim=0)

    print(deform_rot.shape)
    print(deform_s.shape)

    g_delta_rs = torch.matmul(deform_rot, deform_s)

    gaussians.gaussian_deform_cov = torch.matmul(torch.matmul(g_delta_rs, gaussians.get_covariance()), g_delta_rs.transpose(1,2))

    gaussians.vertices = deform_vert
    gaussians.normals = normals
    gaussians._xyz = deform_vert

def deformation(dataset : ModelParams, iteration : int, pipeline : PipelineParams, deform_path : str, skip_train : bool, skip_test : bool, skip_val : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = DualGaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        deform_mesh = trimesh.load(deform_path, force='mesh', process=False)
        deform_vertices = torch.tensor(deform_mesh.vertices).float().cuda()

        faces = gaussians.face_idx
        verties = gaussians.vertices

        update_gaussians_MBG(gaussians, verties, deform_vertices, faces)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_val:
             render_set(dataset.model_path, "val", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform_path", type=str, default="")
    args = get_combined_args(parser)
    print("Deforming " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    deformation(model.extract(args), args.iteration, pipeline.extract(args), args.deform_path, args.skip_train, args.skip_test, args.skip_val, SPARSE_ADAM_AVAILABLE)