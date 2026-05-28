import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import compute_vertex_radii_approx

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

import trimesh
import time


def _normals_to_quaternions(normals):
    """
    Shortest rotation from canonical z=[0,0,1] to each face normal.
    Aligns Gaussian local z-axis with the face normal so splats lie flat on the mesh.
    normals : (N, 3) unit-length face normals (CUDA tensor)
    Returns : (N, 4) quaternions [w, x, y, z]
    """
    nz = normals[:, 2].clamp(-1.0, 1.0)
    cos_half = torch.sqrt(((1.0 + nz) / 2.0).clamp(min=0.0))   # w = cos(θ/2)
    sin_half = torch.sqrt(((1.0 - nz) / 2.0).clamp(min=0.0))   # sin(θ/2)

    # cross([0,0,1], n) = [-ny, nx, 0] — normalize to get rotation axis
    ax = -normals[:, 1]
    ay =  normals[:, 0]
    a_norm = torch.sqrt(ax * ax + ay * ay).clamp(min=1e-8)
    ax = ax / a_norm
    ay = ay / a_norm

    quats = torch.stack([cos_half, ax * sin_half, ay * sin_half,
                         torch.zeros_like(cos_half)], dim=1)

    # n ≈ [0,0,-1]: axis is undefined; use 180° rotation about x
    anti = nz < -0.9999
    if anti.any():
        quats[anti] = torch.tensor([0.0, 1.0, 0.0, 0.0],
                                   device=normals.device, dtype=normals.dtype)
    return quats


class StructuralGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.bc_activation = torch.softmax
        self.distance_activation = torch.sigmoid

    def __init__(self, sh_degree, optimizer_type="default", mesh_path="mesh_path"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  

        # geometry gaussians
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # appearance gaussians
        self._added_xyz = None
        self._added_features_dc = None
        self._added_features_rest = None
        self._added_scaling = None
        self._added_rotation = None
        self._added_opacity = None

        self._bc = None # (N, 3)
        self._distance = None # (N, 1)
        self.fid = None # (N, 1)

        self.gaussian_deform_cov = None

        self.mesh_path = mesh_path
        self.normals = torch.empty(0) # (F, 3)
        self.vertices = torch.empty(0) # (F, 3)
        self.face_idx = torch.empty(0) # (F, 3)
        self.vertex_radius = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)


    @property
    def get_bc(self):
        return self.bc_activation(self._bc,dim=1)
    
    @property
    def get_distance(self):
        return self.distance_activation(self._distance)
    
    @property
    def get_fid(self):
        return self.fid
    
    @property
    def get_app_scaling(self):
        return self.scaling_activation(self._added_scaling)

    @property
    def get_app_rotation(self):
        return self.rotation_activation(self._added_rotation)

    @property
    def get_geo_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_geo_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_scaling(self):
        if self._added_scaling is not None:
            return self.scaling_activation(torch.cat((self._scaling, self._added_scaling), dim=0))
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        if self._added_rotation is not None:
            return self.rotation_activation(torch.cat((self._rotation, self._added_rotation), dim=0))
        return self.rotation_activation(self._rotation)
        
    @property
    def get_xyz(self):
        pos = self._xyz
        if self._added_xyz is not None:
            pos = torch.cat((self._xyz, self._added_xyz), dim=0)
        if self._bc is not None:
            bc = self.bc_activation(self._bc,dim=1)

            v_idx = self.face_idx[self.fid[:, 0]]
            v0 = self.vertices[v_idx[:, 0]]
            v1 = self.vertices[v_idx[:, 1]]
            v2 = self.vertices[v_idx[:, 2]]
            n = self.normals[self.fid[:, 0]]

            pro_pos = bc[:, 0:1] * v0 + bc[:, 1:2] * v1 + bc[:, 2:3] * v2
            offset = self.distance_activation(self._distance) * n
            added_xyz = pro_pos + offset

            pos = torch.cat((self._xyz, added_xyz), dim=0)
        return pos
    
    @property
    def get_add_xyz(self):

        if self._bc is not None:
            bc = self.bc_activation(self._bc,dim=1)

            v_idx = self.face_idx[self.fid[:, 0]]
            v0 = self.vertices[v_idx[:, 0]]
            v1 = self.vertices[v_idx[:, 1]]
            v2 = self.vertices[v_idx[:, 2]]
            n = self.normals[self.fid[:, 0]]

            pro_pos = bc[:, 0:1] * v0 + bc[:, 1:2] * v1 + bc[:, 2:3] * v2
            offset = self.distance_activation(self._distance) * n
            added_xyz = pro_pos + offset

            return added_xyz
        return self._added_xyz
        
    @property
    def get_features(self):
        if self._added_features_dc is not None and self._added_features_rest is not None:
            features_dc = torch.cat((self._features_dc, self._added_features_dc), dim=0)
            features_rest = torch.cat((self._features_rest, self._added_features_rest), dim=0)
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)  
        
    @property
    def get_opacity(self):
        if self._added_opacity is not None:
            return self.opacity_activation(torch.cat((self._opacity, self._added_opacity), dim=0))
        return self.opacity_activation(self._opacity)
    
    @property
    def get_geo_opacity(self):
        return self.opacity_activation(self._opacity)
        
    @property
    def get_vertex(self):
        v_idx = self.face_idx[self.fid[:, 0]]
        v0 = self.vertices[v_idx[:, 0]]
        v1 = self.vertices[v_idx[:, 1]]
        v2 = self.vertices[v_idx[:, 2]]

        return v0, v1, v2

    @property
    def get_vertex_radius(self):
        return self.vertex_radius

    @property
    def get_app_vertex_radius(self):
        """Per-app-Gaussian radius: max 1-ring vertex radius of its bound face."""
        if self.vertex_radius.shape[0] == 0 or self.fid is None:
            return None
        v_idx = self.face_idx[self.fid[:, 0]]
        r0 = self.vertex_radius[v_idx[:, 0]]
        r1 = self.vertex_radius[v_idx[:, 1]]
        r2 = self.vertex_radius[v_idx[:, 2]]
        return torch.stack([r0, r1, r2], dim=1).max(dim=1).values

    @property
    def get_app_face_longest_edge(self):
        """Per-app-Gaussian longest edge of its bound face."""
        if self.fid is None or self._added_scaling is None:
            return None
        face_ids = self.fid[:, 0]
        v_idx = self.face_idx[face_ids]
        v0 = self.vertices[v_idx[:, 0]]
        v1 = self.vertices[v_idx[:, 1]]
        v2 = self.vertices[v_idx[:, 2]]
        e01 = (v1 - v0).norm(dim=1)
        e02 = (v2 - v0).norm(dim=1)
        e12 = (v2 - v1).norm(dim=1)
        return torch.stack([e01, e02, e12], dim=1).max(dim=1).values  # (N_app,)

    def clamp_app_scale_floor(self, ratio=0.5):
        """Hard-clamp both in-plane app Gaussian scales to >= ratio × face longest edge."""
        if self.fid is None or self._added_scaling is None:
            return
        with torch.no_grad():
            face_longest = self.get_app_face_longest_edge
            s_min = (face_longest * ratio).clamp(min=1e-4)
            for dim in [0, 1]:
                cur_s = self.scaling_activation(self._added_scaling[:, dim])
                too_small = cur_s < s_min
                if too_small.any():
                    self._added_scaling.data[too_small, dim] += torch.log(
                        s_min[too_small] / cur_s[too_small].clamp(min=1e-8)
                    )

    def get_deform_covariance(self):
        return strip_symmetric(self.gaussian_deform_cov)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):

        mesh = trimesh.load(self.mesh_path, process=False)

        self.vertices = torch.tensor(mesh.vertices).float().cuda()
        self.face_idx = torch.tensor(mesh.faces).long().cuda()

        v0 = self.vertices[self.face_idx[:, 0]]
        v1 = self.vertices[self.face_idx[:, 1]]
        v2 = self.vertices[self.face_idx[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        normals = torch.cross(e1, e2, dim=1)
        normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-9)

        self.normals = normals
        self.vertex_radius = compute_vertex_radii_approx(self.vertices, self.face_idx)

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_s1_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def training_s2_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self._added_xyz is not None:
            self._added_xyz.requires_grad_(True)
            self._added_features_dc.requires_grad_(True)
            self._added_features_rest.requires_grad_(True)
            self._added_opacity.requires_grad_(True)
            self._added_scaling.requires_grad_(True)
            self._added_rotation.requires_grad_(True)

            l += [
                {'params': [self._added_features_dc], 'lr': training_args.feature_lr, "name": "added_f_dc"},
                {'params': [self._added_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "added_f_rest"},
                {'params': [self._added_opacity], 'lr': training_args.opacity_lr, "name": "added_opacity"},
                {'params': [self._added_scaling], 'lr': training_args.scaling_lr, "name": "added_scaling"},
                {'params': [self._added_rotation], 'lr': training_args.rotation_lr, "name": "added_rotation"},
                {'params': [self._added_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "added_xyz"},
            ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    def training_s3_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self._bc is not None:
            self._bc.requires_grad_(True)
            self._distance.requires_grad_(True)
            self._added_features_dc.requires_grad_(True)
            self._added_features_rest.requires_grad_(True)
            self._added_opacity.requires_grad_(True)
            self._added_scaling.requires_grad_(True)
            self._added_rotation.requires_grad_(True)

            l += [
                {'params': [self._added_features_dc], 'lr': training_args.feature_lr, "name": "added_f_dc"},
                {'params': [self._added_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "added_f_rest"},
                {'params': [self._added_opacity], 'lr': training_args.opacity_lr, "name": "added_opacity"},
                {'params': [self._added_scaling], 'lr': training_args.scaling_lr, "name": "added_scaling"},
                {'params': [self._added_rotation], 'lr': training_args.rotation_lr, "name": "added_rotation"},
                {'params': [self._bc], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "bc"},
                {'params': [self._distance], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "distance"},
            ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.bc_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "added_xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            if param_group["name"] == "bc":
                lr = self.bc_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "distance":
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def construct_list_of_app_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'ca', 'cb', 'cc', 'd', 'face_id']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        if self._added_xyz is not None:
            xyz = self.get_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            features_dc = torch.cat((self._features_dc, self._added_features_dc), dim=0)
            features_rest = torch.cat((self._features_rest, self._added_features_rest), dim=0)
            f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = torch.cat((self._opacity, self._added_opacity), dim=0).detach().cpu().numpy()
            scale = torch.cat((self._scaling, self._added_scaling), dim=0).detach().cpu().numpy()
            rotation = torch.cat((self._rotation, self._added_rotation), dim=0).detach().cpu().numpy()
        else :
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def compute_barycentric_and_offset(self, xyz, face_idx, vertices, faces, face_normals, p_on_surface, clamp_bary=True, eps=1e-12):
        """
        計算每個點相對於其最近面的 barycentric 權重 (w1, w2, w3)
        與法向量方向上的 offset t。

        Args:
            xyz: (N,3) torch.Tensor, 每個點的原始位置
            face_idx: (N,) torch.LongTensor, 每個點對應的最近面索引
            vertices: (V,3) torch.Tensor, mesh 頂點座標
            faces: (F,3) torch.LongTensor, mesh 面的頂點索引
            face_normals: (F,3) torch.Tensor, 每個面的法向量（已單位化）
            p_on_surface: (N,3) torch.Tensor, 每個點在 mesh 上的最近點
            clamp_bary: bool, 是否對 barycentric 權重做 clamp 與 normalize
            eps: float, 防止除以零的小值

        Returns:
            w: (N,3) barycentric 權重 (w1,w2,w3)
            t: (N,) 沿法向量的偏移量
        """

        # 1️⃣ 取出每個面三個頂點與法向量
        v_idx = faces[face_idx]  # (N,3)
        v0 = vertices[v_idx[:, 0]]
        v1 = vertices[v_idx[:, 1]]
        v2 = vertices[v_idx[:, 2]]
        n = face_normals[face_idx]

        # 2️⃣ 計算 barycentric (u,v,w)
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p_on_surface - v0

        d00 = (v0v1 * v0v1).sum(dim=1)
        d01 = (v0v1 * v0v2).sum(dim=1)
        d11 = (v0v2 * v0v2).sum(dim=1)
        d20 = (v0p * v0v1).sum(dim=1)
        d21 = (v0p * v0v2).sum(dim=1)

        denom = (d00 * d11 - d01 * d01).clamp(min=eps)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        bary = torch.stack([u, v, w], dim=1)

        if clamp_bary:
            bary = bary.clamp(min=0.0, max=1.0)
            bary_sum = bary.sum(dim=1, keepdim=True).clamp(min=eps)
            bary = bary / bary_sum

        offset_vec = xyz - p_on_surface
        t = (n * offset_vec).sum(dim=1)

        recon = bary[:,0:1]*v0 + bary[:,1:2]*v1 + bary[:,2:3]*v2 + t.unsqueeze(1)*n  # (N,3)

        return bary, t

    def find_closet_faces(self):
        device = self._xyz.device

        mesh = trimesh.load(self.mesh_path, process=False)
        points_np = self._added_xyz.detach().cpu().numpy()
        
        # find closest faces
        closest_pts_np, dists_np, face_idx = trimesh.proximity.closest_point(mesh, points_np)

        closest_pts = torch.from_numpy(closest_pts_np).to(device)
        face_idx = torch.from_numpy(face_idx).long().to(device)
        
        vertices = self.vertices
        faces = self.face_idx
        face_normals = self.normals

        bary, t = self.compute_barycentric_and_offset(
            xyz=self._added_xyz,
            face_idx=face_idx,
            vertices=vertices,
            faces=faces,
            face_normals=face_normals,
            p_on_surface=closest_pts
        )

        self._bc = bary.float().detach()
        # Initialize all app Gaussians very close to the surface.
        # sigmoid(-6.9078) ≈ 0.001, so offset = 0.001 * face_normal ≈ on surface.
        # Stage 2 distances (|t|) can be large (0.3-0.8); using them would leave Gaussians
        # floating far above the mesh. Starting from near-zero lets dis_loss keep them close.
        _DIST_INIT_LOGIT = -6.9078  # logit(0.001)
        self._distance = torch.full(
            (bary.shape[0], 1), _DIST_INIT_LOGIT,
            dtype=torch.float, device=device
        ).detach()
        self.fid = face_idx.unsqueeze(1).long()

        # Boundary face mask: faces adjacent to a sharp edge (angle > 20°) are
        # exempt from snap_rotations_to_normals so their Gaussians can freely
        # orient to cover the visible seam between differently-angled faces.
        _BOUNDARY_THRESH = np.radians(20.0)
        adj_faces  = mesh.face_adjacency          # (E, 2)
        adj_angles = mesh.face_adjacency_angles   # (E,)  radians
        sharp      = adj_angles > _BOUNDARY_THRESH
        bnd_indices = np.unique(adj_faces[sharp].ravel())
        bnd_mask = np.zeros(len(mesh.faces), dtype=bool)
        bnd_mask[bnd_indices] = True
        self._boundary_face_mask = torch.from_numpy(bnd_mask).to(device)
        print(f"  Boundary faces (>{np.degrees(_BOUNDARY_THRESH):.0f}°): "
              f"{bnd_mask.sum()} / {len(mesh.faces)}")

        # Align each app Gaussian's local z-axis with its face normal so that
        # the splat lies flat on the mesh surface (critical for sharp texture projection).
        face_normals_assigned = self.normals[face_idx]           # (N_app, 3)
        aligned_quats = _normals_to_quaternions(face_normals_assigned)
        with torch.no_grad():
            self._added_rotation.data.copy_(aligned_quats)

        # After rotation aligns local-z to face normal, dims 0 and 1 are in-plane.
        # Clamp up any Gaussians whose in-plane scale is too small to cover their face.
        # (Stage 2 learning can leave some Gaussians very small; mrloss has no lower bound.)
        with torch.no_grad():
            v_idx = self.face_idx[face_idx]                              # (N_app, 3)
            r0 = self.vertex_radius[v_idx[:, 0]]
            r1 = self.vertex_radius[v_idx[:, 1]]
            r2 = self.vertex_radius[v_idx[:, 2]]
            face_r = torch.stack([r0, r1, r2], dim=1).max(dim=1).values  # (N_app,)
            s_floor = face_r * 0.5
            # Use min(s_x, s_y) so both axes are checked independently.
            s0 = self.scaling_activation(self._added_scaling[:, 0])
            s1 = self.scaling_activation(self._added_scaling[:, 1])
            # Scale up dim 0 if too small
            too_small_0 = s0 < s_floor
            if too_small_0.any():
                self._added_scaling.data[too_small_0, 0] += torch.log(
                    s_floor[too_small_0] / s0[too_small_0].clamp(min=1e-8)
                )
            # Scale up dim 1 if too small
            too_small_1 = s1 < s_floor
            if too_small_1.any():
                self._added_scaling.data[too_small_1, 1] += torch.log(
                    s_floor[too_small_1] / s1[too_small_1].clamp(min=1e-8)
                )
            n_fixed = (too_small_0 | too_small_1).sum().item()
            if n_fixed > 0:
                print(f"  Stage 3 init: rescaled {n_fixed} undersized app Gaussians "
                      f"(in-plane scale < 50% of face vertex_radius)")

    def snap_rotations_to_normals(self):
        """
        Hard-reset app Gaussian rotations so each z-axis equals its face normal.
        Boundary faces (adjacent to a sharp edge) are exempt: their Gaussians are
        free to rotate and find an orientation that covers the visible seam.
        """
        if self.fid is None or self._added_rotation is None:
            return

        face_ids = self.fid[:, 0]  # (N_app,)

        bnd = getattr(self, '_boundary_face_mask', None)
        if bnd is not None:
            snap_mask = ~bnd[face_ids]           # snap only non-boundary Gaussians
        else:
            snap_mask = torch.ones(face_ids.shape[0], dtype=torch.bool, device=face_ids.device)

        if snap_mask.any():
            aligned_quats = _normals_to_quaternions(self.normals[face_ids[snap_mask]])
            self._added_rotation.data[snap_mask] = aligned_quats

    def save_geo_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
            
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        vertex_element = PlyElement.describe(elements, 'vertex')

        faces_np = (#存face
            self.face_idx.detach()
            .cpu()
            .numpy()
            .astype(np.int32)
        )  # (F, 3)

        face_dtype = [("vertex_indices", "i4", (3,))]
        face_elements = np.empty(faces_np.shape[0], dtype=face_dtype)
        face_elements["vertex_indices"] = faces_np

        face_element = PlyElement.describe(face_elements, "face")
                
        PlyData([vertex_element, face_element]).write(path)

    def save_app_ply(self, path):
        mkdir_p(os.path.dirname(path))

        if self._bc is None:
            self.find_closet_faces()

        xyz = self.get_add_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._added_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._added_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._added_opacity.detach().cpu().numpy()
        scale = self._added_scaling.detach().cpu().numpy()
        rotation = self._added_rotation.detach().cpu().numpy()

        bc = self._bc.detach().cpu().numpy()
        d = self._distance.detach().cpu().numpy()
        fid = self.fid.detach().cpu().numpy()
            
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_app_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, bc, d, fid, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        """
        將舊與新 Gaussians 的 opacity 重設為 0.01，
        並保持 optimizer 可追蹤。
        """
        # reset 舊的
        opacity_old_new = self.inverse_opacity_activation(
            torch.min(self.opacity_activation(self._opacity), torch.ones_like(self._opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacity_old_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        # reset 新增的 (若存在)
        if self._added_opacity is not None:
            opacity_new_new = self.inverse_opacity_activation(
                torch.min(self.opacity_activation(self._added_opacity), torch.ones_like(self._added_opacity) * 0.01)
            )
            optimizable_tensors = self.replace_tensor_to_optimizer(opacity_new_new, "added_opacity")
            self._added_opacity = optimizable_tensors["added_opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_geo_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        faces = np.array(
            [f for f in plydata.elements[1]["vertex_indices"]],
            dtype=np.int64
        )  # shape: (F,3)
        
        self.vertices = torch.tensor(xyz, dtype=torch.float, device="cuda")
        self.face_idx = torch.tensor(faces, dtype=torch.long, device="cuda")

        v0 = self.vertices[self.face_idx[:, 0]]
        v1 = self.vertices[self.face_idx[:, 1]]
        v2 = self.vertices[self.face_idx[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        normals = torch.cross(e1, e2, dim=1)
        normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-9)

        self.normals = normals

        self.active_sh_degree = self.max_sh_degree

    def load_app_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        bc = np.stack((np.asarray(plydata.elements[0]["ca"]),
                        np.asarray(plydata.elements[0]["cb"]),
                        np.asarray(plydata.elements[0]["cc"])),  axis=1)
        
        distance = np.asarray(plydata.elements[0]["d"])[..., np.newaxis]
        face_idx = np.asarray(plydata.elements[0]["face_id"])[..., np.newaxis]

        self._added_xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._added_features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._added_opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._bc = torch.tensor(bc, dtype=torch.float, device="cuda")
        self._distance = torch.tensor(distance, dtype=torch.float, device="cuda")
        self.fid = torch.tensor(face_idx, dtype=torch.long, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, tensors_name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)

            name = group["name"]
            if name not in tensors_name:
                continue

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def prune_points(self, mask):
        name = ["bc", "distance", "added_f_dc", "added_f_rest", "added_opacity", "added_scaling", "added_rotation"]

        valid_added_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_added_mask, name)

        self._bc = optimizable_tensors["bc"]
        self._distance = optimizable_tensors["distance"]
        self._added_features_dc = optimizable_tensors["added_f_dc"]
        self._added_features_rest = optimizable_tensors["added_f_rest"]
        self._added_opacity = optimizable_tensors["added_opacity"]
        self._added_scaling = optimizable_tensors["added_scaling"]
        self._added_rotation = optimizable_tensors["added_rotation"]

        self.fid = self.fid[valid_added_mask]

        valid_points_mask = torch.cat((torch.ones(self.vertices.shape[0], device="cuda", dtype=bool), valid_added_mask))

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            name = group["name"]

            if name not in tensors_dict:
                continue

            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1   

    def densify_and_split_init(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self._added_xyz = new_xyz.detach()
        self._added_scaling = torch.nn.Parameter(new_scaling)
        self._added_rotation = torch.nn.Parameter(new_rotation)
        self._added_features_dc = torch.nn.Parameter(new_features_dc)
        self._added_features_rest = torch.nn.Parameter(new_features_rest)
        self._added_opacity = torch.nn.Parameter(new_opacity)

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_prune_init(self, max_grad, min_opacity, extent, max_screen_size, radii): 
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_split_init(grads, max_grad, extent)

        torch.cuda.empty_cache()    

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        #self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        torch.cuda.empty_cache()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        if(selected_pts_mask.sum().item()==0):
            return
        
        features_dc = torch.cat((self._features_dc, self._added_features_dc), dim=0)
        features_rest = torch.cat((self._features_rest, self._added_features_rest), dim=0)
        opacities = torch.cat((self._opacity, self._added_opacity), dim=0)
        scale = torch.cat((self._scaling, self._added_scaling), dim=0)
        rotation = torch.cat((self._rotation, self._added_rotation), dim=0)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = opacities[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"added_xyz": new_xyz,
        "added_f_dc": new_features_dc,
        "added_f_rest": new_features_rest,
        "added_opacity": new_opacities,
        "added_scaling" : new_scaling,
        "added_rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._added_xyz = optimizable_tensors["added_xyz"]
        self._added_features_dc = optimizable_tensors["added_f_dc"]
        self._added_features_rest = optimizable_tensors["added_f_rest"]
        self._added_opacity = optimizable_tensors["added_opacity"]
        self._added_scaling = optimizable_tensors["added_scaling"]
        self._added_rotation = optimizable_tensors["added_rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def split_neighbor_gaussians(self, mask):
        selected_faces_mask = mask[self.face_idx].any(dim=1)
        selected_face_ids = torch.nonzero(selected_faces_mask, as_tuple=True)[0]

        return selected_face_ids

    def densify_and_prune_sg3(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_split_sg3(grads, max_grad, extent)

        # Prune low-opacity app Gaussians (mirrors Stage-2 densify_and_prune)
        n_geo = self.vertices.shape[0]
        all_opacity = self.get_opacity          # (n_geo + n_app,)
        app_opacity = all_opacity[n_geo:]       # (n_app,)
        prune_mask = (app_opacity < min_opacity).squeeze()
        if max_screen_size is not None:
            big_vs = self.max_radii2D[n_geo:] > max_screen_size
            big_ws = self.get_app_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_vs | big_ws
        if prune_mask.any():
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify_and_split_sg3(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        if(selected_pts_mask.sum().item()==0):
            return
        
        selected_vertex_mask = selected_pts_mask[:self.vertices.shape[0]]
        selected_added_mask = selected_pts_mask[self.vertices.shape[0]:]

        # Split from geometry gaussians
        selected_fid = self.split_neighbor_gaussians(selected_vertex_mask)
        selected_faces = self.face_idx[selected_fid]
        new_bc = torch.ones_like(selected_faces).float().cuda() / 3
        _DIST_INIT_LOGIT = -6.9078  # logit(0.001): sigmoid gives 0.001, very close to surface
        new_distance = torch.full((selected_fid.shape[0], 1), _DIST_INIT_LOGIT, dtype=torch.float, device="cuda")
        new_fid = selected_fid.unsqueeze(1).long().cuda()

        # Init scale from face vertex_radius instead of inherited geo scale.
        # Geo scale shrinks across densification rounds; inheriting it causes
        # new app Gaussians to also be microscopic and produce holes.
        r0 = self.vertex_radius[selected_faces[:, 0]]
        r1 = self.vertex_radius[selected_faces[:, 1]]
        r2 = self.vertex_radius[selected_faces[:, 2]]
        face_r = torch.stack([r0, r1, r2], dim=1).max(dim=1).values
        s_xy = (face_r * 0.5).clamp(min=1e-4)
        new_scaling = self.scaling_inverse_activation(
            torch.stack([s_xy, s_xy, torch.full_like(s_xy, 1e-4)], dim=1)
        )
        new_rotation = _normals_to_quaternions(self.normals[selected_fid])
        selected_vertex = selected_faces[:, 0]
        new_features_dc = self._features_dc[selected_vertex]
        new_features_rest = self._features_rest[selected_vertex]
        # Do not inherit geo Gaussian opacity — geo opacities are reset to ~0.01 by
        # Stage 2 reset_opacity calls, which would leave new app Gaussians invisible.
        # Initialize at 0.5 (logit = 0.0) so they contribute to the render immediately.
        new_opacity = torch.zeros(
            (selected_fid.shape[0], 1), dtype=torch.float, device="cuda"
        )  # sigmoid(0.0) = 0.5

        vertex_num = selected_vertex.shape[0]
            
        # Split from appearance gaussians
        bc = self._bc[selected_added_mask].unsqueeze(1).repeat(1, N, 1).view(-1, 3)
        new_added_bc = torch.ones_like(bc) / 3
        distance = self._distance[selected_added_mask].unsqueeze(1).repeat(1, N, 1).view(-1, 1)
        new_added_distance = torch.full_like(distance, _DIST_INIT_LOGIT)
        new_added_scaling = self.scaling_inverse_activation(
            (self.get_app_scaling[selected_added_mask].repeat(N, 1) / (0.8 * N)).clamp(min=1e-4)
        )
        # Use face-aligned rotations so split children also lie flat on the mesh.
        parent_face_ids = self.fid[selected_added_mask, 0]
        new_added_rotation = _normals_to_quaternions(self.normals[parent_face_ids]).repeat(N, 1)
        new_added_features_dc = self._added_features_dc[selected_added_mask].repeat(N,1,1)
        new_added_features_rest = self._added_features_rest[selected_added_mask].repeat(N,1,1)
        new_added_opacity = self._added_opacity[selected_added_mask].repeat(N,1)
        new_added_fid = self.fid[selected_added_mask].repeat(N,1)

        self.fid = torch.cat((self.fid, new_fid), dim=0)
        self.densification_postfix_sg3(new_bc, new_distance, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        self.fid = torch.cat((self.fid, new_added_fid), dim=0)
        self.densification_postfix_sg3(new_added_bc, new_added_distance, new_added_features_dc, new_added_features_rest, new_added_opacity, new_added_scaling, new_added_rotation)
    
        prune_filter = torch.cat((selected_added_mask, torch.zeros(vertex_num, device="cuda", dtype=bool), torch.zeros(N * selected_added_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densification_postfix_sg3(self, new_bc, new_distance, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"bc": new_bc,
        "distance": new_distance,
        "added_f_dc": new_features_dc,
        "added_f_rest": new_features_rest,
        "added_opacity": new_opacities,
        "added_scaling" : new_scaling,
        "added_rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._bc = optimizable_tensors["bc"]
        self._distance = optimizable_tensors["distance"]
        self._added_features_dc = optimizable_tensors["added_f_dc"]
        self._added_features_rest = optimizable_tensors["added_f_rest"]
        self._added_opacity = optimizable_tensors["added_opacity"]
        self._added_scaling = optimizable_tensors["added_scaling"]
        self._added_rotation = optimizable_tensors["added_rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def fill_uncovered_faces(self, coverage_threshold=0.5, max_new=500):
        """
        Add one app Gaussian at the centroid of each face whose opacity-weighted
        in-plane coverage falls below coverage_threshold.

        coverage(face) = Σ_i  opacity_i × min(s_xi × s_yi / face_area, 1.0)

        At most max_new faces are filled per call; priority goes to the faces
        with the lowest coverage first.

        Returns the number of new Gaussians added.
        """
        if self.fid is None or self._added_opacity is None:
            return 0

        device = self._added_opacity.device
        n_faces = self.face_idx.shape[0]
        face_ids = self.fid[:, 0]  # (N_app,)

        with torch.no_grad():
            # ── face areas ────────────────────────────────────────────────────
            v0 = self.vertices[self.face_idx[:, 0]]
            v1 = self.vertices[self.face_idx[:, 1]]
            v2 = self.vertices[self.face_idx[:, 2]]
            face_area = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1)  # (n_faces,)

            # ── per-Gaussian contribution ──────────────────────────────────────
            app_opac = torch.sigmoid(self._added_opacity).squeeze(1)          # (N_app,)
            app_s    = self.get_app_scaling                                    # (N_app, 3)
            inplane  = app_s[:, 0] * app_s[:, 1]                              # (N_app,)
            contrib  = app_opac * (inplane / face_area[face_ids].clamp(min=1e-8)).clamp(max=1.0)

            # ── aggregate to faces ─────────────────────────────────────────────
            face_coverage = torch.zeros(n_faces, device=device)
            face_coverage.scatter_add_(0, face_ids, contrib)

            # ── select faces to fill ───────────────────────────────────────────
            needs_fill   = face_coverage < coverage_threshold
            fill_face_ids = torch.nonzero(needs_fill, as_tuple=True)[0]  # (M,)

            if fill_face_ids.shape[0] == 0:
                return 0

            # Prioritise lowest-coverage faces
            if fill_face_ids.shape[0] > max_new:
                scores   = face_coverage[fill_face_ids]
                _, order = torch.sort(scores)
                fill_face_ids = fill_face_ids[order[:max_new]]

            n_new = fill_face_ids.shape[0]

            # ── build new Gaussian parameters ──────────────────────────────────
            fill_faces   = self.face_idx[fill_face_ids]                        # (n_new, 3)

            new_bc       = torch.full((n_new, 3), 1/3, dtype=torch.float, device=device)
            new_distance = torch.full((n_new, 1), -6.9078, dtype=torch.float, device=device)
            new_fid      = fill_face_ids.unsqueeze(1).long()

            # Scale from triangle geometry: elliptical splat aligned to longest edge
            v0f = self.vertices[fill_faces[:, 0]]   # (n_new, 3)
            v1f = self.vertices[fill_faces[:, 1]]
            v2f = self.vertices[fill_faces[:, 2]]
            e01 = (v1f - v0f).norm(dim=1)
            e02 = (v2f - v0f).norm(dim=1)
            e12 = (v2f - v1f).norm(dim=1)
            longest_edge = torch.stack([e01, e02, e12], dim=1).max(dim=1).values  # (n_new,)
            fill_area    = face_area[fill_face_ids]  # (n_new,)
            s_major = (longest_edge * 0.5).clamp(min=1e-4)
            s_minor = (fill_area / longest_edge.clamp(min=1e-8)).clamp(min=1e-4)
            new_scaling = self.scaling_inverse_activation(
                torch.stack([s_major, s_minor, torch.full_like(s_major, 1e-4)], dim=1)
            )

            new_rotation = _normals_to_quaternions(self.normals[fill_face_ids])

            # Colour from first vertex of the face (geo Gaussian)
            first_vertex    = fill_faces[:, 0]
            new_features_dc   = self._features_dc[first_vertex].clone()
            new_features_rest = self._features_rest[first_vertex].clone()

            # Opacity: sigmoid(0.0) = 0.5
            new_opacity = torch.zeros((n_new, 1), dtype=torch.float, device=device)

        # ── register with optimizer ────────────────────────────────────────────
        self.fid = torch.cat((self.fid, new_fid), dim=0)
        self.densification_postfix_sg3(
            new_bc, new_distance,
            new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation
        )

        return n_new

    def weight_control_optimizer(self, geo_grad_scale=0.1):
        ''' Control Geometry Gaussian Gradient '''
        geo_name = ["f_dc", "f_rest", "opacity", "scaling", "rotation"]

        with torch.no_grad():
            for param_group in self.optimizer.param_groups:
                name = param_group.get("name", "")

                if name in geo_name:
                    for p in param_group["params"]:
                        if p.grad is not None:
                            p.grad.mul_(geo_grad_scale)