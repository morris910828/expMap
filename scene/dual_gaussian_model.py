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

class DualGaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            #symm = strip_symmetric(actual_covariance)
            return actual_covariance
        
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

        self.mesh_path = mesh_path
        self.normals = torch.empty(0) # (F, 3)
        self.vertices = torch.empty(0) # (F, 3)
        self.face_idx = torch.empty(0) # (F, 3)
        self.vertex_radius = torch.empty(0)
        self.gaussian_deform_cov = None

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
        return self._distance

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
            offset = self._distance * n
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
            offset = self._distance * n
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
    
    def get_deform_covariance(self):
        return strip_symmetric(self.gaussian_deform_cov)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        mesh= trimesh.load(self.mesh_path, force='mesh', process=False)

        vertices = mesh.vertices
        faces = mesh.faces

        self.vertices = torch.tensor(vertices).float().cuda()
        self.face_idx = torch.tensor(faces).long().cuda()

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
        v_idx = faces[face_idx]
        v0 = vertices[v_idx[:, 0]]
        v1 = vertices[v_idx[:, 1]]
        v2 = vertices[v_idx[:, 2]]
        n = face_normals[face_idx]

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

        return bary, t

    def find_closet_faces(self):
        device = self._xyz.device

        mesh= trimesh.load(self.mesh_path, force='mesh', process=False)
        points_np = self._added_xyz.cpu().numpy()
        
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

        self._bc = bary.float()
        self._distance = t.unsqueeze(1).float()
        self.fid = face_idx.unsqueeze(1).long()

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

        faces_np = (
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
        opacity_old_new = self.inverse_opacity_activation(
            torch.min(self.opacity_activation(self._opacity), torch.ones_like(self._opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacity_old_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

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

        num_vertices = selected_vertex_mask.shape[0]
        vertex_face_count = torch.bincount(self.face_idx.flatten(), minlength=num_vertices)

        # 取出 mask == True 的頂點的面數
        masked_face_counts = vertex_face_count[selected_vertex_mask].unsqueeze(1) 

        with torch.no_grad():
            self._scaling[selected_vertex_mask] = self.scaling_inverse_activation(
                self.get_geo_scaling[selected_vertex_mask] / (masked_face_counts * 0.8)
            )

        # Split from geometry gaussians
        selected_fid = self.split_neighbor_gaussians(selected_vertex_mask)
        selected_faces = self.face_idx[selected_fid]
        new_bc = torch.ones_like(selected_faces).float().cuda() / 3
        new_distance = torch.zeros_like(selected_fid.unsqueeze(1)).float().cuda()
        new_fid = selected_fid.unsqueeze(1).long().cuda()

        selected_vertex = selected_faces[:, 0]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_vertex])
        new_rotation = self._rotation[selected_vertex]
        new_features_dc = self._features_dc[selected_vertex]
        new_features_rest = self._features_rest[selected_vertex]
        new_opacity = self._opacity[selected_vertex]

        vertex_num = selected_vertex.shape[0]
            
        # Split from appearance gaussians
        bc = self._bc[selected_added_mask].unsqueeze(1).repeat(1, N, 1).view(-1, 3)
        new_added_bc = torch.ones_like(bc) / 3
        distance = self._distance[selected_added_mask].unsqueeze(1).repeat(1, N, 1).view(-1, 1)
        new_added_distance = torch.zeros_like(distance)
        new_added_scaling = self.scaling_inverse_activation(self.get_app_scaling[selected_added_mask].repeat(N,1) / (0.8*N))
        new_added_rotation = self._added_rotation[selected_added_mask].repeat(N,1)
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