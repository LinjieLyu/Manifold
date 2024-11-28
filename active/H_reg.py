import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import os
from gaussian_renderer import render, network_gui, modified_render
from scene import Scene
from random import randint
from utils.system_utils import mkdir_p
import torchvision
from utils.general_utils import inverse_sigmoid,get_expon_lr_func
import GPUtil
import matplotlib.pyplot as plt
from scipy.stats import qmc
from utils.loss_utils import l1_loss, ssim
from scene.cameras import DummyCamera
from utils.general_utils import build_rotation,look_at
from utils.graphics_utils import geom_transform_points
import math
from einops import reduce, repeat, rearrange


def linear_interp(u, v, val, map):
    h, w = map.shape[1], map.shape[2]
    x0=int(u*(w))
    y0=int(v*(h))
    map[:,y0,x0]=val


def uv2car(u, v):
    theta = v * np.pi
    phi = u * 2 * np.pi
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return torch.tensor([x, y, z], dtype=torch.float32, device='cuda')

def car2uv(xyz):
    xyz = xyz.reshape(-1)
    x, y, z = xyz[0], xyz[1], xyz[2]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)
    u = phi / (2 * np.pi)
    v = theta / (np.pi)
    return u, v

def sobol_generator(spp,d):
    sampler = qmc.Sobol(d=d,bits=64)
    # m=int(np.ceil(np.log2(spp)))
    sample = sampler.random(spp)
    return torch.tensor(sample,device="cuda",dtype=torch.float32)

def halton_generator(spp,d):
    sampler = qmc.Halton(d=d, seed=6174)
    # m=int(np.ceil(np.log2(spp)))
    sample = sampler.random(spp)
    return torch.tensor(sample,device="cuda",dtype=torch.float32)

class HRegSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed
        self.reg_lambda = args.reg_lambda
        self.I_test: bool = args.I_test
        self.I_acq_reg: bool = args.I_acq_reg

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in args.filter_out_grad]

    
    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, exit_func,use_test=False,with_depth=False) -> List[int]:
        candidate_views = list(deepcopy(scene.get_candidate_set()))

        viewpoint_cams = scene.getTrainCameras().copy()

        if self.I_test == True:
            viewpoint_cams = scene.getTestCameras()

        params = gaussians.capture()[1:7]
        params = [p for i, p in enumerate(params) if i not in self.filter_out_idx]

        # off load to cpu to avoid oom with greedy algo
        device = params[0].device if num_views == 1 else "cpu"
        # device = "cpu" # we have to load to cpu because of inflation

        H_train = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype)
        if use_test:
            candidate_cameras = scene.getTestCameras().copy()
            candidate_views = list(np.arange(len(candidate_cameras)))
        else:
            candidate_cameras = scene.getCandidateCameras()
        # Run heesian on training set
        for cam in tqdm(viewpoint_cams, desc="Calculating diagonal Hessian on training views",disable=True):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])
            # for i, p in enumerate(params):
            #     print(i,p.grad.max(),p.grad.min())
            H_train += cur_H

            gaussians.optimizer.zero_grad(set_to_none = True) 

        H_train = H_train.to(device)
        I_train = torch.reciprocal(H_train + self.reg_lambda)

        ################### render uncertainty map ###################
        seq_path = os.path.join(scene.model_path, "seq")
        mkdir_p(seq_path)
        test_path= os.path.join(scene.model_path, "Hessian")
        mkdir_p(test_path)
        view = scene.getTestCameras().copy()[0]

        # uncertainty_map = torch.zeros((3, 64, 128)).cuda()
        # cam_indicator = torch.zeros((3, 64, 128)).cuda()
        # object_center = torch.tensor([0., 0., 0.]).cuda()
        # # object_center=means[0].mean(0)
        # radius = (view.camera_center - object_center).norm().item()
        # print(view.R, view.T)
        #
        # for h in range(64):
        #     for w in range(128):
        #         camera_center = uv2car(w / 128, (h + 1) / 65) * radius + object_center
        #         R, T = look_at(camera_center, object_center)
        #         if h == 0 and w == 0:
        #             print(R, T)
        #         view = DummyCamera(R, T, view)
        #         render_pkg = modified_render(view, gaussians, pipe, background)
        #         pred_img = render_pkg["render"]
        #         pred_img.backward(gradient=torch.ones_like(pred_img))
        #
        #         cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])
        #         if self.I_acq_reg:
        #             acq_score = torch.sum((cur_H + self.I_acq_reg) * I_train).item()
        #         else:
        #             acq_score = torch.sum(cur_H * I_train).item()
        #         u,v=(w/128,(h+1)/(64+1))
        #         linear_interp(u, v, acq_score, uncertainty_map)
        #         gaussians.optimizer.zero_grad(set_to_none=True)
        #
        # torch.save(uncertainty_map, os.path.join(seq_path, 'uncertainty_map.pt'))
        # print(uncertainty_map.max(),uncertainty_map.min())
        # uncertainty_map/= uncertainty_map.max()
        # # for idx, view in enumerate(tqdm(scene.getTrainCameras().copy(),disable=True)):
        # #     u, v = car2uv(view.camera_center)
        # #     linear_interp(u, v, torch.tensor([1, 0, 0]).cuda(), uncertainty_map)
        #
        # torchvision.utils.save_image(uncertainty_map, os.path.join(seq_path, 'uncertainty_map.png'))

        ################### testing ###################
        # if num_views == 1:
        #     return self.select_single_view(H_train, candidate_cameras, candidate_views, gaussians, pipe, background, params, exit_func)

        H_candidates = []
        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating diagonal Hessian on candidate views",disable=True)):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])
            if with_depth:
                with torch.no_grad():
                    H_per_gaussian = sum(reduce(p.grad.detach(), "n ... -> n", "sum") for p in params)
                    hessian_color = repeat(H_per_gaussian.detach(), "n -> n c", c=3)

                    # compute depth of gaussian in current view
                    to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)], dim=-1)
                    pts3d_homo = to_homo(params[0])
                    pts3d_cam = pts3d_homo @ view.world_view_transform
                    gaussian_depths = pts3d_cam[:, 2, None]

                    hessian_color = hessian_color * gaussian_depths

                    render_pkg = modified_render(view, gaussians, pipe, background, override_color=hessian_color)

                    uncertanity_map = reduce(render_pkg["render"], "c h w -> h w", "mean").detach()
                    depth = render_pkg["depth"].detach()
                    torchvision.utils.save_image(depth / depth.max(), os.path.join(test_path, '{0:03d}_depth'.format(candidate_views[idx]) + ".png"))
                    torchvision.utils.save_image(uncertanity_map, os.path.join(test_path, '{0:03d}_uncertainty'.format(candidate_views[idx]) + ".png"))
                    np.savez(os.path.join(test_path, f"uncertainty_{idx:03d}.npz"),
                             uncertanity_map=uncertanity_map.cpu(),
                             depth=depth.cpu(),
                             )


            H_candidates.append(cur_H.to(device))

            gaussians.optimizer.zero_grad(set_to_none = True) 
        
        selected_idxs = []

        for _ in range(num_views):

            if self.I_acq_reg:
                acq_scores = np.array([torch.sum((cur_H + self.I_acq_reg) * I_train).item() for cur_H in H_candidates])
            else:
                acq_scores = np.array([torch.sum(cur_H * I_train).item() for cur_H in H_candidates])
            selected_idx = acq_scores.argmax()
            selected_idxs.append(candidate_views.pop(selected_idx))

            H_train += H_candidates.pop(selected_idx)

        return selected_idxs

    
    def forward(self, x):
        return x
    
    
    def select_single_view(self, I_train, candidate_cameras, candidate_views, gaussians, pipe, background, params, exit_func, num_views=1):
        """
        A memory effcient way when doing single view selection
        """
        I_train = torch.reciprocal(I_train + self.reg_lambda)
        acq_scores = torch.zeros(len(candidate_cameras))

        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating diagonal Hessian on candidate views",disable=True)):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

            I_acq = cur_H

            if self.I_acq_reg:
                I_acq += self.reg_lambda

            gaussians.optimizer.zero_grad(set_to_none = True) 
            acq_scores[idx] += torch.sum(I_acq * I_train).item()
        
        print(f"acq_scores: {acq_scores.tolist()}")
        if self.I_test == True:
            acq_scores *= -1

        _, indices = torch.sort(acq_scores, descending=True)
        selected_idxs = [candidate_views[i] for i in indices[:num_views].tolist()]
        print(f"acq_scores_max: {[acq_scores[i] for i in indices[:num_views+3].tolist()]}, selected_idxs: {[candidate_views[i] for i in indices[:num_views+3].tolist()]}")
        return selected_idxs


class VIMCSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed
        self.with_LA= args.with_LA
        self.bg_weight = args.bg_weight


    def nbvs(self, gaussians, scene: Scene, num_views, pipeline, background, exit_func) -> List[int]:


        return None



    def nbvs_testing(self, gaussians, scene: Scene, num_views, pipeline, background, exit_func,volume_scale=1.0,use_test=False,with_depth=False) -> List[int]:
        torch.cuda.empty_cache()
        # GPUtil.showUtilization()
        render_path = os.path.join(scene.model_path, "optimization")
        test_path = os.path.join(scene.model_path, "test_variance")
        uncertainty_path = os.path.join(scene.model_path, 'uncertainty_map')

        mkdir_p(render_path)
        mkdir_p(test_path)
        mkdir_p(uncertainty_path)

        ######## camera views ########
        if use_test:
            candidate_cameras = scene.getTestCameras().copy()
            candidate_views = list(np.arange(len(candidate_cameras)))

        else:
            candidate_views = list(deepcopy(scene.get_candidate_set()))
            candidate_cameras = scene.getCandidateCameras()

        train_views = scene.getTrainCameras().copy()




        ######## testing ########
        with torch.no_grad():
            means = list(gaussians.capture()[1:7])
            scale_low = gaussians.capture_uncertainty()
            acq_scores = torch.zeros(len(candidate_cameras))

            bg_color = self.bg_weight*(1+15)/(len(train_views)+15) *torch.abs(scale_low[1]).mean() * torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")


            ch_cut = [0, 3, 6, 51, 54, 58, 59]
            rank = gaussians.rank
            spp = 2
            grid_res =2
            print(spp,grid_res)
            sampler = qmc.Sobol(d=rank, seed=6174)


            sample = sampler.random(spp)
            weights_all = torch.tensor(sample, device="cuda", dtype=torch.float32)*2-1
            weights_all*=volume_scale





            g_sampler = qmc.Sobol(d=rank, seed=0)
            num_gaussians = gaussians.get_xyz.shape[0]
            g_sample = g_sampler.random(59*num_gaussians) * grid_res
            g_weight = torch.tensor(g_sample, device="cuda", dtype=torch.float32).view(num_gaussians,59,rank).int().float() / (0.5 * grid_res - 0.5) - 1
            gs_weight = [g_weight[:,ch_cut[p_idx]:ch_cut[p_idx + 1], :].view(means[p_idx].shape + (rank,))   for p_idx in range(len(means))] #if p_idx in [0,3,4] else 1.0

            for idx, view in enumerate(tqdm(candidate_cameras, disable=True)):
                outs = []
                depthes = []
                gaussians.set_params(means)
                out_clean = render(view, gaussians, pipeline, bg_color=bg_color)["render"]



                for j in range(spp):


                    weights = [weights_all[j].view((1,) * len(means[p_idx].shape) + (rank,))* gs_weight[p_idx] for p_idx in range(6)]#


                    if rank > 0:
                        if idx==0:
                            print('test')
                        perturbed_means = \
                            [1 * (weights[p_idx] * scale_low[p_idx][..., :rank]).sum(-1) + means[p_idx] for p_idx in range(len(means))]
                    else:
                        weights1 = [torch.rand(means[p_idx].shape, device="cuda", dtype=torch.float32) * 2 - 1 for p_idx in range(6)]

                        perturbed_means = \
                            [weights1[p_idx] * scale_low[p_idx][..., -1] + means[p_idx] for p_idx in range(len(means))]  #


                    gaussians.set_params(perturbed_means)
                    out = render(view, gaussians, pipeline, bg_color=bg_color*float(j%spp)/(spp-1))["render"]
                    if with_depth:
                        depth=modified_render(view, gaussians, pipeline, bg_color=bg_color*float(j%spp)/(spp-1))["depth"]
                        depthes.append(depth)
                    outs.append(out.detach())

                out_mean = torch.stack(outs).mean(0)
                out_variance = torch.stack(outs).var(0)
                if with_depth:
                    depth_mean = torch.stack(depthes).mean(0)
                    torchvision.utils.save_image(depth_mean/depth_mean.max(), os.path.join(test_path, '{0:03d}_depth_mean'.format(candidate_views[idx]) + ".png"))
                    np.savez(os.path.join(test_path, f"uncertainty_{idx:03d}.npz"),
                             uncertanity_map=out_variance.cpu(),
                             depth=depth_mean.cpu(),
                             )
                acq_scores[idx] = out_variance.sum().item()

                torchvision.utils.save_image(out_clean, os.path.join(test_path, '{0:03d}_clean'.format(candidate_views[idx]) + ".png"))
                torchvision.utils.save_image(out_mean, os.path.join(test_path, '{0:03d}_mean'.format(candidate_views[idx]) + ".png"))
                torchvision.utils.save_image(out_variance / out_variance.max(), os.path.join(test_path, '{0:03d}_variance'.format(candidate_views[idx]) + ".png"))

            ######## uncertainty map ########
        #     view = scene.getTestCameras().copy()[0]
        #     H=64
        #     W=H*2
        #     uncertainty_map=torch.zeros((3,H,W)).cuda()
        #     cam_indicator=torch.zeros((3,H,W)).cuda()
        #     object_center=torch.tensor([0.,0.,0.]).cuda()
        #     # object_center=means[0].mean(0)
        #     radius = (view.camera_center-object_center).norm().item()
        #
        #
        #     for h in range(H):
        #         for w in range(W):
        #             camera_center = uv2car(w/W,(h+1)/(H+1))*radius+object_center
        #             R,T=look_at(camera_center,object_center)
        #             view=DummyCamera(R, T, view)
        #             outs = []
        #
        #             for j in range(spp):
        #                 weights1 = [torch.rand(means[p_idx].shape, device="cuda", dtype=torch.float32) * 2 - 1 for p_idx in range(6)]
        #                 weights = [weights_all[j].view((1,) * len(means[p_idx].shape) + (rank,)) * gs_weight[p_idx] for p_idx in range(6)]  #
        #
        #
        #                 if rank > 0:
        #                     if idx == 0:
        #                         print('test')
        #                     perturbed_means = \
        #                         [1 * (weights[p_idx] * scale_low[p_idx][..., :rank]).sum(-1) + means[p_idx] for p_idx in range(len(means))]  # +weights1[p_idx] * scale_low[p_idx][...,-1]
        #                 else:
        #                     perturbed_means = \
        #                         [weights1[p_idx] * scale_low[p_idx][..., -1] + means[p_idx] for p_idx in range(len(means))]  #
        #
        #                 gaussians.set_params(perturbed_means)
        #                 out = render(view, gaussians, pipeline, bg_color=bg_color * float(j % spp) / (spp - 1))["render"]
        #                 outs.append(out.detach())
        #
        #             out_variance = torch.stack(outs).var(0)
        #             out_mean = torch.stack(outs).mean(0)
        #
        #             # torchvision.utils.save_image(out_mean, os.path.join(uncertainty_path, 'test_{}_{}'.format(h,w) + "_mean.png"))
        #             # torchvision.utils.save_image(out_variance , os.path.join(uncertainty_path, 'test_{}_{}'.format(h,w) + "_variance.png"))
        #
        #             u,v=(w/W,(h+1)/(H+1))
        #             linear_interp(u,v,out_variance.sum(),uncertainty_map)
        #             linear_interp(u, v, torch.tensor([0,1,0],device='cuda'), cam_indicator)
        #     uncertainty_map/=uncertainty_map.max()
        #
        #
            for train_id, train_view in enumerate(tqdm(train_views,disable=True)):
                # u,v=car2uv(train_view.camera_center-object_center)
                # linear_interp(u, v, torch.tensor([1,0,0],device='cuda'), cam_indicator)

                outs = []
                depthes = []
                gaussians.set_params(means)
                out_clean = render(train_view, gaussians, pipeline, bg_color=bg_color)["render"]

                for j in range(spp):

                    weights = [weights_all[j].view((1,) * len(means[p_idx].shape) + (rank,))* gs_weight[p_idx] for p_idx in range(6)]

                    perturbed_means = \
                        [(weights[p_idx] * scale_low[p_idx][...,:rank]).sum(-1) + means[p_idx] for p_idx in range(len(means))]

                    gaussians.set_params(perturbed_means)
                    out = render(train_view, gaussians, pipeline, bg_color=bg_color*float(j%spp)/(spp-1))["render"]
                    outs.append(out.detach())
                    if with_depth:
                        depth = modified_render(train_view, gaussians, pipeline, bg_color=bg_color * float(j % spp) / (spp - 1))["depth"]
                        depthes.append(depth)
                out_mean = torch.stack(outs).mean(0)
                out_variance = torch.stack(outs).var(0)
                if with_depth:
                    depth_mean = torch.stack(depthes).mean(0)
                    torchvision.utils.save_image(depth_mean / depth_mean.max(), os.path.join(test_path, 'train_{0:03d}_depth_mean'.format(train_id) + ".png"))

                torchvision.utils.save_image(out_clean, os.path.join(test_path, 'train_{0:03d}_clean'.format(train_id) + ".png"))
                torchvision.utils.save_image(out_mean, os.path.join(test_path, 'train_{0:03d}_mean'.format(train_id) + ".png"))
                torchvision.utils.save_image(out_variance / out_variance.max(), os.path.join(test_path, 'train_{0:03d}_variance'.format(train_id) + ".png"))

        gaussians.set_params(means)
        #
        # torchvision.utils.save_image(uncertainty_map, os.path.join(uncertainty_path, 'uncertainty_map' + ".png"))
        # torchvision.utils.save_image(cam_indicator, os.path.join(uncertainty_path, 'cam_indicator' + ".png"))

        _, indices = torch.sort(acq_scores, descending=True)
        selected_idxs = [candidate_views[i] for i in indices[:num_views].tolist()]
        print(f"acq_scores_max: {[acq_scores[i] for i in indices[:num_views + 3].tolist()]}, selected_idxs: {[candidate_views[i] for i in indices[:num_views + 3].tolist()]}")

        return selected_idxs

    def forward(self, x):
        return x


class KSMetricSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed

    def nbvs(self, gaussian, scene, num_views, *args, **kwargs) -> List[int]:
        with torch.no_grad():
            means = list(gaussian.capture()[1:7])
            xyz=means[0].detach()
            xyz_bounds = torch.tensor([xyz[:,0].max(), xyz[:,1].max(), xyz[:,2].max(), xyz[:,0].min(), xyz[:,1].min(), xyz[:,2].min()]).cuda()
            resolution = 32
            new_xyz_grid = torch.linspace(0, 1, resolution).cuda()
            new_xyz = torch.stack(torch.meshgrid(new_xyz_grid, new_xyz_grid, new_xyz_grid), -1).view(-1, 3) * (xyz_bounds[3:] - xyz_bounds[:3]) + xyz_bounds[:3] # 32^3 x 3

            def per_pixel_area_of_sphere(H, W):
                D_theta = math.pi / H
                D_phi = 2 * math.pi / W
                perpixel_area = torch.sin(((torch.arange(H) + 0.5) / H) * math.pi)[:, None].repeat(1, W) * D_theta * D_phi

                return perpixel_area



            def point_cam_mask(cam, xyz):
                projection_matrix = cam.projection_matrix
                camera_space_points  = geom_transform_points(xyz, projection_matrix)
                camera_space_points[:,:2]/=camera_space_points[:,2:3]
                positive_z_filter = camera_space_points[:, 2:3] > 0
                incamera_filter = (camera_space_points[:, 0:1] > -cam.image_width / 2.0) & (camera_space_points[:, 0:1] < cam.image_width / 2.0) & \
                                  (camera_space_points[ :, 1:2] > -cam.image_height / 2.0) & (camera_space_points[ :, 1:2] < cam.image_height / 2.0)

                return torch.logical_and(positive_z_filter, incamera_filter)

            def getDirsOnPolar(dirs, H, W):
                def cartesian_to_polar(xyz):
                    ptsnew = torch.cat((xyz, torch.zeros(xyz.shape, device="cuda")), dim=-1)
                    xy = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
                    ptsnew[..., 3] = torch.sqrt(xy + xyz[..., 2] ** 2)
                    ptsnew[..., 4] = torch.arctan2(torch.sqrt(xy), xyz[..., 2])  # for elevation angle defined from Z-axis down
                    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
                    ptsnew[..., 5] = torch.arctan2(xyz[..., 1], xyz[..., 0])
                    return ptsnew

                # r, theta, phi

                sph_coord = cartesian_to_polar(dirs)[..., 3:].float()
                sph_coord[..., 1] = sph_coord[..., 1] / math.pi  # from 0 - pi to 0 - 1
                sph_coord[..., 2] = ((sph_coord[..., 2] + math.pi) / (2 * math.pi))  # from -pi - pi to 0 - 1

                # 0 - 1 to 0 - H/W
                y = sph_coord[..., 1] * H
                x = sph_coord[..., 2] * W

                return x, y


            candidate_views = list(deepcopy(scene.get_candidate_set()))
            candidate_cameras = scene.getCandidateCameras().copy()
            train_views = scene.getTrainCameras().copy()
            acq_scores = torch.zeros(len(candidate_cameras))

            cdf_resolution = 16
            sphere_areas = per_pixel_area_of_sphere(cdf_resolution, cdf_resolution * 2).unsqueeze(0).cuda()
            pdf_u = torch.ones((cdf_resolution, 2 * cdf_resolution), device="cuda") / (cdf_resolution * 2 * cdf_resolution)
            hist_gk = torch.zeros((new_xyz.shape[0],cdf_resolution, 2 * cdf_resolution), device="cuda")
            point_cam_masks = []


            for train_view in train_views:
                point_cam_mask_train = point_cam_mask(train_view, new_xyz).float()  # 32^3 * 1
                point_cam_masks.append(point_cam_mask_train)

                NC_vector = new_xyz - train_view.camera_center.reshape(1, 3)
                NC_vector /= torch.norm(NC_vector, dim=-1, keepdim=True)
                x, y = getDirsOnPolar(NC_vector, cdf_resolution, 2*cdf_resolution) # 32^3, 32^3
                x,y=torch.floor(x).long(),torch.floor(y).long()
                hist_gk[:, y, x] += point_cam_mask_train.float()


            point_cam_masks = torch.cat(point_cam_masks, dim=-1).float()


            for i,cand_view in enumerate(candidate_cameras):

                hist_gk_new = hist_gk+torch.zeros_like(hist_gk)
                point_cam_masks_cand = point_cam_mask(cand_view, new_xyz).float()
                point_cam_masks_new=torch.cat([point_cam_masks,point_cam_masks_cand],dim=-1)


                NC_vector = (new_xyz - cand_view.camera_center.reshape(1, 3))
                NC_vector/=torch.norm(NC_vector,dim=-1,keepdim=True)
                x, y = getDirsOnPolar(NC_vector, cdf_resolution, 2 * cdf_resolution)  # 32^3, 32^3
                x,y=torch.floor(x).long(),torch.floor(y).long()

                hist_gk_new[:, y, x] += point_cam_masks_cand.float()

                hist_gk_area_weighted = hist_gk_new / sphere_areas


                pdf_gk = hist_gk_area_weighted/hist_gk_area_weighted.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).clamp_min(0.00001)

                angular_distance = 1 - torch.abs(pdf_gk - pdf_u[None]).sum(dim=-1).sum(dim=-1) / 2.0
                not_observed = pdf_gk.view(pdf_gk.shape[0], -1).sum(1) == 0.0

                angular_distance[not_observed] = 0.0

                spatial_distance = point_cam_masks_new.mean(dim=1)
                # print('123',spatial_distance.shape, spatial_distance.max())
                acq_scores[i] = (angular_distance + torch.pow(spatial_distance, 0.1/2.0)).sum().item()

            _, indices = torch.sort(acq_scores, descending=True)
            selected_idxs = [candidate_views[i] for i in indices[:num_views].tolist()]
            print(f"acq_scores_max: {[acq_scores[i] for i in indices[:num_views + 3].tolist()]}, selected_idxs: {[candidate_views[i] for i in indices[:num_views + 3].tolist()]}")

            seq_path = os.path.join(scene.model_path, "seq")
            mkdir_p(seq_path)
            view = scene.getTestCameras().copy()[0]

            # uncertainty_map = torch.zeros((3, 64, 128)).cuda()
            # cam_indicator = torch.zeros((3, 64, 128)).cuda()
            # object_center = torch.tensor([0., 0., 0.]).cuda()
            # # object_center=means[0].mean(0)
            # radius = (view.camera_center - object_center).norm().item()
            #
            # for h in range(64):
            #     for w in range(128):
            #         camera_center = uv2car(w / 128, (h + 1) / 65) * radius + object_center
            #         R, T = look_at(camera_center, object_center)
            #         view = DummyCamera(R, T, view)
            #         hist_gk_new = hist_gk + torch.zeros_like(hist_gk)
            #         point_cam_masks_cand = point_cam_mask(view, new_xyz).float()
            #         point_cam_masks_new = torch.cat([point_cam_masks, point_cam_masks_cand], dim=-1)
            #
            #         NC_vector = (new_xyz - view.camera_center.reshape(1, 3))
            #         NC_vector /= torch.norm(NC_vector, dim=-1, keepdim=True)
            #         x, y = getDirsOnPolar(NC_vector, cdf_resolution, 2 * cdf_resolution)  # 32^3, 32^3
            #         x, y = torch.floor(x).long(), torch.floor(y).long()
            #
            #         hist_gk_new[:, y, x] += point_cam_masks_cand.float()
            #
            #         hist_gk_area_weighted = hist_gk_new / sphere_areas
            #
            #         pdf_gk = hist_gk_area_weighted / hist_gk_area_weighted.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).clamp_min(0.00001)
            #
            #         angular_distance = 1 - torch.abs(pdf_gk - pdf_u[None]).sum(dim=-1).sum(dim=-1) / 2.0
            #         not_observed = pdf_gk.view(pdf_gk.shape[0], -1).sum(1) == 0.0
            #
            #         angular_distance[not_observed] = 0.0
            #
            #         spatial_distance = point_cam_masks_new.mean(dim=1)
            #         acq_score = (angular_distance + torch.pow(spatial_distance, 0.1 / 2.0)).sum().item()
            #         u,v=(w/128,(h+1)/(64+1))
            #         linear_interp(u, v, acq_score, uncertainty_map)
            #
            # uncertainty_map /= uncertainty_map.max()
            # # for idx, view in enumerate(tqdm(scene.getTrainCameras().copy(),disable=True)):
            # #     u, v = car2uv(view.camera_center)
            # #     linear_interp(u, v, torch.tensor([1, 0, 0]).cuda(), uncertainty_map)
            #
            # torchvision.utils.save_image(uncertainty_map, os.path.join(seq_path, 'uncertainty_map.png'))
            return selected_idxs

    def forward(self, x):
        return x
