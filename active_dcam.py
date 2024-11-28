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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from active import methods_dict
import torchvision
from utils.system_utils import mkdir_p
from lpipsPyTorch import lpips, lpips_func
from scipy.stats import qmc
from utils.general_utils import  build_rotation,matrix_to_quaternion,transform_gaussians,transform_gaussians2,car2uv,linear_interp,uv2car_torch,look_at_torch,uv2car,look_at
from scene.cameras import DummyCamera
import numpy as np
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.cluster_manager import ClusterStateManager

csm = ClusterStateManager()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,args,ref_view_stack=[]):
    total_views=args.num_views
    active_method = methods_dict[args.method](args)

    rank = args.rank
    random_init_pcd = args.random_init_pcd

    gaussians = GaussianModel(dataset.sh_degree, rank)
    tb_writer = prepare_output_and_logger(dataset)
    scene = Scene(dataset, gaussians, random_init=random_init_pcd)
    model_path = scene.model_path
    gaussians.training_setup(opt, with_uncertainty=args.with_uncertainty)
    init_model_params = gaussians.capture()
    scene.train_idxs = [0]
    train_viewpoint_stack = scene.getTrainCameras().copy()

    if len(ref_view_stack)>0:
        train_viewpoint_stack=ref_view_stack.copy()
    ref_gaussians = GaussianModel(dataset.sh_degree, rank)
    ref_gaussians.load_ply(os.path.join(args.source_path,"point_cloud", "iteration_" + str(30000),"point_cloud.ply"))
    ref_gaussians.active_sh_degree = 3
    ref_gaussians.training_setup(opt, with_uncertainty=False)
    while len(train_viewpoint_stack) < total_views:
        cur_views = len(train_viewpoint_stack)
        print('Training with {}/{} views'.format(cur_views, total_views))
        first_iter = 0


        # Load the cameras

        scene.model_path = os.path.join(model_path, str(cur_views))
        optim_path = os.path.join(scene.model_path, "optimization")
        test_path = os.path.join(scene.model_path, "test_variance")
        mkdir_p(optim_path)
        mkdir_p(test_path)

        gaussians.restore(init_model_params, opt)

        if not gaussians.with_uncertainty:
            ply_path = os.path.join(scene.model_path, "point_cloud", "iteration_" + str(7000), "point_cloud.ply")
            # if os.path.exists(ply_path):
            #     gaussians.load_ply(ply_path)
            #     opt.iterations = 0
            #     testing_iterations = [0]

        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)



        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")



        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1

        ############################################################################################################
        if args.with_uncertainty:
            print(f'scene scale: {gaussians.spatial_lr_scale, scene.cameras_extent}')
            means = list(gaussians.capture()[1:7])
            ch_cut = [0, 3, 6, 51, 54, 58, 59]
            scale = [1, 1, 1, 1, 1, 1]
            print(scale, rank)

            rank = gaussians.rank
            spp = rank * 4
            grid_res = 2
            expand = 10
            print(spp, grid_res, expand, opt.opacity_reset_interval)
            sampler = qmc.Sobol(d=rank, seed=6174, bits=64)

            sample = sampler.random(opt.iterations + 5)
            weights_all = torch.tensor(sample, device="cuda", dtype=torch.float32) * 2 - 1




            g_sampler = qmc.Sobol(d=rank, seed=0, bits=64)
            num_gaussians = gaussians.get_xyz.shape[0]
            g_sample = g_sampler.random(59 * num_gaussians) * grid_res
            g_weight = torch.tensor(g_sample, device="cuda", dtype=torch.float32).view(num_gaussians, 59, rank).int().float() / (0.5 * grid_res - 0.5) - 1
            gs_weight = [g_weight[:, ch_cut[p_idx]:ch_cut[p_idx + 1], :].view(means[p_idx].shape + (rank,)) for p_idx in range(len(means))]  # if p_idx in [0,3,4] else 1.0


        ############################################################################################################

        for iteration in range(first_iter, opt.iterations + 1):
            # if network_gui.conn == None:
            #     network_gui.try_connect()
            # while network_gui.conn != None:
            #     try:
            #         net_image_bytes = None
            #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            #         if custom_cam != None:
            #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
            #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            #         network_gui.send(net_image_bytes, dataset.source_path)
            #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
            #             break
            #     except Exception as e:
            #         network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = train_viewpoint_stack.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            # viewpoint_cam = viewpoint_stack.pop(0)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            ############################################################################################################
            if args.with_uncertainty and iteration<=args.opacity_reset_interval:

                means = list(gaussians.capture()[1:7])
                scale_low = gaussians.capture_uncertainty()

                weights = [weights_all[iteration].view((1,) * len(means[p_idx].shape) + (rank,)) * gs_weight[p_idx] for p_idx in range(6)]  #


                if rank > 0:
                    perturbed_means = \
                        [1 * (weights[p_idx] * scale_low[p_idx][..., :rank]).sum(-1) + means[p_idx] for p_idx in range(len(means))]
                else:
                    weights1 = [torch.rand(means[p_idx].shape, device="cuda", dtype=torch.float32) * 2 - 1 for p_idx in range(6)]

                    if iteration == 0:
                        print('rank 0')
                    perturbed_means = \
                        [weights1[p_idx] * scale_low[p_idx][..., -1] + means[p_idx] for p_idx in range(len(means))]  #
                gaussians.set_params(perturbed_means)
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                gaussians.set_params(means)

            ############################################################################################################
            else:
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            ############################################################################################################
            if args.with_uncertainty:
                scale_mean = 0
                if iteration % expand == 0 and iteration < opt.iterations - 100:
                    scale_mean = sum([torch.abs(scale_low[p_idx]).sum() for p_idx in range(len(means))])

                loss += -scale_mean/cur_views
            ############################################################################################################
            loss.backward()

            iter_end.record()

            with torch.no_grad():

                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 500 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(500)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    if not gaussians.with_uncertainty:
                        scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, size_threshold)
                        if gaussians.with_uncertainty:
                            g_sampler = qmc.Sobol(d=rank, seed=0, bits=64)
                            means = list(gaussians.capture()[1:7])
                            num_gaussians = gaussians.get_xyz.shape[0]
                            g_sample = g_sampler.random(59 * num_gaussians) * grid_res
                            g_weight = torch.tensor(g_sample, device="cuda", dtype=torch.float32).view(num_gaussians, 59, rank).int().float() / (0.5 * grid_res - 0.5) - 1
                            gs_weight = [g_weight[:, ch_cut[p_idx]:ch_cut[p_idx + 1], :].view(means[p_idx].shape + (rank,)) for p_idx in range(len(means))]  # if p_idx in [0,3,4] else 1.0


                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if len(ref_view_stack)==0:
            ############################ Active selection   ############################
            if args.method !='VI_MC':
                selected_views = active_method.nbvs(gaussians, scene, 1, pipe, background, exit_func=csm.should_exit)
            else:
                selected_views = active_method.nbvs_testing(gaussians, scene, 1, pipe, background, exit_func=csm.should_exit)
            scene.train_idxs += selected_views
            cadidate_view = scene.getTrainCameras().copy()[-1]
            # cadidate_view = scene.getTestCameras().copy()[0]
            test_cam_center = cadidate_view.camera_center.detach()
            u, v = car2uv(test_cam_center)

            # cadidate_view = scene.getCandidateCameras().copy()[0]
            object_center = torch.zeros_like(means[0].mean(0).clone().detach())
            radius = (cadidate_view.camera_center - object_center).norm().item()



            means = list(gaussians.capture()[1:7])
            scale_low = gaussians.capture_uncertainty()


            R=np.eye(3)
            t=np.zeros(3)
            dummy_cam = DummyCamera(R, t, cadidate_view)
            test_view = DummyCamera(R, t, cadidate_view)


            test_u,test_v=torch.tensor([u],device='cuda',requires_grad=True),torch.tensor([v],device='cuda',requires_grad=True)

            # if len(train_viewpoint_stack)%2==0:
            #     test_v.data= test_v.data*0 + 0.02

            test_r=torch.tensor([radius],device='cuda',requires_grad=True)
            test_r.requires_grad=True

            ##################################

            uncertainty_map = torch.zeros((3, 256, 512)).cuda()
            cam_indicator = torch.zeros((3, 256, 512)).cuda()


            optimizer = torch.optim.Adam([{'params': [test_u], 'lr': 5e-4}, #
                                          {'params': [test_v], 'lr': 5e-4}, #
                                          {'params': [test_r], 'lr': 0e-2}])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=1.0)

            spp = 2
            steps=401
            sampler = qmc.Sobol(d=rank, seed=6174)

            sample = sampler.random(spp)
            weights_all = torch.tensor(sample, device="cuda", dtype=torch.float32) * 2 - 1
            # weights_all *= 0.1

            g_sampler = qmc.Sobol(d=rank, seed=0)
            num_gaussians = gaussians.get_xyz.shape[0]
            g_sample = g_sampler.random(59 * num_gaussians) * grid_res
            g_weight = torch.tensor(g_sample, device="cuda", dtype=torch.float32).view(num_gaussians, 59, rank).int().float() / (0.5 * grid_res - 0.5) - 1
            gs_weight = [g_weight[:, ch_cut[p_idx]:ch_cut[p_idx + 1], :].view(means[p_idx].shape + (rank,)) for p_idx in range(len(means))]  # if p_idx in [0,3,4] else 1.0

            torch.autograd.set_detect_anomaly(True)

            for e in range(steps):
                optimizer.zero_grad()
                ####### camera center to R,T ########
                test_v.data = test_v.data.clamp(0.01, 0.46)
                test_u.data=torch.remainder(test_u.data,1)
                test_cam_center = uv2car_torch(test_u, test_v)
                normalized_cam_center = test_cam_center* test_r + object_center
                test_w2c=look_at_torch(normalized_cam_center,object_center)
                outs = []

                if e==200:
                    weights_all *= 0.1
                for j in range(spp):
                    weights = [weights_all[j].view((1,) * len(means[p_idx].shape) + (rank,)) * gs_weight[p_idx] for p_idx in range(6)]  #
                    # weights = [(float(j%spp)/(0.5*spp-0.5)-1) * gs_weight[p_idx] for p_idx in range(6)]

                    if rank > 0:
                        perturbed_means = \
                            [1 * (weights[p_idx] * scale_low[p_idx][..., :rank]).sum(-1) + means[p_idx] for p_idx in range(len(means))]  # +weights1[p_idx] * scale_low[p_idx][...,-1]
                    else:
                        weights1 = [torch.rand(means[p_idx].shape, device="cuda", dtype=torch.float32) * 2 - 1 for p_idx in range(6)]
                        # weights1 = [torch.rand((1,) + means[p_idx].shape[1:], device="cuda", dtype=torch.float32) * 2 - 1 for p_idx in range(6)]
                        perturbed_means = \
                            [weights1[p_idx] * scale_low[p_idx][..., -1] + means[p_idx] for p_idx in range(len(means))]  #

                    xyz, rotation = perturbed_means[0], perturbed_means[4]
                    xyz_, rotation_ = transform_gaussians2(xyz,rotation,test_w2c.transpose(0, 1))
                    perturbed_means_ = [xyz_] + perturbed_means[1:4] + [rotation_] + perturbed_means[5:6]

                    gaussians.set_params(perturbed_means_)
                    out = render(dummy_cam, gaussians, pipe, background)["render"]
                    outs.append(out)

                out_mean = torch.stack(outs).mean(0)
                out_var = torch.stack(outs).var(0)
                loss = -1*out_var.mean()



                loss.backward()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    u, v = test_u.item(), test_v.item()
                    linear_interp(u, v, torch.tensor([1,0,0],device='cuda'), cam_indicator)
                    linear_interp(u, v, out_var.sum(), uncertainty_map)

                if e%100==0:
                    print(f'Loss: {loss.item(),u, v,test_r.item()}')
                    ################# small test #################
                    with torch.no_grad():
                        ####### R,T to camera w2c ########

                        test_v.data = test_v.data.clamp(0.01, 0.46)
                        test_u.data = torch.remainder(test_u.data, 1)
                        test_cam_center = uv2car_torch(test_u, test_v)
                        normalized_cam_center = test_cam_center * test_r + object_center
                        test_w2c = look_at_torch(normalized_cam_center, object_center)

                        test_view.world_view_transform = test_w2c.detach()
                        test_view.full_proj_transform = (test_view.world_view_transform.unsqueeze(0).bmm(test_view.projection_matrix.unsqueeze(0))).squeeze(0)
                        test_view.camera_center = test_view.world_view_transform.inverse()[3, :3]


                        ref_img = render(test_view, ref_gaussians, pipe, background)["render"]
                        torchvision.utils.save_image(ref_img, os.path.join(optim_path, f"ref_{e}.png"))
                        test_view.original_image = ref_img.detach()



                        #################   #################

                        torchvision.utils.save_image(out_mean, os.path.join(optim_path, f"mean_{e}.png"))
                        torchvision.utils.save_image(out_var/out_var.max(), os.path.join(optim_path, f"var_{e}.png"))
                        torchvision.utils.save_image(cam_indicator, os.path.join(optim_path, f"cam_indicator_{e}.png"))
                        torchvision.utils.save_image(uncertainty_map/uncertainty_map.max(), os.path.join(optim_path, f"uncertainty_map_{e}.png"))


            train_viewpoint_stack.append(test_view)
            # Save training sequence
            seq_path = os.path.join(scene.model_path, "seq")
            mkdir_p(seq_path)
            camera_data = {}
            for i,view in enumerate(train_viewpoint_stack):
                print(view.camera_center)
                torchvision.utils.save_image(view.original_image , os.path.join(seq_path, f"r_{i}.png"))
                ## store camera to json
                W2C = view.world_view_transform.cpu().numpy()
                c2w = np.linalg.inv(W2C).transpose()
                c2w[:3, 1:3] *= -1
                fov = view.FoVx
                camera_data['wmat_%d' % i]=c2w
                camera_data['fov_%d' % i]=fov
            np.save(os.path.join(seq_path, 'camera_data.npy'), camera_data)
        else:
            test_path = os.path.join(scene.model_path, "test")
            viewpoint = scene.getTrainCameras().copy()[0]
            object_center = torch.tensor([0., 0., 0.]).cuda()
            radius = (viewpoint.camera_center - object_center).norm().item()
            for i in range(256):
                w = i % 64
                camera_center = uv2car(w / 64, (i + 32) / (64 + 512)) * radius + object_center
                R, T = look_at(camera_center, object_center)
                view = DummyCamera(R, T, viewpoint)

                image = torch.clamp(render(view, scene.gaussians, pipe, background)["render"], 0.0, 1.0)
                torchvision.utils.save_image(image, os.path.join(test_path, "view_{}.png".format(i)))
                ref_img = render(view, ref_gaussians, pipe, background)["render"]
                torchvision.utils.save_image(ref_img, os.path.join(test_path, "ref_view_{}.png".format(i)))
            print('No more reference cameras')
            return train_viewpoint_stack

    return train_viewpoint_stack

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    test_path = os.path.join(scene.model_path, "test")
    os.makedirs(test_path, exist_ok=True)
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        lpips = lpips_func("cuda", net_type='vgg')
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras().copy()},#+scene.getCandidateCameras().copy()
                              {'name': 'train', 'cameras': scene.getTrainCameras().copy()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    torchvision.utils.save_image(image, os.path.join(test_path, "{}_{}.png".format(config['name'], idx)))
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips.to(image.device)
                    lpips_test += lpips(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def load_train_viewpoint_stack(dataset, num_views,args):
    gaussians = GaussianModel(dataset.sh_degree, 1)
    tb_writer = prepare_output_and_logger(dataset)
    scene = Scene(dataset, gaussians)
    train_viewpoint_stack = []
    camera_dict = np.load(os.path.join(args.model_path, str(9),"seq","camera_data.npy"), allow_pickle=True).item()
    for i in range(num_views):
        viewpoint = DummyCamera(np.eye(3), np.zeros(3), scene.getTrainCameras().copy()[0])
        c2w=camera_dict['wmat_%d' % i]
        fov=camera_dict['fov_%d' % i]
        # projection_matrix=getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).transpose(0, 1).cuda()
        # print(projection_matrix,viewpoint.projection_matrix)
        c2w[:3, 1:3] *= -1
        W2C = np.linalg.inv(c2w.transpose())
        W2C=torch.from_numpy(W2C).float().cuda()
        viewpoint.world_view_transform = W2C
        viewpoint.full_proj_transform = (viewpoint.world_view_transform.unsqueeze(0).bmm(viewpoint.projection_matrix.unsqueeze(0))).squeeze(0)
        viewpoint.camera_center = viewpoint.world_view_transform.inverse()[3, :3]

        image_path = os.path.join(args.model_path, str(9),"seq",f"r_{i}.png")
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
        resized_image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0
        viewpoint.original_image = resized_image.float().cuda()
        train_viewpoint_stack.append(viewpoint)

    return train_viewpoint_stack

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--cam_seqeunce", nargs="+", type=int, default=[0])
    parser.add_argument("--cam_seqeunce_reference", nargs="+", type=int, default=[])
    parser.add_argument("--num_views", type=int, default=11)
    # Flags for view selections
    parser.add_argument("--method", type=str, default="rand")
    parser.add_argument("--schema", type=str, default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reg_lambda", type=float, default=1e-6)
    parser.add_argument("--I_test", action="store_true", help="Use I test to get the selection base")
    parser.add_argument("--I_acq_reg", action="store_true", help="apply reg_lambda to acq H too")
    parser.add_argument("--sh_up_every", type=int, default=5_000, help="increase spherical harmonics every N iterations")
    parser.add_argument("--sh_up_after", type=int, default=-1, help="start to increate active_sh_degree after N iterations")
    parser.add_argument("--min_opacity", type=float, default=0.005, help="min_opacity to prune")
    parser.add_argument("--filter_out_grad", nargs="+", type=str, default=["rotation"])  #
    parser.add_argument("--log_every_image", action="store_true", help="log every images during traing")
    parser.add_argument("--with_LA", action="store_true")
    parser.add_argument("--bg_weight",type=float, default=0.00)
    parser.add_argument("--refine_scale", action="store_true")
    parser.add_argument("--with_uncertainty", action="store_true")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--random_init_pcd", action="store_true")
    parser.add_argument("--volume_scale", type=float, default=1.0)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    print(args.with_uncertainty, args.resolution)
    train_viewpoint_stack=training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args)

    # train_viewpoint_stack = load_train_viewpoint_stack(lp.extract(args), args.num_views,args)
    # All done
    print("\nTraining complete.")

    n_seq_cameras = len(train_viewpoint_stack)
    args.num_views = n_seq_cameras + 1
    args.with_uncertainty = False
    args.resolution = -1
    args.iterations = 7000
    args.opacity_reset_interval = 3000
    args.random_init_pcd = False
    print(args.with_uncertainty, args.resolution, args.random_init_pcd)

    _ = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
                 args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args,train_viewpoint_stack)

    args.num_views = n_seq_cameras // 2 + 1
    print(args.with_uncertainty, args.resolution, args.random_init_pcd)
    _ = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
                 args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args,train_viewpoint_stack[:n_seq_cameras // 2])
