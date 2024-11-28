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
from utils.general_utils import  build_rotation,look_at,uv2car
from scene.cameras import DummyCamera
from scene.dataset_readers import sceneLoadTypeCallbacks
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.cluster_manager import ClusterStateManager

csm = ClusterStateManager()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,args):
    active_method = methods_dict[args.method](args)
    total_views=args.num_views
    seq_cameras=args.cam_seqeunce
    ref_cameras=args.cam_seqeunce_reference
    rank = args.rank
    random_init_pcd = args.random_init_pcd
    shuffle_data = args.shuffle_data

    gaussians = GaussianModel(dataset.sh_degree, rank)
    tb_writer = prepare_output_and_logger(dataset)
    scene = Scene(dataset, gaussians,random_init=random_init_pcd,shuffle_data=shuffle_data)
    model_path = scene.model_path
    gaussians.training_setup(opt, with_uncertainty=args.with_uncertainty)
    init_model_params = gaussians.capture()
    print(ref_cameras)
    while len(seq_cameras) < total_views:
        cur_views = len(seq_cameras)
        print('Training with {}/{} views'.format(cur_views, total_views))
        first_iter = 0

        # Load the cameras
        scene.train_idxs = seq_cameras
        print('selected cameras: {}'.format(scene.train_idxs))
        scene.model_path = os.path.join(model_path, str(cur_views))

        gaussians.restore(init_model_params, opt)
        # if not gaussians.with_uncertainty:
        #     ply_path = os.path.join(scene.model_path, "point_cloud","iteration_" + str(7000), "point_cloud.ply")
        #     if  os.path.exists(ply_path):
        #          gaussians.load_ply(ply_path)
        #          opt.iterations=0
        #          testing_iterations=[0]
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

            print(f'scene scale: {gaussians.spatial_lr_scale,scene.cameras_extent}')
            means = list(gaussians.capture()[1:7])
            ch_cut = [0, 3, 6, 51, 54, 58, 59]
            scale = [1, 1, 1, 1, 1, 1]
            print(scale, rank)

            rank=gaussians.rank
            spp = rank*4
            grid_res = 2
            expand = 10
            print(spp,grid_res,expand,opt.opacity_reset_interval)
            sampler = qmc.Sobol(d=rank,seed=6174,bits=64)

            sample = sampler.random(opt.iterations+5)
            weights_all = torch.tensor(sample, device="cuda", dtype=torch.float32)*2-1



            g_sampler = qmc.Sobol(d=rank, seed=0,bits=64)
            num_gaussians = gaussians.get_xyz.shape[0]
            g_sample = g_sampler.random(59 * num_gaussians) * grid_res
            g_weight = torch.tensor(g_sample, device="cuda", dtype=torch.float32).view(num_gaussians, 59, rank).int().float() / (0.5 * grid_res - 0.5) - 1
            gs_weight = [g_weight[:, ch_cut[p_idx]:ch_cut[p_idx + 1], :].view(means[p_idx].shape + (rank,))  for p_idx in range(len(means)) ]  #if p_idx in [0,3,4] else 1.0



        ############################################################################################################
        print('Training', opt.iterations, first_iter, debug_from)
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
                print("One up SH degree")

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))


            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            ############################################################################################################
            if args.with_uncertainty and iteration<=args.opacity_reset_interval:

                means = list(gaussians.capture()[1:7])
                scale_low=gaussians.capture_uncertainty()


                weights = [weights_all[iteration].view((1,) * len(means[p_idx].shape) + (rank,))* gs_weight[p_idx] for p_idx in range(6)]#


                if rank>0:

                    perturbed_means = \
                        [1*(weights[p_idx] * scale_low[p_idx][...,:rank]).sum(-1) + means[p_idx] for p_idx in range(len(means))]
                else:
                    weights1 = [torch.rand(means[p_idx].shape, device="cuda", dtype=torch.float32) * 2 - 1 for p_idx in range(6)]

                    if iteration==0:
                        print('rank 0')
                    perturbed_means = \
                        [ weights1[p_idx] * scale_low[p_idx][..., -1] + means[p_idx] for p_idx in range(len(means))]  #
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
                scale_mean=0
                if iteration % expand== 0 and iteration < opt.iterations-100:
                    scale_mean = sum([torch.abs(scale_low[p_idx]).sum() for p_idx in range(len(means))])

                loss+=-scale_mean/cur_views

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
                            gs_weight = [g_weight[:, ch_cut[p_idx]:ch_cut[p_idx + 1], :].view(means[p_idx].shape + (rank,))  for p_idx in range(len(means))]  #if p_idx in [0,3,4] else 1.0

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if len(ref_cameras)==0:
            # Active selection
            if args.with_uncertainty:
                selected_views = active_method.nbvs_testing(gaussians, scene, 1, pipe, background, exit_func=csm.should_exit,volume_scale=args.volume_scale)
            else:
                selected_views = active_method.nbvs(gaussians, scene, 1, pipe, background, exit_func=csm.should_exit)
            seq_cameras.extend(selected_views)
            # Save training sequence
            train_images = [view.original_image for view in scene.getTrainCameras()]
            seq_path = os.path.join(scene.model_path, "seq")
            mkdir_p(seq_path)
            for i, img in enumerate(train_images):
                torchvision.utils.save_image(img, os.path.join(seq_path, f"r_{i}.png"))

        else:
            next_cam_idx=len(seq_cameras)
            if next_cam_idx<len(ref_cameras):
                seq_cameras.append(ref_cameras[next_cam_idx])
            else:
                print('No more reference cameras')
                return seq_cameras
            # seq_cameras.append(ref_cameras[next_cam_idx])
    return seq_cameras

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
    parser.add_argument("--shuffle_data", action="store_true")
    parser.add_argument("--cf_nerf_name", type=str, default=None)
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

    seq_cameras=training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args)
    print(*seq_cameras, sep=' ')

    # All done
    print("\nTraining complete.")


    n_seq_cameras = len(seq_cameras)
    args.cam_seqeunce=seq_cameras
    args.cam_seqeunce_reference=seq_cameras
    args.num_views=n_seq_cameras+1
    args.with_uncertainty=False
    args.resolution=-1
    args.iterations=7000
    args.opacity_reset_interval=3000
    args.random_init_pcd=False
    print(args.with_uncertainty,args.resolution,args.random_init_pcd,args.shuffle_data)

    _=training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args)


    args.cam_seqeunce = seq_cameras[:n_seq_cameras//2]
    args.cam_seqeunce_reference = seq_cameras[:n_seq_cameras//2]
    args.num_views = n_seq_cameras//2+1
    print(args.with_uncertainty,args.resolution,args.random_init_pcd)
    _ = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
                 args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

