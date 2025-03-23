# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import os
import cv2
import numpy as np
import open3d as o3d
import torch
import random
from tqdm import tqdm

from colorama import Fore, Style
from multiprocessing.connection import Connection
from munch import munchify

from src.utils.datasets import get_dataset, load_mono_depth
from src.utils.common import as_intrinsics_matrix, setup_seed

from src.utils.Printer import Printer, FontColor

from thirdparty.glorie_slam.depth_video import DepthVideo
from thirdparty.gaussian_splatting.gaussian_renderer import render_virtual
from thirdparty.gaussian_splatting.utils.general_utils import rotation_matrix_to_quaternion, quaternion_multiply
from thirdparty.gaussian_splatting.utils.loss_utils import l1_loss, ssim
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from thirdparty.monogs.utils.pose_utils import update_pose, update_pose_knot, get_next_traj, slerp, compute_pose_error, get_next_traj_from_dspo
from thirdparty.monogs.utils.slam_utils import get_loss_mapping, get_median_depth, BAD_mapping_loss, render_video, BAD_tracking_loss, variance_of_laplacian, plot_tensor, BAD_tracking_loss_gap
from thirdparty.monogs.utils.camera_utils import Camera
from thirdparty.monogs.utils.rotation_conv import quaternion_to_matrix, matrix_to_quaternion
from src.utils.eval_utils import eval_ate, eval_rendering
from src.utils.eval_traj import kf_traj_eval
import torchvision.transforms.functional as TF

class Mapper(object):
    """
    Mapper thread.

    """
    def __init__(self, slam, pipe:Connection):
        # setup seed
        setup_seed(slam.cfg["setup_seed"])
        #torch.autograd.set_detect_anomaly(True)

        self.config = slam.cfg
        self.printer:Printer = slam.printer
        if self.config['only_tracking']:
            return
        self.pipe = pipe
        self.verbose = slam.verbose

        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None

        self.dtype = torch.float32
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None
      
        self.video:DepthVideo = slam.video

        model_params = munchify(self.config["mapping"]["model_params"])
        opt_params = munchify(self.config["mapping"]["opt_params"])
        pipeline_params = munchify(self.config["mapping"]["pipeline_params"])
        self.use_spherical_harmonics = self.config["mapping"]["Training"]["spherical_harmonics"]
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.cameras_extent = 6.0

        self.set_hyperparams()

        self.device = torch.device(self.config['device'])
       
        self.frame_reader = get_dataset(
            self.config, device=self.device)

        self.global_optimiz_video = []
        self.global_frame_idx = []

        self.global_optimiz_video_tracking = []
        self.global_sharp2gt = []
        self.frames_idx = []
        self.frames_idx_sharp2gt = []    
        self.render_videos = self.config["render_videos"]
        self.mapping_timings = []
        self.tracking_timings = []
        #torch.autograd.set_detect_anomaly(True)

    def set_pipe(self, pipe):
        self.pipe = pipe

    def set_hyperparams(self):
        mapping_config = self.config["mapping"]

        self.gt_camera = mapping_config["Training"]["gt_camera"]

        self.init_itr_num = mapping_config["Training"]["init_itr_num"]
        self.init_gaussian_update = mapping_config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = mapping_config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = mapping_config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * mapping_config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = mapping_config["Training"]["mapping_itr_num"]
        self.tracking_itr_num = mapping_config["Training"]["tracking_itr_num"]

        self.gaussian_update_every = mapping_config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = mapping_config["Training"]["gaussian_update_offset"]
        self.gaussian_th = mapping_config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * mapping_config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = mapping_config["Training"]["gaussian_reset"]
        self.size_threshold = mapping_config["Training"]["size_threshold"]
        self.window_size = mapping_config["Training"]["window_size"]

        self.save_dir = self.config['data']['output'] + '/' + self.config['scene']

        self.move_points = self.config['mapping']['move_points']
        self.online_plotting = self.config['mapping']['online_plotting']

        

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        # This function computes the new Gaussians to be added given a new keyframe
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )


    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
    
    def update_mapping_points(self, frame_idx, w2c, w2c_old, depth, depth_old, intrinsics, method=None):
        if method == "rigid":
            # just move the points according to their SE(3) transformation without updating depth
            frame_idxs = self.gaussians.unique_kfIDs # idx which anchored the set of points
            frame_mask = (frame_idxs==frame_idx) # global variable
            if frame_mask.sum() == 0:
                return
            # Retrieve current set of points to be deformed
            # But first we need to retrieve all mean locations and clone them
            means = self.gaussians.get_xyz.detach()
            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means[frame_mask], pix_ones), dim=1)
            means[frame_mask] = (transformation @ pts4.T).T[:, :3]
            # put the new means back to the optimizer
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(means, "xyz")["xyz"]
            # transform the corresponding rotation matrices
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(transformation.expand_as(rots[frame_mask]), rots[frame_mask])
           
            with torch.no_grad():
                self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(rots, "rotation")["rotation"]
        else:
            # Update pose and depth by projecting points into the pixel space to find updated correspondences.
            # This strategy also adjusts the scale of the gaussians to account for the distance change from the camera
           
            depth = depth.to(self.device)
            frame_idxs = self.gaussians.unique_kfIDs # idx which anchored the set of points
            frame_mask = (frame_idxs==frame_idx) # global variable
            if frame_mask.sum() == 0:
                return

            # Retrieve current set of points to be deformed
            means = self.gaussians.get_xyz.detach()[frame_mask]

            # Project the current means into the old camera to get the pixel locations
            pix_ones = torch.ones(means.shape[0], 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            pixel_locations = (intrinsics @ (w2c_old @ pts4.T)[:3, :]).T
            pixel_locations[:, 0] /= pixel_locations[:, 2]
            pixel_locations[:, 1] /= pixel_locations[:, 2]
            pixel_locations = pixel_locations[:, :2].long()
            height, width = depth.shape
            # Some pixels may project outside the viewing frustum.
            # Assign these pixels the depth of the closest border pixel
            pixel_locations[:, 0] = torch.clamp(pixel_locations[:, 0], min=0, max=width - 1)
            pixel_locations[:, 1] = torch.clamp(pixel_locations[:, 1], min=0, max=height - 1)

            # Extract the depth at those pixel locations from the new depth 
            depth = depth[pixel_locations[:, 1], pixel_locations[:, 0]]
            depth_old = depth_old[pixel_locations[:, 1], pixel_locations[:, 0]]
            # Next, we can either move the points to the new pose and then adjust the 
            # depth or the other way around.
            # Lets adjust the depth per point first
            # First we need to transform the global means into the old camera frame
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            means_cam = (w2c_old @ pts4.T).T[:, :3]

            rescale_scale = (1 + 1/(means_cam[:, 2])*(depth - depth_old)).unsqueeze(-1) # shift
            # account for 0 depth values - then just do rigid deformation
            rigid_mask = torch.logical_or(depth == 0, depth_old == 0)
            rescale_scale[rigid_mask] = 1
            if (rescale_scale <= 0.0).sum() > 0:
                rescale_scale[rescale_scale <= 0.0] = 1
        
            rescale_mean = rescale_scale.repeat(1, 3)
            means_cam = rescale_mean*means_cam

            # Transform back means_cam to the world space
            pts4 = torch.cat((means_cam, pix_ones), dim=1)
            means = (torch.linalg.inv(w2c_old) @ pts4.T).T[:, :3]

            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pts4 = torch.cat((means, pix_ones), dim=1)
            means = (transformation @ pts4.T).T[:, :3]

            # reassign the new means of the frame mask to the self.gaussian object
            global_means = self.gaussians.get_xyz.detach()
            global_means[frame_mask] = means
            # print("mean nans: ", global_means.isnan().sum()/global_means.numel())
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(global_means, "xyz")["xyz"]

            # update the rotation of the gaussians
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(transformation.expand_as(rots[frame_mask]), rots[frame_mask])
            self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(rots, "rotation")["rotation"]

            # Update the scale of the Gaussians
            scales = self.gaussians._scaling.detach()
            scales[frame_mask] = scales[frame_mask] + torch.log(rescale_scale)
            self.gaussians._scaling = self.gaussians.replace_tensor_to_optimizer(scales, "scaling")["scaling"]


    def get_w2c_and_depth(self, video_idx, idx, mono_depth, print_info=False, init=False):
        est_droid_depth, valid_depth_mask, c2w = self.video.get_depth_and_pose(video_idx, self.device)
        c2w = c2w.to(self.device)
        w2c = torch.linalg.inv(c2w)
        if print_info:
            pass
            #print(f"valid depth number: {valid_depth_mask.sum().item()}, " 
            #        f"valid depth ratio: {(valid_depth_mask.sum()/(valid_depth_mask.shape[0]*valid_depth_mask.shape[1])).item()}")
        if valid_depth_mask.sum() < 100:
            invalid = True
            print(f"Skip mapping frame {idx} at video idx {video_idx} because of not enough valid depth ({valid_depth_mask.sum()}).")  
        else:
            invalid = False

        est_droid_depth[~valid_depth_mask] = 0
        if not invalid:
            #mono_valid_mask = mono_depth < (mono_depth.mean()*3)
            mono_depth_flat = mono_depth[mono_depth > 0]
            if mono_depth_flat.numel() > 0:
                threshold = torch.quantile(mono_depth_flat, 0.95)
                mono_depth[mono_depth > threshold] = 0

            #mono_depth[mono_depth > 4*mono_depth.mean()] = 0
            from scipy.ndimage import binary_erosion
            mono_depth = mono_depth.cpu().numpy()
            binary_image = (mono_depth > 0).astype(int)
            # Add padding around the binary_image to protect the borders
            iterations = 5
            padded_binary_image = np.pad(binary_image, pad_width=iterations, mode='constant', constant_values=1)
            structure = np.ones((3, 3), dtype=int)
            # Apply binary erosion with padding
            eroded_padded_image = binary_erosion(padded_binary_image, structure=structure, iterations=iterations)
            # Remove padding after erosion
            eroded_image = eroded_padded_image[iterations:-iterations, iterations:-iterations]
            # set mono depth to zero at mask
            mono_depth[eroded_image == 0] = 0

            if (mono_depth == 0).sum() > 0:
                mono_depth = torch.from_numpy(cv2.inpaint(mono_depth, (mono_depth == 0).astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_NS)).to(self.device)
            else:
                mono_depth = torch.from_numpy(mono_depth).to(self.device)

            valid_mask = torch.from_numpy(eroded_image).to(self.device)*valid_depth_mask # new

            cur_wq = self.video.get_depth_scale_and_shift(video_idx, mono_depth, est_droid_depth, valid_mask)
            mono_depth_wq = mono_depth * cur_wq[0] + cur_wq[1]

            est_droid_depth[~valid_depth_mask] = mono_depth_wq[~valid_depth_mask]

        return est_droid_depth, w2c, invalid

    def initialize_map(self, cur_frame_idx, viewpoint):
        video_averages_tensor = []
        mapping_bar = tqdm(range(self.init_itr_num), desc="Initializing map")
        for mapping_iteration in mapping_bar:
            self.iteration_count += 1
            images_tensor = torch.empty((viewpoint.n_virtual_cams), 3, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
            depths_tensor = torch.empty((viewpoint.n_virtual_cams), 1, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
            gt_image = viewpoint.original_image
            R, t, theta, rho = viewpoint.get_virtual_extrinsics()
            for virtual_cam in range(viewpoint.n_virtual_cams):
                render_pkg = render_virtual(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam],rho = rho[virtual_cam] 
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                image_ab = image
                images_tensor[virtual_cam] = image_ab
                depths_tensor[virtual_cam] = depth
                if self.render_videos:
                    video_frame = (torch.clamp(torch.cat((image, gt_image), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)

                if mapping_iteration == self.init_itr_num - 1 and self.render_videos:
                    self.global_optimiz_video.append(video_frame)
                    self.global_frame_idx.append(cur_frame_idx)


            avg_image = images_tensor.mean(0)
            if self.render_videos:
                video_frame = (torch.clamp(torch.cat((avg_image, gt_image), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                #appending on new dim at the start
                video_averages_tensor.append(video_frame)

            loss_init = BAD_mapping_loss(
                self.config, avg_image, gt_image, images_tensor, depths_tensor, viewpoint, seen=False, initialization=True
            )
            loss_init.backward()

            mapping_bar.set_postfix({"loss": loss_init.item()})

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        self.printer.print("Initialized map", FontColor.MAPPER)

        # online plotting
        if self.online_plotting:
            from thirdparty.gaussian_splatting.utils.image_utils import psnr
            from src.utils.eval_utils import plot_rgbd_silhouette
            import cv2
            import numpy as np
            cur_idx = self.current_window[np.array(self.current_window).argmax()]
            viewpoint = self.viewpoints[cur_idx]
            R, t, theta, rho = viewpoint.get_virtual_extrinsics()
            images_tensor = torch.empty((viewpoint.n_virtual_cams), 3, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
            pnsr_sharp_array = []
            for virtual_cam in range(viewpoint.n_virtual_cams):
                render_pkg = render_virtual(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam],rho = rho[virtual_cam] 
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                (
                    image,
                    depth,
                ) = (
                    render_pkg["render"].detach(),
                    render_pkg["depth"].detach(),
                )
                image_ab = image
                images_tensor[virtual_cam] = image_ab

                sharp_img = torch.clamp(image, 0.0, 1.0)
                #take closest gt image
                gt_sharp = viewpoint.gt_images[int((virtual_cam/viewpoint.n_virtual_cams)*len(viewpoint.gt_images))]
                mask = gt_sharp > 0
                pnsr_sharp = psnr((sharp_img[mask]).unsqueeze(0), (gt_sharp[mask]).unsqueeze(0))
                pnsr_sharp_array.append(pnsr_sharp)

            image = images_tensor.mean(0)
            gt_image = viewpoint.original_image
            gt_depth = viewpoint.depth

            image = torch.clamp(image, 0.0, 1.0)
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            mask = gt_image > 0
            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            diff_depth_l1 = torch.abs(depth.detach().cpu() - gt_depth)
            diff_depth_l1 = diff_depth_l1 * (gt_depth > 0)
            depth_l1 = diff_depth_l1.sum() / (gt_depth > 0).sum()

            # Add plotting 2x3 grid here
            plot_dir = self.save_dir + "/online_plots"
            plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1,
                                    psnr_score.item(), depth_l1, plot_dir=plot_dir, idx=str(cur_idx),
                                    diff_rgb=np.abs(gt - pred), sharp_psnr = torch.stack(pnsr_sharp_array).mean(0).item())
        render_video(os.path.join(self.save_dir,"init_mapping_averages.mp4"), video_averages_tensor, 10)

        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        print("Memory allocated before map", torch.cuda.memory_allocated('cuda')/1000/1000)
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        video_averages_tensor = [[] for i in range(len(current_window))]
        video_virtuals_tensor = [[] for i in range(self.viewpoints[current_window[0]].n_virtual_cams * len(current_window))]

        mapping_bar = tqdm(range(iters), desc="Mapping at frame {}".format(current_window[0]))

        for it in mapping_bar:
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                images_tensor = torch.empty((viewpoint.n_virtual_cams), 3, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
                depths_tensor = torch.empty((viewpoint.n_virtual_cams), 1, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
                touched_cam = []
                gt_image = viewpoint.original_image
                R, t, theta, rho = viewpoint.get_virtual_extrinsics()
                for virtual_cam in range(viewpoint.n_virtual_cams):
                    render_pkg = render_virtual(
                        viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam],rho = rho[virtual_cam] 
                    )
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )

                    image_ab = image
                    images_tensor[virtual_cam] = image_ab
                    depths_tensor[virtual_cam] = depth
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)
                    touched_cam.append(n_touched)
                    if self.render_videos:
                        video_frame = (torch.clamp(torch.cat((image_ab, gt_image), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                        if virtual_cam%2==0:
                            video_virtuals_tensor[viewpoint.n_virtual_cams*cam_idx + virtual_cam].append(video_frame)
                    if it == iters - 1 and cam_idx == 0 and iters > 1 and self.render_videos:
                        self.global_optimiz_video.append(video_frame)
                        self.global_frame_idx.append(current_window[0])
                
                avg_image = images_tensor.mean(0)

                if self.render_videos:
                    video_frame = (torch.clamp(torch.cat((avg_image, gt_image), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)

                    video_averages_tensor[cam_idx].append(video_frame)

                seen = not (cam_idx == 0)

                loss_mapping += BAD_mapping_loss(
                    self.config, avg_image, gt_image, images_tensor, depths_tensor, viewpoint, seen
                )

                touched_cam = torch.stack(touched_cam)
                n_touched_acm.append(touched_cam.max(dim=0).values)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                images_tensor = torch.empty((viewpoint.n_virtual_cams), 3, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
                depths_tensor = torch.empty((viewpoint.n_virtual_cams), 1, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
                gt_image = viewpoint.original_image
                R, t, theta, rho = viewpoint.get_virtual_extrinsics()

                for virtual_cam in range(viewpoint.n_virtual_cams):

                    render_pkg = render_virtual(
                        viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam], rho = rho[virtual_cam] 
                    )
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )

                    image_ab = image
                    images_tensor[virtual_cam] = image_ab
                    depths_tensor[virtual_cam] = depth
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)
                
                avg_image = images_tensor.mean(0)

                loss_mapping += BAD_mapping_loss(
                    self.config, avg_image, gt_image, images_tensor, depths_tensor, viewpoint,
                )

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()

            mapping_bar.set_postfix({"loss": loss_mapping.item()})

            gaussian_split = False
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # compute the visibility of the gaussians
                # Only prune on the last iteration and when we have a full window
                if prune:
                    if len(current_window) == self.window_size:
                        prune_mode = self.config["mapping"]["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None:
                            print("Pruning", torch.sum(to_prune), "gaussians over", self.gaussians.get_xyz.shape)
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True # not used it seems

                ## Opacity reset
                # self.iteration_count is a global parameter. We use gaussian reset
                # every 2001 iterations meaning if we use 60 per mapping frame
                # and there are 160 keyframes in the sequence, we do resetting
                # 4 times. Using more mapping iterations leads to more resetting
                # which can prune away more gaussians.
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    self.printer.print("Resetting the opacity of non-visible Gaussians", FontColor.MAPPER)
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                # comment for debugging
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    for knot in range(viewpoint.num_control_knots):
                        update_pose_knot(viewpoint, knot)
        
        if self.render_videos:
            for j in range(len(current_window)):
                render_video(os.path.join(self.save_dir,f"mapping_averages_{current_window[0]}_{j}.mp4"), video_averages_tensor[j], 10)
                for i in range(viewpoint.n_virtual_cams):
                    render_video(os.path.join(self.save_dir,f"mapping_virtuals_{current_window[0]}_cam_{i}_wind_{j}.mp4"), video_virtuals_tensor[viewpoint.n_virtual_cams*j + i], 5)
            
            render_video(os.path.join(self.save_dir,"global_mapping.mp4"), self.global_optimiz_video, 5, self.global_frame_idx)
            

        # online plotting
        if self.online_plotting:
            from thirdparty.gaussian_splatting.utils.image_utils import psnr
            from src.utils.eval_utils import plot_rgbd_silhouette
            import cv2
            import numpy as np
            cur_idx = current_window[np.array(current_window).argmax()]
            viewpoint = self.viewpoints[cur_idx]
            R, t, theta, rho = viewpoint.get_virtual_extrinsics()
            images_tensor = torch.empty((viewpoint.n_virtual_cams), 3, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
            depths_tensor = torch.empty((viewpoint.n_virtual_cams), 1, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
            pnsr_sharp_array = []
            for virtual_cam in range(viewpoint.n_virtual_cams):
                render_pkg = render_virtual(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam],rho = rho[virtual_cam] 
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                (
                    image,
                    depth,
                ) = (
                    render_pkg["render"].detach(),
                    render_pkg["depth"].detach(),
                )
                
                image_ab = image
                images_tensor[virtual_cam] = image_ab
                depths_tensor[virtual_cam] = depth

                sharp_img = torch.clamp(image, 0.0, 1.0)
                gt_sharp = viewpoint.gt_images[int((virtual_cam/viewpoint.n_virtual_cams)*len(viewpoint.gt_images))]
                mask = gt_sharp > 0
                pnsr_sharp = psnr((sharp_img[mask]).unsqueeze(0), (gt_sharp[mask]).unsqueeze(0))
                pnsr_sharp_array.append(pnsr_sharp)

            image = images_tensor.mean(0)

            gt_image = viewpoint.original_image
            gt_depth = viewpoint.depth 

            image = torch.clamp(image, 0.0, 1.0)
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            mask = gt_image > 0
            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            diff_depth_l1 = torch.abs(depth.detach().cpu() - gt_depth)
            diff_depth_l1 = diff_depth_l1 * (gt_depth > 0)
            depth_l1 = diff_depth_l1.sum() / (gt_depth > 0).sum()

            # Add plotting 2x3 grid here
            plot_dir = self.save_dir + "/online_plots"
            plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1,
                                    psnr_score.item(), depth_l1, plot_dir=plot_dir, idx=str(cur_idx),
                                    diff_rgb=np.abs(gt - pred), sharp_psnr = torch.stack(pnsr_sharp_array).mean(0).item())
        
        return gaussian_split


    def final_refine(self, prune=False, iters=26000):
        self.printer.print("Starting final refinement", FontColor.MAPPER)
        w2c_temps = []
        depth_temps = []


        # Do final update of depths and poses
        for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):
            _, _, depth_gtd, _, _ = self.frame_reader[frame_idx]
            #depth_gt_numpy = depth_gtd.cpu().numpy()
            intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(self.device)
            mono_depth = load_mono_depth(frame_idx, self.save_dir).to(self.device)
            depth_temp, w2c_temp, invalid = self.get_w2c_and_depth(keyframe_idx, frame_idx, mono_depth, init=False)
            w2c_temps.append(w2c_temp)
            depth_temps.append(depth_temp)
            # Update tracking parameters
            for knot in range(self.cameras[keyframe_idx].num_control_knots):
                R, t, _, _ = self.cameras[keyframe_idx].get_mid_extrinsic()
                w2c_old = torch.cat((self.cameras[keyframe_idx].R_i[knot], self.cameras[keyframe_idx].t_i[knot].unsqueeze(-1)), dim=1)
                w2c_old = torch.cat((w2c_old, torch.tensor([[0, 0, 0, 1]], device="cuda")), dim=0)
                self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3], knot)
            # Update depth for viewpoint
            self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()

            if keyframe_idx in self.viewpoints:
                # Update tracking parameters
                for knot in range(self.viewpoints[keyframe_idx].num_control_knots):
                    self.viewpoints[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3], knot)
                # Update depth for viewpoint
                self.viewpoints[keyframe_idx].depth = depth_temp.cpu().numpy()

            # Update mapping parameters
                if self.move_points and self.is_kf[keyframe_idx]:
                    if invalid:
                        self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics, method="rigid")
                    else:
                        self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics)
                        self.depth_dict[keyframe_idx] = depth_temp # not needed since it is the last deformation but keeping for clarity.
        
        for idx, (keyframe_idx, frame_idx) in enumerate(zip(self.keyframe_idxs, self.video_idxs)):
            if idx > 0:
                prev_keyframe_idx = self.keyframe_idxs[idx - 1]
                prev_frame_idx = self.video_idxs[idx - 1]
            else:
                prev_keyframe_idx = None
                prev_frame_idx = None

            # Find next keyframe and its frame index
            if idx < len(self.keyframe_idxs) - 1:
                next_keyframe_idx = self.keyframe_idxs[idx + 1]
                next_frame_idx = self.video_idxs[idx + 1]
            else:
                next_keyframe_idx = None
                next_frame_idx = None
            
            R, t, _, _ = self.cameras[frame_idx].get_mid_extrinsic()
            mid_q_cur = matrix_to_quaternion(R)

            # Handle previous frame
            if prev_keyframe_idx is not None:
                R_prev, t_prev, _, _ = self.cameras[prev_frame_idx].get_mid_extrinsic()
                mid_q_prev = matrix_to_quaternion(R_prev)
                delta_prev = keyframe_idx - prev_keyframe_idx
                fraction_start = (keyframe_idx - 0.5 + self.cameras[frame_idx].prev_gap - prev_keyframe_idx) / delta_prev

                t_expected_start = torch.lerp(t_prev, t, fraction_start)
                q_expected_start = slerp(torch.tensor(fraction_start), mid_q_prev, mid_q_cur)
                R_start = quaternion_to_matrix(q_expected_start)

                with torch.no_grad():
                    self.cameras[frame_idx].update_RT(R_start, t_expected_start, 0)

            # Handle next frame
            if next_keyframe_idx is not None:
                R_next, t_next, _, _ = self.cameras[next_frame_idx].get_mid_extrinsic()
                mid_q_next = matrix_to_quaternion(R_next)
                delta_next = next_keyframe_idx - keyframe_idx
                fraction_end = (0.5 + self.cameras[frame_idx].next_gap)  / delta_next

                t_expected_end = torch.lerp(t, t_next, fraction_end)
                q_expected_end = slerp(torch.tensor(fraction_end), mid_q_cur, mid_q_next)
                R_end = quaternion_to_matrix(q_expected_end)

                with torch.no_grad():
                    self.cameras[frame_idx].update_RT(R_end, t_expected_end, self.cameras[frame_idx].num_control_knots - 1)
            
            if self.cameras[frame_idx].num_control_knots > 2 and next_keyframe_idx is not None and prev_keyframe_idx is not None:
                fraction_mid_start = 0.33
                fraction_mid_end = 0.66

                t_second = torch.lerp(t_expected_start, t_expected_end, fraction_mid_start)
                q_second = slerp(torch.tensor(fraction_mid_start), q_expected_start, q_expected_end)
                R_second = quaternion_to_matrix(q_second)

                t_third = torch.lerp(t_expected_start, t_expected_end, fraction_mid_end)
                q_third = slerp(torch.tensor(fraction_mid_end), q_expected_start, q_expected_end)
                R_third = quaternion_to_matrix(q_third)

                with torch.no_grad():
                    self.cameras[frame_idx].update_RT(R_second, t_second, 1)
                    self.cameras[frame_idx].update_RT(R_third, t_third, 2)

        random_viewpoint_stack = []
        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]

        for cam_idx, viewpoint in self.viewpoints.items():
            random_viewpoint_stack.append(viewpoint)

        for _ in tqdm(range(iters)):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []
           
            rand_idx = np.random.randint(0, len(random_viewpoint_stack))
            viewpoint = random_viewpoint_stack[rand_idx]
            images_tensor = torch.empty((viewpoint.n_virtual_cams), 3, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
            depths_tensor = torch.empty((viewpoint.n_virtual_cams), 1, viewpoint.image_height, viewpoint.image_width, device="cuda:0")

            R, t, theta, rho = viewpoint.get_virtual_extrinsics()
            for virtual_cam in range(viewpoint.n_virtual_cams):
                render_pkg = render_virtual(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam], rho = rho[virtual_cam] 
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                image_ab = image
                images_tensor[virtual_cam] = image_ab
                depths_tensor[virtual_cam] = depth
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
            
            avg_image = images_tensor.mean(0)
            gt_image = viewpoint.original_image
            loss_mapping += BAD_mapping_loss(
                    self.config, avg_image, gt_image, images_tensor, depths_tensor, viewpoint,
                )

            n_touched_acm.append(n_touched)

            scaling = self.gaussians.get_scaling
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                # optimize the exposure compensation
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
        
        self.printer.print("Final refinement done", FontColor.MAPPER)


    def initialize(self, cur_frame_idx, viewpoint):
        # self.initialized only False at beginning for monocular MonoGS
        # in the slam_frontend.py it is used in the monocular setting
        # for some minor things for bootstrapping, but it is not relevant
        # in out "with proxy depth" setting.
        self.initialized = True
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        self.mapped_video_idxs = []
        self.mapped_kf_idxs = []

        # Initialise the frame at the ground truth pose
        for knot in range(viewpoint.num_control_knots):
            viewpoint.update_RT(viewpoint.R_gt[knot], viewpoint.T_gt[knot], knot)


    def add_new_keyframe(self, cur_frame_idx, idx, depth=None, opacity=None):
        rgb_boundary_threshold = self.config["mapping"]["Training"]["rgb_boundary_threshold"]
        self.mapped_video_idxs.append(cur_frame_idx)
        self.mapped_kf_idxs.append(idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        # Filter out RGB pixels where the R + G + B values < 0.01
        # valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        valid_rgb = (gt_img.sum(dim=0) > -1)[None]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels. THIS LINE OVERWRITES THE self.viewpoints[cur_frame_idx].depth with "initial_depth"
        return initial_depth[0].cpu().numpy()

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["mapping"]["Training"]["kf_translation"]
        kf_min_translation = self.config["mapping"]["Training"]["kf_min_translation"]
        kf_overlap = self.config["mapping"]["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        R, t, _, _ = curr_frame.get_mid_extrinsic()
        R_last, t_last, _, _ = last_kf.get_mid_extrinsic()
        pose_CW = getWorld2View2(R, t)
        last_kf_CW = getWorld2View2(R_last, t_last)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        # multiply by median depth in rgb-only setting to account for scale ambiguity
        dist_check = dist > kf_translation * self.median_depth 
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["mapping"]["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["mapping"]["Training"]
                else 0.4
            )
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        R, t, _, _ = curr_frame.get_mid_extrinsic()
        kf_0_WC = torch.linalg.inv(getWorld2View2(R, t))
        
        if len(window) > self.window_size:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                R_i, t_i, _, _ = kf_i.get_mid_extrinsic()
                kf_i_CW = getWorld2View2(R_i, t_i)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    R_j, t_j, _, _ = kf_j.get_mid_extrinsic()
                    kf_j_WC = torch.linalg.inv(getWorld2View2(R_j, t_j))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    
    def tracking(self, cur_frame_idx, viewpoint):
        print("Memory allocated before tracking", torch.cuda.memory_allocated('cuda')/1000/1000)

        print("Tracking frame", cur_frame_idx)
        if cur_frame_idx - 1 in self.cameras:
            prev = self.cameras[cur_frame_idx - 1]
        else:
            prev = self.cameras[0]
        print(viewpoint.uid, prev.uid)

        opt_params = []
        for i in range(viewpoint.num_control_knots):
            opt_params.append(
                {
                    "params": [viewpoint.T_i_rot_delta[i]],
                    "lr": self.config['mapping']["Training"]["lr"]["cam_rot_delta"],
                    "name": "rot_{}_{}".format(i, viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.T_i_trans_delta[i]],
                    "lr": self.config['mapping']["Training"]["lr"]["cam_trans_delta"],
                    "name": "trans_{}_{}".format(i, viewpoint.uid),
                }
            )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )
            
        video_averages_tensor = []
        #video_virtuals_tensor = [ [] for i in range(viewpoint.n_virtual_cams)]
        pose_optimizer = torch.optim.Adam(opt_params)

        gt_image = viewpoint.original_image

        #measure diff between knot gt poses
        tracking_bar = tqdm(range(self.tracking_itr_num), 
                                desc="Tracking frame {}".format(cur_frame_idx))
        for tracking_itr in tracking_bar:    

            loss_tracking = 0
                
            images_tensor = torch.empty((viewpoint.n_virtual_cams), 3, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
            depths_tensor = torch.empty((viewpoint.n_virtual_cams), 1, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
            opacities_tensor = torch.empty((viewpoint.n_virtual_cams), 1, viewpoint.image_height, viewpoint.image_width, device="cuda:0")
            
            render_pkg_ret = []
            R, t, theta, rho = viewpoint.get_virtual_extrinsics()

            for virtual_cam in range(viewpoint.n_virtual_cams):
                render_pkg = render_virtual(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam], rho = rho[virtual_cam] 
                )

                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )
                image_ab = image
                images_tensor[virtual_cam] = image_ab
                depths_tensor[virtual_cam] = depth
                opacities_tensor[virtual_cam] = opacity
                #if index_viewpoint == 1:
                with torch.no_grad():
                    if self.render_videos:
                        video_frame_sharp = (torch.clamp(torch.cat((image, gt_image), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                        #video_frame_sharp2gt = (torch.clamp(torch.cat((image, viewpoint.gt_images[i]), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                    #if i%2==0:
                    #    video_virtuals_tensor[i].append(video_frame_sharp)
                    if tracking_itr == self.tracking_itr_num-1 and self.render_videos:
                        self.global_optimiz_video_tracking.append(video_frame_sharp)
                        #self.global_sharp2gt.append(video_frame_sharp2gt)
                        self.frames_idx.append(cur_frame_idx)
                        self.frames_idx_sharp2gt.append(cur_frame_idx)
                    
                render_pkg_ret.append(render_pkg)

                # Delete variables not needed anymore
                del render_pkg, image, opacity,
                torch.cuda.empty_cache()

            avg_image = images_tensor.mean(0)
            avg_opacity = opacities_tensor.mean(0)

            #stack gt image and then save them together
            #if index_viewpoint == 1:
            with torch.no_grad():
                if self.render_videos:
                    video_frame = (torch.clamp(torch.cat((avg_image, gt_image), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                    video_averages_tensor.append(video_frame)
            
            torch.cuda.empty_cache()

            pose_optimizer.zero_grad()

            loss_tracking += BAD_tracking_loss(
                self.config, avg_image, viewpoint.original_image, images_tensor, opacities_tensor, viewpoint, prev
            )

            tracking_bar.set_postfix({"Loss": loss_tracking.item()})

            # Delete images and opacities tensors
            del images_tensor, opacities_tensor, avg_image

            loss_tracking.backward()
            converged = False
            with torch.no_grad():
                pose_optimizer.step()
            
                converged = True
                
                for i in range(prev.num_control_knots):
                    update_pose_knot(prev, i)
                
                for i in range(viewpoint.num_control_knots):
                    #("Updating pose knot: ", i)
                    converged = update_pose_knot(viewpoint, i) and converged
                
            if converged:
                print("Tracking converged...")

                with torch.no_grad():
                    for i in range(viewpoint.num_control_knots):
                        viewpoint.T_i_rot_delta[i].data.fill_(0)
                        viewpoint.T_i_trans_delta[i].data.fill_(0)
                        
                    R, t, theta, rho = viewpoint.get_virtual_extrinsics()

                    for virtual_cam in range(viewpoint.n_virtual_cams):
                        render_pkg = render_virtual(
                            viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam], rho = rho[virtual_cam] 
                        )

                        image, depth, opacity = (
                            render_pkg["render"],
                            render_pkg["depth"],
                            render_pkg["opacity"],
                        )
                        if self.render_videos:
                            video_frame_sharp = (torch.clamp(torch.cat((image, gt_image), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                            #video_frame_sharp2gt = (torch.clamp(torch.cat((image, viewpoint.gt_images[i]), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                        if (len(self.frames_idx) == 0 or self.frames_idx[-1] != cur_frame_idx) and self.render_videos:
                            self.global_optimiz_video_tracking.append(video_frame_sharp)
                            #self.global_sharp2gt.append(video_frame_sharp2gt)
                            self.frames_idx.append(cur_frame_idx)
                            self.frames_idx_sharp2gt.append(cur_frame_idx)
                    break
        if self.render_videos:
            print("Creating videos")
            render_video(os.path.join(self.save_dir,"global_tracking.mp4"), self.global_optimiz_video_tracking, 5, self.frames_idx)
            #render_video(os.path.join(self.save_dir,"global_tracking_sharp2gt.mp4"), self.global_sharp2gt, 5, self.frames_idx_sharp2gt)
            render_video(os.path.join(self.save_dir,f"tracking_averages_{cur_frame_idx}.mp4"), video_averages_tensor,10)
            #for i in range(viewpoint.n_virtual_cams):
            #    render_video(os.path.join(self.save_dir_video,f"tracking_virtuals_{cur_frame_idx}_{i}.mp4"), video_virtuals_tensor[i], 5)

        del video_averages_tensor
        del avg_opacity
        return render_pkg_ret


    def run(self):
        """
        Trigger mapping process, get estimated pose and depth from tracking process,
        send continue signal to tracking process when the mapping of the current frame finishes.  
        """
        config = self.config

        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.frame_reader.fx,
            fy=self.frame_reader.fy,
            cx=self.frame_reader.cx,
            cy=self.frame_reader.cy,
            W=self.frame_reader.W_out,
            H=self.frame_reader.H_out,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
    
        num_frames = len(self.frame_reader)

        # Initialize list to keep track of Keyframes
        self.keyframe_idxs = [] # 
        self.video_idxs = [] # keyframe numbering (note first
        # keyframe for mapping is the 7th keyframe in total)
        self.is_kf = dict() # keys are video_idx and value is boolean. This prevents trying to deform frames that were never mapped.
        # this is only a problem when the last keyframe is not mapped as this would otherwise be handled by the code.
        
        # Init Variables to keep track of ground truth poses and runtimes
        self.gt_w2c_all_frames = []

        init = True

        # Define first frame pose
        _, color, _, first_frame_c2w, gt_sharp = self.frame_reader[0]
        intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(self.device)

        # Create dictionary which stores the depth maps from the previous iteration
        # This depth is used during map deformation if we have missing pixels
        self.depth_dict = dict()
        # global camera dictionary - updated during mapping.
        self.cameras = dict()
        self.depth_dict = dict()

        self.prev_cameras = dict()
        self.next_cameras = dict()

        while (1):
            frame_info = self.pipe.recv()
            idx = frame_info['timestamp'] # frame index
            video_idx = frame_info['video_idx'] # keyframe index
            is_finished = frame_info['end']
            is_keyframe_flag = frame_info['is_keyframe']


            if not is_keyframe_flag:
                print("START TRACKING GAP")
                print(self.video_idxs, self.keyframe_idxs)
                viewpoint = self.cameras[video_idx]
                R, t, _, _ = viewpoint.get_mid_extrinsic()
                mid_q_cur = matrix_to_quaternion(R)  

                index_on_array = self.video_idxs.index(video_idx)
                keyframe_idx = self.keyframe_idxs[index_on_array]
                frame_idx = self.video_idxs[index_on_array]

                if video_idx not in self.prev_cameras:
                    prev_keyframe_idx = self.keyframe_idxs[index_on_array - 1]
                    prev_frame_idx = self.video_idxs[index_on_array - 1]
                    
                    print("prev_keyframe_idx", prev_keyframe_idx, "prev_frame_idx", prev_frame_idx)

                    _, color, depth_gt, c2w_gt, gt_sharp = self.frame_reader[prev_keyframe_idx]
                    color = color.to(self.device)
                    c2w_gt = c2w_gt.to(self.device)
                    w2c_gt = torch.linalg.inv(c2w_gt)
                    mono_depth = load_mono_depth(prev_keyframe_idx, self.save_dir).to(self.device)

                    depth, w2c, invalid = self.get_w2c_and_depth(prev_frame_idx, idx, mono_depth, init=False, print_info=True)

                    R_prev, t_prev, _, _ = self.cameras[prev_frame_idx].get_mid_extrinsic()
                    mid_q_prev = matrix_to_quaternion(R_prev)
                    delta_prev = keyframe_idx - prev_keyframe_idx
                    fraction_start = (keyframe_idx - 0.5 - prev_keyframe_idx) / delta_prev

                    print("interpolating between ", prev_keyframe_idx, " and ", keyframe_idx, "at", fraction_start)

                    t_expected_start = torch.lerp(t_prev, t, fraction_start)
                    q_expected_start = slerp(torch.tensor(fraction_start), mid_q_prev, mid_q_cur)
                    R_start = quaternion_to_matrix(q_expected_start)

                    #with torch.no_grad():
                    #    #self.cameras[frame_idx].update_RT(R_start, t_expected_start, 0)

                    w2c_prev = torch.eye(4).to(self.device)
                    w2c_prev[0:3, 0:3] = R_start
                    w2c_prev[0:3, 3] = t_expected_start

                    _, color, depth_gt, c2w_gt, gt_sharp = self.frame_reader[idx-1]
                    color = color.to(self.device)
                    c2w_gt = c2w_gt.to(self.device)
                    w2c_gt = torch.linalg.inv(c2w_gt)

                    data = {"gt_color": color.squeeze(), "glorie_depth": depth.cpu().numpy(), "glorie_pose": w2c_prev, \
                        "gt_pose": w2c_gt, "idx": video_idx - 0.5,
                        "gt_images": gt_sharp, "n_virtual_cams": self.config["n_virtual_cams"],
                        "interpolation": self.config["interpolation"]
                        }
                    viewpoint_prev = Camera.init_from_dataset(
                        self.frame_reader, data, projection_matrix, 
                    )
                    self.prev_cameras[video_idx] = viewpoint_prev
                    viewpoint_prev.compute_grad_mask(self.config)
                    for knot in range(viewpoint_prev.num_control_knots):
                        viewpoint_prev.update_RT(viewpoint_prev.R_gt[knot], viewpoint_prev.T_gt[knot], knot)

                else:
                    viewpoint_prev = self.prev_cameras[video_idx]


                if video_idx not in self.next_cameras:

                    next_keyframe_idx = self.keyframe_idxs[index_on_array + 1]
                    next_frame_idx = self.video_idxs[index_on_array + 1]
                    
                    print("next_keyframe_idx", next_keyframe_idx, "next_frame_idx", next_frame_idx)

                    _, color, depth_gt, c2w_gt, gt_sharp = self.frame_reader[next_keyframe_idx]
                    mono_depth = load_mono_depth(next_keyframe_idx, self.save_dir).to(self.device)
                    color = color.to(self.device)
                    c2w_gt = c2w_gt.to(self.device)
                    w2c_gt = torch.linalg.inv(c2w_gt)
                    depth, w2c, invalid = self.get_w2c_and_depth(next_frame_idx, idx, mono_depth, init=False, print_info=True)

                    R_next, t_next, _, _ = self.cameras[next_frame_idx].get_mid_extrinsic()
                    mid_q_next = matrix_to_quaternion(R_next)
                    delta_next = next_keyframe_idx - keyframe_idx
                    fraction_end = 0.5 / delta_next

                    print("interpolating between ", keyframe_idx, " and ", next_keyframe_idx, "at", fraction_end)

                    t_expected_end = torch.lerp(t, t_next, fraction_end)
                    q_expected_end = slerp(torch.tensor(fraction_end), mid_q_cur, mid_q_next)
                    R_end = quaternion_to_matrix(q_expected_end)

                    w2c_next = torch.eye(4).to(self.device)
                    w2c_next[0:3, 0:3] = R_end
                    w2c_next[0:3, 3] = t_expected_end

                    _, color, depth_gt, c2w_gt, gt_sharp = self.frame_reader[idx+1]
                    color = color.to(self.device)
                    c2w_gt = c2w_gt.to(self.device)
                    w2c_gt = torch.linalg.inv(c2w_gt)

                    data = {"gt_color": color.squeeze(), "glorie_depth": depth.cpu().numpy(), "glorie_pose": w2c_next, \
                        "gt_pose": w2c_gt, "idx": video_idx + 0.5,
                        "gt_images": gt_sharp, "n_virtual_cams": self.config["n_virtual_cams"],
                        "interpolation": self.config["interpolation"]
                        }
                    viewpoint_next = Camera.init_from_dataset(
                        self.frame_reader, data, projection_matrix, 
                    )
                    self.next_cameras[video_idx] = viewpoint_next
                    viewpoint_next.compute_grad_mask(self.config)
                    for knot in range(viewpoint_next.num_control_knots):
                        viewpoint_next.update_RT(viewpoint_next.R_gt[knot], viewpoint_next.T_gt[knot], knot)
                else:
                    viewpoint_next = self.next_cameras[video_idx]

                self.tracking_est_gap(viewpoint_prev, viewpoint, viewpoint_next, video_idx)
                self.pipe.send("continue")
                continue

            print("Received frame ", idx, " video_idx ", video_idx)

            if self.verbose:
                self.printer.print(f"\nMapping Frame {idx} ...", FontColor.MAPPER)
            
            if is_finished:
                print("Done with Mapping and Tracking")
                break

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx)
                print(Style.RESET_ALL)

            self.keyframe_idxs.append(idx)
            self.video_idxs.append(video_idx)

            _, color, depth_gt, c2w_gt, gt_sharp = self.frame_reader[idx]
            mono_depth = load_mono_depth(idx, self.save_dir).to(self.device)
            color = color.to(self.device)
            c2w_gt = c2w_gt.to(self.device) 
            w2c_gt = torch.linalg.inv(c2w_gt)
            #depth_gt_numpy = depth_gt.numpy()
            #depth_gt = depth_gt.to(self.device)

            depth, w2c, invalid = self.get_w2c_and_depth(video_idx, idx, mono_depth, init=False, print_info=True)

            if invalid:
                print("WARNING: Too few valid pixels from droid depth")
                 # online glorieslam pose and depth
                data = {"gt_color": color.squeeze(), "glorie_depth": depth.cpu().numpy(), "glorie_pose": w2c, \
                        "gt_pose": w2c_gt, "idx": video_idx,
                        "gt_images": gt_sharp, "n_virtual_cams": self.config["n_virtual_cams"],
                        "interpolation": self.config["interpolation"]}
                self.is_kf[video_idx] = False
                viewpoint = Camera.init_from_dataset(
                        self.frame_reader, data, projection_matrix, 
                    )
                # update the estimated pose to be the glorie pose
                for knot in range(viewpoint.num_control_knots):
                    viewpoint.update_RT(viewpoint.R_gt[knot], viewpoint.T_gt[knot], knot)
                viewpoint.compute_grad_mask(self.config)
                # Dictionary of Camera objects at the frame index
                # self.cameras contains all cameras.
                self.cameras[video_idx] = viewpoint
                self.pipe.send("continue")

                
                if False and  len(self.cameras)>1:
                    #check difference between glorie init and constant velocity init
                    glorie_error = compute_pose_error(w2c, w2c_gt)
                    
                    R_i, t_i = get_next_traj(self.cameras[video_idx-1])
                    
                    T_0 = torch.eye(4).to(self.device)
                    T_0[0:3, 0:3] = R_i[0]
                    T_0[0:3, 3] = t_i[0]

                    T_1 = torch.eye(4).to(self.device)
                    T_1[0:3, 0:3] = R_i[1]
                    T_1[0:3, 3] = t_i[1]

                    pred_w2c = torch.zeros((2, 4, 4)).to(self.device)
                    pred_w2c[0] = T_0
                    pred_w2c[1] = T_1

                    pred_error = compute_pose_error(pred_w2c, w2c_gt)

                    print("Glorie Error: ", glorie_error, "Pred Error: ", pred_error)
                

                continue # too few valid pixels from droid depth
            
            w2c_gt = torch.linalg.inv(c2w_gt)
            self.gt_w2c_all_frames.append(w2c_gt)

            if len(self.cameras) > 1:
                _, w2c_prev, _ = self.get_w2c_and_depth(video_idx-1, idx, mono_depth, init=False, print_info=True)


            # online glorieslam pose and depth
            data = {"gt_color": color.squeeze(), "glorie_depth": depth.cpu().numpy(), "glorie_pose": w2c, \
                    "gt_pose": w2c_gt, "idx": video_idx,
                    "gt_images": gt_sharp, "n_virtual_cams": self.config["n_virtual_cams"],
                    "interpolation": self.config["interpolation"]
                    }
            print("w2c_gt", w2c_gt.shape)
            viewpoint = Camera.init_from_dataset(
                    self.frame_reader, data, projection_matrix, 
                )

            
            if False and len(self.cameras)>1:
                #check difference between glorie init and constant velocity init
                glorie_error = compute_pose_error(w2c, w2c_gt)
                
                R_i, t_i = get_next_traj(self.cameras[video_idx-1])
                
                T_0 = torch.eye(4).to(self.device)
                T_0[0:3, 0:3] = R_i[0]
                T_0[0:3, 3] = t_i[0]

                T_1 = torch.eye(4).to(self.device)
                T_1[0:3, 0:3] = R_i[1]
                T_1[0:3, 3] = t_i[1]

                pred_w2c = torch.zeros((2, 4, 4)).to(self.device)
                pred_w2c[0] = T_0
                pred_w2c[1] = T_1

                pred_error = compute_pose_error(pred_w2c, w2c_gt)

                print("Glorie Error: ", glorie_error, "Pred Error: ", pred_error)
            

            # update the estimated pose to be the glorie pose
            for knot in range(viewpoint.num_control_knots):
                viewpoint.update_RT(viewpoint.R_gt[knot], viewpoint.T_gt[knot], knot)

            viewpoint.compute_grad_mask(self.config)
            # Dictionary of Camera objects at the frame index
            # self.cameras contains all cameras.
            self.cameras[video_idx] = viewpoint

            if init:
                self.initialize(video_idx, viewpoint)

                self.printer.print("Resetting the system", FontColor.MAPPER)
                self.reset()
                self.current_window.append(video_idx)
                # Add first depth map to depth dictionary - important for the first deformation
                # of the first frame
                self.depth_dict[video_idx] = depth
                self.is_kf[video_idx] = True # we map the first keyframe (after warmup)

                self.viewpoints[video_idx] = viewpoint
                depth = self.add_new_keyframe(video_idx, idx)
                print("min depth", depth.min(), "max depth", depth.max(), "median depth", np.median(depth))
                self.add_next_kf(
                    video_idx, viewpoint, depth_map=depth, init=True
                )
                start_mapping_time = time.process_time()
                self.initialize_map(video_idx, viewpoint)
                end_mapping_time = time.process_time()
                self.mapping_timings.append(end_mapping_time - start_mapping_time)
                init = False
                self.pipe.send("continue")
                continue

            # check if to add current frame as keyframe and map it, otherwise, continue tracking and deform
            # the map only once we have a keyframe to be mapped.

            # we need to render from the current pose to obtain the "n_touched" variable
            # which is used during keyframe selection

            # TODO should I extend to all cams? Now approximation using midpose
            R, t, theta, rho = viewpoint.get_virtual_extrinsics()
            virtual_cam = viewpoint.n_virtual_cams//2
            render_pkg = render_virtual(
                viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam],rho = rho[virtual_cam] 
            )
            start_tr_time = time.process_time()
            render_pkgs = self.tracking(video_idx, viewpoint)
            end_tr_time = time.process_time()
            self.tracking_timings.append(end_tr_time - start_tr_time)

            #take last pkgs
            #render_pkg = render_pkgs[-1]

            # compute median depth which is used during keyframe selection to account for the
            # global scale ambiguity during rgb-only SLAM 
            self.median_depth = get_median_depth(render_pkg["depth"], render_pkg["opacity"])


            # keyframe selection
            last_keyframe_idx = self.current_window[0]
            
            curr_visibility = (render_pkg["n_touched"] > 0).long()
            create_kf = self.is_keyframe(
                video_idx,
                last_keyframe_idx,
                curr_visibility,
                self.occ_aware_visibility,
            )
            if len(self.current_window) < self.window_size:
                # When we have not filled up the keyframe window size
                # we rely on just the covisibility thresholding, not the 
                # translation thresholds.
                union = torch.logical_or(
                    curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                ).count_nonzero()
                intersection = torch.logical_and(
                    curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                ).count_nonzero()
                point_ratio = intersection / union
                create_kf = (
                    point_ratio < self.config["mapping"]["Training"]["kf_overlap"]
                )
            
            if create_kf:
                self.current_window, removed = self.add_to_window(
                    video_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                    self.current_window,
                )
                self.is_kf[video_idx] = True
            else:
                self.is_kf[video_idx] = False
                self.pipe.send("continue")
                continue

            last_idx = self.keyframe_idxs[-1]
            w2c_temps = []
            depth_temps = []
            
            print(self.video_idxs, self.keyframe_idxs)
            
            for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):
                # need to update depth_dict even if the last idx since this is important
                # for the first deformation of the keyframe
                _, _, depth_gtd, _, _ = self.frame_reader[frame_idx]
                #depth_gt_numpy = depth_gtd.cpu().numpy()
                mono_depth = load_mono_depth(frame_idx, self.save_dir).to(self.device)
                # depth_temp, w2c_temp = self.get_w2c_and_depth(keyframe_idx, frame_idx, mono_depth, depth_gt_numpy, init=not init)
                depth_temp, w2c_temp, invalid = self.get_w2c_and_depth(keyframe_idx, frame_idx, mono_depth, init=False)
                w2c_temps.append(w2c_temp)
                depth_temps.append(depth_temp)
                if keyframe_idx not in self.depth_dict and self.is_kf[keyframe_idx]:
                    self.depth_dict[keyframe_idx] = depth_temp

                # No need to move the latest pose and depth
                if frame_idx != last_idx:
                    # Update tracking parameters
                    R, t, _, _ = self.cameras[keyframe_idx].get_mid_extrinsic()
                    w2c_old = torch.cat((R.squeeze(0), t.unsqueeze(-1)), dim=1)
                    w2c_old = torch.cat((w2c_old, torch.tensor([[0, 0, 0, 1]], device="cuda")), dim=0)

                    for knot in range(self.cameras[keyframe_idx].num_control_knots):
                        self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3], knot)

                    # Update depth for viewpoint
                    self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()

                    if keyframe_idx in self.viewpoints:
                        # Update tracking parameters
                        for knot in range(self.viewpoints[keyframe_idx].num_control_knots):
                            self.viewpoints[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3], knot)
                        self.viewpoints[keyframe_idx].depth = depth_temp.cpu().numpy()

                    # Update mapping parameters
                    if self.move_points and self.is_kf[keyframe_idx]:
                        if invalid:
                            # if the frame was invalid, we don't update the depth old and just do a rigid correction for this frame
                            self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics, method="rigid")
                        else:
                            self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics)
                            self.depth_dict[keyframe_idx] = depth_temp # line does not matter since it is the last deformation anyway
             
            for idx, (keyframe_idx, frame_idx) in enumerate(zip(self.keyframe_idxs, self.video_idxs)):
                if idx > 0:
                    prev_keyframe_idx = self.keyframe_idxs[idx - 1]
                    prev_frame_idx = self.video_idxs[idx - 1]
                else:
                    prev_keyframe_idx = None
                    prev_frame_idx = None

                # Find next keyframe and its frame index
                if idx < len(self.keyframe_idxs) - 1:
                    next_keyframe_idx = self.keyframe_idxs[idx + 1]
                    next_frame_idx = self.video_idxs[idx + 1]
                else:
                    next_keyframe_idx = None
                    next_frame_idx = None
                
                # Print current keyframe and neighboring indices
                print(f"\nProcessing keyframe_idx: {keyframe_idx} (frame_idx: {frame_idx})")
                if prev_keyframe_idx is not None:
                    print(f"Previous keyframe index (prev_keyframe_idx): {prev_keyframe_idx} (prev_frame_idx: {prev_frame_idx})")
                else:
                    print("Previous keyframe index (prev_keyframe_idx): None")
                if next_keyframe_idx is not None:
                    print(f"Next keyframe index (next_keyframe_idx): {next_keyframe_idx} (next_frame_idx: {next_frame_idx})")
                else:
                    print("Next keyframe index (next_keyframe_idx): None")
                
                R, t, _, _ = self.cameras[frame_idx].get_mid_extrinsic()
                mid_q_cur = matrix_to_quaternion(R)

                # Handle previous frame
                if prev_keyframe_idx is not None:
                    R_prev, t_prev, _, _ = self.cameras[prev_frame_idx].get_mid_extrinsic()
                    mid_q_prev = matrix_to_quaternion(R_prev)
                    delta_prev = keyframe_idx - prev_keyframe_idx
                    fraction_start = (keyframe_idx - 0.5 + self.cameras[frame_idx].prev_gap - prev_keyframe_idx) / delta_prev
                    #fraction_start = (keyframe_idx - self.cameras[frame_idx].exposure_time_left - prev_keyframe_idx) / delta_prev


                    # Print details for previous frame interpolation
                    print(f"delta_prev: {delta_prev}")
                    print(f"fraction_start: {fraction_start.item()}")
                    #print(f"prev_gap: {self.cameras[frame_idx].prev_gap.item()}")

                    t_expected_start = torch.lerp(t_prev, t, fraction_start)
                    q_expected_start = slerp(torch.tensor(fraction_start), mid_q_prev, mid_q_cur)
                    R_start = quaternion_to_matrix(q_expected_start)

                    with torch.no_grad():
                        self.cameras[frame_idx].update_RT(R_start, t_expected_start, 0)

                # Handle next frame
                if next_keyframe_idx is not None:
                    R_next, t_next, _, _ = self.cameras[next_frame_idx].get_mid_extrinsic()
                    mid_q_next = matrix_to_quaternion(R_next)
                    delta_next = next_keyframe_idx - keyframe_idx
                    fraction_end = (0.5 + self.cameras[frame_idx].next_gap)  / delta_next
                    #fraction_end = (self.cameras[frame_idx].exposure_time_right) / delta_next

                    # Print details for next frame interpolation
                    print(f"delta_next: {delta_next}")
                    print(f"fraction_end: {fraction_end.item()}")
                    #print(f"next_gap: {self.cameras[frame_idx].next_gap.item()}")

                    t_expected_end = torch.lerp(t, t_next, fraction_end)
                    q_expected_end = slerp(torch.tensor(fraction_end), mid_q_cur, mid_q_next)
                    R_end = quaternion_to_matrix(q_expected_end)

                    with torch.no_grad():
                        self.cameras[frame_idx].update_RT(R_end, t_expected_end, self.cameras[frame_idx].num_control_knots - 1)
                
                if self.cameras[frame_idx].num_control_knots > 2 and next_keyframe_idx is not None and prev_keyframe_idx is not None:
                    fraction_mid_start = 0.33
                    fraction_mid_end = 0.66

                    t_second = torch.lerp(t_expected_start, t_expected_end, fraction_mid_start)
                    q_second = slerp(torch.tensor(fraction_mid_start), q_expected_start, q_expected_end)
                    R_second = quaternion_to_matrix(q_second)

                    t_third = torch.lerp(t_expected_start, t_expected_end, fraction_mid_end)
                    q_third = slerp(torch.tensor(fraction_mid_end), q_expected_start, q_expected_end)
                    R_third = quaternion_to_matrix(q_third)

                    with torch.no_grad():
                        self.cameras[frame_idx].update_RT(R_second, t_second, 1)
                        self.cameras[frame_idx].update_RT(R_third, t_third, 2)
                
            # Do mapping
            # self.viewpoints contains the subset of self.cameras where we did mapping
            self.viewpoints[video_idx] = viewpoint
            depth = self.add_new_keyframe(video_idx, idx)
            #normalize depth to 0-1
            #depth = (depth - depth.min()) / (depth.max() - depth.min())
            print("min depth", depth.min(), "max depth", depth.max(), "median depth", np.median(depth))
            self.add_next_kf(video_idx, viewpoint, depth_map=depth, init=False) # set init to True for debugging

            self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

            opt_params = []
            frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]
            iter_per_kf = self.mapping_itr_num

            for cam_idx in range(len(self.current_window)):
                if self.current_window[cam_idx] == 0:
                    # Do not add GT frame pose for optimization
                    continue
                viewpoint = self.viewpoints[self.current_window[cam_idx]]
                if not self.gt_camera and self.config["mapping"]["BA"]:
                    if cam_idx < frames_to_optimize:
                        for knot in range(viewpoint.num_control_knots):
                            opt_params.append(
                                {
                                    "params": [viewpoint.T_i_rot_delta[knot]],
                                    "lr": self.config["mapping"]["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}_{}".format(viewpoint.uid, knot),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.T_i_trans_delta[knot]],
                                    "lr": self.config["mapping"]["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}_{}".format(viewpoint.uid, knot),
                                }
                            )
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_a],
                        "lr": 0.01,
                        "name": "exposure_a_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_b],
                        "lr": 0.01,
                        "name": "exposure_b_{}".format(viewpoint.uid),
                    }
                )
            self.keyframe_optimizers = torch.optim.Adam(opt_params)
            mapping_time_start = time.process_time()
            self.map(self.current_window, iters=iter_per_kf)
            #self.map(self.current_window, prune=True)
            mapping_time_end = time.process_time()
            self.mapping_timings.append(mapping_time_end - mapping_time_start)

            if(len(self.cameras) > 2):
                ATE, traj_est_aligned, global_scale = eval_ate(
                        self.cameras,
                        self.video_idxs,
                        self.save_dir,
                        0,
                        final=False,
                        monocular=True,
                        dataset_name="replicaglorieslam",
                    )
            
            '''
            try:
                self.video.save_video(f"{self.save_dir}/video.npz")
                ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj",
                    "kf_traj",self.frame_reader, None,self.printer)
            except Exception as e:
                self.printer.print(e,FontColor.ERROR)
            
            
            rendering_result = eval_rendering(
                    self,
                    self.save_dir,
                    iteration="before_refine",
                    monocular=True,
                    mesh=self.config["meshing"]["mesh_before_final_ba"],
                    traj_est_aligned=None,
                    global_scale=global_scale,
                    scene=self.config['scene'],
                    eval_mesh=True if self.config['dataset'] == 'replica' else False,
                    gt_mesh_path=self.config['meshing']['gt_mesh_path']
                )
            '''
            self.pipe.send("continue")


    def tracking_est_gap(self, viewpoint_prev, viewpoint, viewpoint_next, video_idx):
        
        print(video_idx)
        print("Memory allocated before tracking", torch.cuda.memory_allocated('cuda')/1000/1000)

        opt_params = []

        # List of all viewpoints
        viewpoints = [viewpoint_prev, viewpoint, viewpoint_next]

        for vp in viewpoints:
            if vp == None:
                continue

            for i in range(vp.num_control_knots):
                vp.R_i = vp.R_i.detach()
                vp.t_i = vp.t_i.detach()
                vp.T_i_rot_delta[i] = vp.T_i_rot_delta[i].detach().requires_grad_()
                vp.T_i_trans_delta[i] = vp.T_i_trans_delta[i].detach().requires_grad_()
                opt_params.append(
                    {
                        "params": [vp.T_i_rot_delta[i]],
                        "lr": self.config['mapping']["Training"]["lr"]["cam_rot_delta"],
                        "name": "rot_{}_{}".format(i, vp.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [vp.T_i_trans_delta[i]],
                        "lr": self.config['mapping']["Training"]["lr"]["cam_trans_delta"],
                        "name": "trans_{}_{}".format(i, vp.uid),
                    }
                )

            opt_params.append(
                {
                    "params": [vp.exposure_a],
                    "lr": 0.01,
                    "name": "exposure_a_{}".format(vp.uid),
                }
            )
            opt_params.append(
                {
                    "params": [vp.exposure_b],
                    "lr": 0.01,
                    "name": "exposure_b_{}".format(vp.uid),
                }
            )
            # Add prev_gap and next_gap parameters only for the current viewpoint
            if vp == viewpoint:
                
                opt_params.append(
                    {
                        "params": [vp.prev_gap],
                        "lr": 0.01,
                        "name": "prev_gap_{}".format(vp.uid),
                    }
                )
                
                opt_params.append(
                    {
                        "params": [vp.next_gap],
                        "lr": 0.01,
                        "name": "next_gap_{}".format(vp.uid),
                    }
                )
            
        video_averages_tensor = []
        #video_virtuals_tensor = [ [] for i in range(viewpoint.n_virtual_cams)]
        pose_optimizer = torch.optim.Adam(opt_params)

        gt_image = viewpoint.original_image

        #measure diff between knot gt poses

        tracking_video = [[], [], []]

        for tracking_itr in range(self.tracking_itr_num):
            #print("Tracking iteration: ", tracking_itr, "/", self.tracking_itr_num)
            loss_tracking = 0
                
            # Lists to collect data from all viewpoints
            images_tensor_list = []
            depths_tensor_list = []
            opacities_tensor_list = []
            avg_images_list = []
            avg_opacities_list = []
            gt_images_list = []

            render_pkg_ret_list = []

            for idx, vp in enumerate(viewpoints):
                if vp == None:
                    continue
                images_tensor = torch.empty(
                    (vp.n_virtual_cams, 3, vp.image_height, vp.image_width),
                    device="cuda:0"
                )
                depths_tensor = torch.empty(
                    (vp.n_virtual_cams, 1, vp.image_height, vp.image_width),
                    device="cuda:0"
                )
                opacities_tensor = torch.empty(
                    (vp.n_virtual_cams, 1, vp.image_height, vp.image_width),
                    device="cuda:0"
                )

                render_pkg_ret = []
                R, t, theta, rho = vp.get_virtual_extrinsics()

                for virtual_cam in range(vp.n_virtual_cams):
                    render_pkg = render_virtual(
                        vp, self.gaussians, self.pipeline_params, self.background,
                        R=R[virtual_cam], t=t[virtual_cam],
                        theta=theta[virtual_cam], rho=rho[virtual_cam]
                    )

                    image, depth, opacity = (
                        render_pkg["render"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                    )

                    images_tensor[virtual_cam] = image
                    depths_tensor[virtual_cam] = depth
                    opacities_tensor[virtual_cam] = opacity

                    render_pkg_ret.append(render_pkg)

                    # Clean up to save memory
                    del render_pkg, image, opacity
                    torch.cuda.empty_cache()

                # Compute averages
                avg_image = images_tensor.mean(0)
                avg_opacity = opacities_tensor.mean(0)

                # Collect data
                images_tensor_list.append(images_tensor)
                depths_tensor_list.append(depths_tensor)
                opacities_tensor_list.append(opacities_tensor)
                avg_images_list.append(avg_image)
                avg_opacities_list.append(avg_opacity)
                gt_images_list.append(vp.original_image)
                render_pkg_ret_list.extend(render_pkg_ret)

                # Clean up per-viewpoint tensors
                del images_tensor, depths_tensor, opacities_tensor


                if self.render_videos:
                    with torch.no_grad():
                        gt_image = vp.original_image  # Assuming this is the ground truth image
                        video_frame = (torch.clamp(torch.cat((avg_image, gt_image), dim=2).cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                        tracking_video[idx].append(video_frame)

            # Aggregate all virtual cameras' data
            # Note: If you need to concatenate along the batch dimension
            # images_tensor_all = torch.cat(images_tensor_list, dim=0)

            pose_optimizer.zero_grad()

            # Compute the loss using BAD_tracking_loss_gap
            loss_tracking += BAD_tracking_loss_gap(
                self.config,
                avg_images_list,
                gt_images_list,
                images_tensor_list,
                opacities_tensor_list,
                viewpoints,
                vp_prev=viewpoint_prev,
                vp_cur=viewpoint,
                vp_next=viewpoint_next
            )

            loss_tracking.backward()
            pose_optimizer.step()

            # Update poses for all viewpoints
            with torch.no_grad():
                converged = True
                for vp in viewpoints:
                    if vp == None:
                        continue
                    for i in range(vp.num_control_knots):
                        converged = update_pose_knot(vp, i) and converged

            if converged:
                print("Tracking converged...")

                with torch.no_grad():
                    for vp in viewpoints:
                        if vp == None:
                            continue
                        # Reset delta transformations
                        for i in range(vp.num_control_knots):
                            vp.T_i_rot_delta[i].data.fill_(0)
                            vp.T_i_trans_delta[i].data.fill_(0)
                break  # Exit the tracking loop if converged

        # Optional: Render and save videos
        if self.render_videos:
            print("Creating videos")
            for idx, video_frames in enumerate(tracking_video):
                render_video(
                    os.path.join(self.save_dir, f"tracking_gap_{idx}_video_idx_{video_idx}.mp4"),
                    video_frames, 5
                )

        #print("prev_gap", viewpoint.prev_gap.item(), "next_gap", viewpoint.next_gap.item())
        # Clean up
        del avg_images_list, avg_opacities_list
        return render_pkg_ret_list
