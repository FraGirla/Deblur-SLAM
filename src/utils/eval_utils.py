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

import json
import os

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import open3d as o3d
import trimesh

from thirdparty.gaussian_splatting.gaussian_renderer import render_virtual
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.gaussian_splatting.utils.system_utils import mkdir_p
from src.utils.datasets import load_mono_depth
from thirdparty.monogs.utils.slam_utils import render_video
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
import evo 
from thirdparty.monogs.utils.slam_utils import variance_of_laplacian
import traceback
from evaluate_3d_reconstruction import run_evaluation

def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    timestamps = []
    traj_ref = []
    traj_est = []
    for i in range(len(poses_gt)):
        val = poses_gt[i].sum()
        if np.isnan(val) or np.isinf(val):
            print(f'Nan or Inf found in gt poses, skipping {i}th pose!')
            continue
        traj_est.append(poses_est[i])
        traj_ref.append(poses_gt[i])
        timestamps.append(float(i))
    
    n_poses = len(traj_ref)

    from evo.core.trajectory import PoseTrajectory3D, PosePath3D
    # from evo.core import trajectory
    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    from evo.core import sync
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, global_scale = traj_est.align(traj_ref, correct_scale=monocular)

    # traj_ref = PosePath3D(poses_se3=poses_gt)
    
    # # TODO: In my monocular setting I should have correct_scale = True
    # traj_est_aligned, r_a, t_a, global_scale = trajectory.align_trajectory(
    #     traj_est, traj_ref, correct_scale=monocular, return_parameters=True
    # )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    print("RMSE ATE \[m]", ape_stat)

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}_{}.png".format(str(label), str(n_poses))), dpi=90)
    print("Saving ATE data to: ", os.path.join(plot_dir, "evo_2dplot_{}_{}.png".format(str(label), str(n_poses))))

    # in the event that some poses in the GT path were invalid, we will miss those poses
    # in the "traj_est" variable, but they are critical for the rendering and meshing 
    # evaluation. Therefore, use the computed alignment transformation and apply it to
    # the original list of estimated poses.
    from evo.core import lie_algebra as lie
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est.scale(global_scale)
    traj_est.transform(lie.se3(r_a, t_a))
    # Log("monogs rotation alignment: ", r_a, tag="Eval")
    # Log("monogs translation alignment: ", t_a, tag="Eval")

    return ape_stat, traj_est.poses_se3, global_scale


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False, dataset_name=None, printer=None):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose
    print("kj_ids: ", kf_ids)
    for kf_id in kf_ids:
        kf = frames[kf_id]

        R_i, t_i, _, _= kf.get_virtual_extrinsics(return_gradients=False)
        
        for cam in range(kf.n_virtual_cams):

            pose_est = np.linalg.inv(gen_pose_matrix(R_i[cam], t_i[cam]))
            if dataset_name is not None:
                R_i_gt, t_i_gt = kf.get_gt_virtual_extrinsics(realgt_pose=True)
                pose_gt = np.linalg.inv(gen_pose_matrix(R_i_gt[cam], t_i_gt[cam]))
            else:
                R_i_gt, t_i_gt = kf.get_gt_virtual_extrinsics(realgt_pose=False)
                pose_gt = np.linalg.inv(gen_pose_matrix(R_i_gt[cam], t_i_gt[cam]))

            trj_id.append(frames[kf_id].uid)
            trj_est.append(pose_est.tolist())
            trj_gt.append(pose_gt.tolist())

            trj_est_np.append(pose_est)
            trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    print("Evaluating ATE with {} poses".format(len(trj_gt)))
    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate, trj_est_aligned, global_scale = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    # wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    # Log("frame_idx: ", latest_frame_idx, "ate: ", ate)
    return ate, trj_est_aligned, global_scale


def eval_rendering(
    mapper,
    save_dir,
    iteration="after_refine",
    monocular=False,
    mesh=False,
    traj_est_aligned=None,
    global_scale=None,
    eval_mesh=True,
    scene=None,
    gt_mesh_path=None
):  
    dataset = mapper.frame_reader
    frames = mapper.cameras
    gaussians = mapper.gaussians
    background = mapper.background
    pipe = mapper.pipeline_params
    video_idxs = mapper.video_idxs

    mkdir_p(os.path.join(save_dir, iteration))

    keyframe_idxs = mapper.keyframe_idxs
    end_idx = len(frames) - 1

    img_pred, img_gt, saved_frame_idx = [], [], []
    
    psnr_array, ssim_array, lpips_array, depth_l1_array = [], [], [], []
    psnr_sharp_array, ssim_sharp_array, lpips_sharp_array, depth_l1_sharp_array = [], [], [], []
    psnr_mid_sharp_array, ssim_mid_sharp_array, lpips_mid_sharp_array = [], [], []

    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    if mesh:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    global_optimiz_video = []
    global_optimiz_video_sharp = []

    global_frame_idx = []

    variances = dict()

    for k, (kf_idx, video_idx) in enumerate(zip(keyframe_idxs, video_idxs)):

        saved_frame_idx.append(video_idx)
        frame = frames[video_idx]
       
        _, gt_image, gt_depth, _, gt_images = dataset[kf_idx]
        #gt_depth = gt_depth.cpu().numpy()
        gt_image = gt_image.squeeze().to("cuda:0")
        # retrieve mono depth
        mono_depth = load_mono_depth(kf_idx, save_dir).to("cuda:0")
        # retrieve sensor 
        sensor_depth, _, invalid = mapper.get_w2c_and_depth(video_idx, kf_idx, mono_depth, gt_depth, init=False)
        sensor_depth = sensor_depth.cpu()

        rgb = []
        depths = []

        #convert to black and white
        gt_image_bw = torch.mean(gt_image, dim=0, keepdim=True)

        variances[kf_idx], _ = variance_of_laplacian(gt_image_bw.unsqueeze(0))

        R, t, theta, rho = frame.get_virtual_extrinsics()

        for cam in range(frame.n_virtual_cams):
            rendering_pkg = render_virtual(frame, gaussians, pipe, background, R = R[cam], t = t[cam], theta = theta[cam], rho = rho[cam])
            rendering = rendering_pkg["render"].detach()
            depth = rendering_pkg["depth"].detach()
            
            image = rendering
            rgb.append(image)
            depths.append(depth)

            image = torch.clamp(image, 0.0, 1.0)
            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
            )
            video_frame = (torch.clamp(torch.cat((image, gt_image), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)

            #gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

            gt_sharp = gt_images[int((cam/frame.n_virtual_cams)*len(gt_images))].squeeze().to("cuda:0")
            #print("index:", int((cam/frame.n_virtual_cams)*len(gt_images)))
            mask = gt_sharp > 0

           
            video_frame_sharp = (torch.clamp(torch.cat((image, gt_sharp), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)

            psnr_score_sharp = psnr((image[mask]).unsqueeze(0), (gt_sharp[mask]).unsqueeze(0))
            ssim_score_sharp = ssim((image).unsqueeze(0), (gt_sharp).unsqueeze(0))
            lpips_score_sharp = cal_lpips((image).unsqueeze(0), (gt_sharp).unsqueeze(0))

            psnr_sharp_array.append(psnr_score_sharp.item())
            ssim_sharp_array.append(ssim_score_sharp.item())
            lpips_sharp_array.append(lpips_score_sharp.item())

            global_optimiz_video.append(video_frame)
            global_optimiz_video_sharp.append(video_frame_sharp)
            global_frame_idx.append(kf_idx)
            #print(f"Saving frame {kf_idx} cam {cam}")

        # Compute mid rendering of sharp images
        mid_index = len(rgb) // 2
        mid_index_gt = len(gt_images) // 2
        mid_rendering = rgb[mid_index]
        mid_gt_sharp = gt_images[mid_index_gt].squeeze().to("cuda:0")
        mask_mid = mid_gt_sharp > 0

        mid_rendering = torch.clamp(mid_rendering, 0.0, 1.0)

        psnr_score_mid_sharp = psnr((mid_rendering[mask_mid]).unsqueeze(0), (mid_gt_sharp[mask_mid]).unsqueeze(0))
        ssim_score_mid_sharp = ssim((mid_rendering).unsqueeze(0), (mid_gt_sharp).unsqueeze(0))
        lpips_score_mid_sharp = cal_lpips((mid_rendering).unsqueeze(0), (mid_gt_sharp).unsqueeze(0))

        psnr_mid_sharp_array.append(psnr_score_mid_sharp.item())
        ssim_mid_sharp_array.append(ssim_score_mid_sharp.item())
        lpips_mid_sharp_array.append(lpips_score_mid_sharp.item())

        rendering = torch.stack(rgb).mean(dim=0)
        depth = torch.stack(depths).mean(dim=0)

        gt = (gt_image.squeeze().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        # include optimized exposure compensation
        if k > 0: # first mapping frame is reference for exposure
            image = (torch.exp(frame.exposure_a.detach())) * rendering + frame.exposure_b.detach()
        else:
            image = rendering
        image = torch.clamp(image, 0.0, 1.0)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        #img_gt.append(gt)

        mask = gt_image > 0 

        #gt_depth = torch.tensor(gt_depth)
        depth = depth.detach().cpu()
        

        # compute depth errors
        depth_mask = (sensor_depth > 0) * (depth > 0) #* (gt_depth > 0)
        diff_depth_l1 = torch.abs(depth - sensor_depth)
        diff_depth_l1 = diff_depth_l1 * depth_mask
        depth_l1 = diff_depth_l1.sum() / depth_mask.sum()

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

        # Add plotting 2x3 grid here
        plot_dir = save_dir + "/plots_" + iteration
        plot_rgbd_silhouette(gt_image, sensor_depth, image, depth, diff_depth_l1,
                                 psnr_score.item(), depth_l1, plot_dir=plot_dir, idx='video_idx_' + str(video_idx) + "_kf_idx_" + str(kf_idx),
                                 diff_rgb=np.abs(gt - pred), sharp_psnr=psnr_score_sharp.item())

        # do volumetric TSDF fusion from which the mesh will be extracted later
        if mesh:
            # mask out the pixels where the GT mesh is non-existent. Do this with the gt depth mask
            #depth[gt_depth.unsqueeze(0) == 0] = 0
            depth_o3d = np.ascontiguousarray(depth.permute(1, 2, 0).numpy().astype(np.float32))
            depth_o3d = o3d.geometry.Image(depth_o3d)
            color_o3d = np.ascontiguousarray((np.clip(image.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)*255.0).astype(np.uint8))
            color_o3d = o3d.geometry.Image(color_o3d)

            w2c_o3d = np.linalg.inv(traj_est_aligned[k]) # convert from c2w to w2c
                    
            fx = frame.fx
            fy = frame.fy
            cx = frame.cx
            cy = frame.cy
            W =  depth.shape[-1]
            H = depth.shape[1]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=30,
                convert_rgb_to_intensity=False)
            # use gt pose for debugging
            # w2c_o3d = torch.linalg.inv(pose).cpu().numpy() @ dataset.w2c_first_pose
            volume.integrate(rgbd, intrinsic, w2c_o3d)
    save_dir_video = mapper.save_dir
    render_video(os.path.join(save_dir_video,"global_mapping_eval.mp4"), global_optimiz_video, 5, global_frame_idx)
    render_video(os.path.join(save_dir_video,"global_mapping_eval_sharp.mp4"), global_optimiz_video_sharp, 5, global_frame_idx)
    if mesh:
        # Mesh the final volumetric model
        mesh_out_file = os.path.join(save_dir, iteration, "mesh.ply")
        o3d_mesh = volume.extract_triangle_mesh()
        o3d_mesh = clean_mesh(o3d_mesh)
        o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
        print('Meshing finished.')

        # evaluate the mesh
        if eval_mesh:
            try:
                pred_ply = mesh_out_file.split('/')[-1]
                last_slash_index = mesh_out_file.rindex('/')
                path_to_pred_ply = mesh_out_file[:last_slash_index]
                gt_mesh = gt_mesh_path
                result_3d = run_evaluation(pred_ply, path_to_pred_ply, "mesh",
                                        distance_thresh=0.05, full_path_to_gt_ply=gt_mesh, icp_align=True)

                print(f"3D Mesh evaluation: {result_3d}")

            except Exception as e:
                traceback.print_exception(e)

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_psnr_sharp"] = float(np.mean(psnr_sharp_array))
    output["mean_ssim_sharp"] = float(np.mean(ssim_sharp_array))
    output["mean_lpips_sharp"] = float(np.mean(lpips_sharp_array))
    output["mean_psnr_mid_sharp"] = float(np.mean(psnr_mid_sharp_array))
    output["mean_ssim_mid_sharp"] = float(np.mean(ssim_mid_sharp_array))
    output["mean_lpips_mid_sharp"] = float(np.mean(lpips_mid_sharp_array))
    
    print(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, depth l1: {0}, depth l1 sensor: {0},',
        f'psnr sharp: {output["mean_psnr_sharp"]}, ssim sharp: {output["mean_ssim_sharp"]}, lpips sharp: {output["mean_lpips_sharp"]},',
        f'psnr mid sharp: {output["mean_psnr_mid_sharp"]}, ssim mid sharp: {output["mean_ssim_mid_sharp"]}, lpips mid sharp: {output["mean_lpips_mid_sharp"]}', 
    )
    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )

    # Create gif
    create_gif_from_directory(plot_dir, plot_dir + '/output.gif', online=True)

    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

def plot_rgbd_silhouette(color, depth, rastered_color, rastered_depth, diff_depth_l1,
                         psnr, depth_l1, plot_dir=None, idx=None, 
                         save_plot=True, diff_rgb=None, depth_max=5, opacities=None,
                         scales=None, sharp_psnr=None):

    os.makedirs(plot_dir, exist_ok=True)
    # Determine Plot Aspect Ratio
    aspect_ratio = color.shape[2] / color.shape[1]
    fig_height = 8
    fig_width = 14/1.55
    fig_width = fig_width * aspect_ratio
    # Plot the Ground Truth and Rasterized RGB & Depth, along with Diff Depth & Silhouette
    if opacities is not None or scales is not None:
        fig, axs = plt.subplots(2, 4, figsize=(fig_width, fig_height))
    else:
        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
    axs[0, 0].imshow(color.cpu().permute(1, 2, 0))
    axs[0, 0].set_title("Ground Truth RGB")
    axs[0, 1].imshow(depth, cmap='jet', vmin=0, vmax=depth_max)
    axs[0, 1].set_title("Input Depth")
    rastered_color = torch.clamp(rastered_color, 0, 1)
    axs[1, 0].imshow(rastered_color.detach().cpu().permute(1, 2, 0))
    if sharp_psnr is not None:
        axs[1, 0].set_title("Rasterized RGB, PSNR: {:.2f}".format(psnr) + ", Sharp PSNR: {:.2f}".format(sharp_psnr))
    else:
        axs[1, 0].set_title("Rasterized RGB, PSNR: {:.2f}".format(psnr))    
    axs[1, 1].imshow(rastered_depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=depth_max)
    axs[1, 1].set_title("Rasterized Depth, L1: {:.2f}".format(depth_l1))
    if diff_rgb is not None:
        axs[0, 2].imshow(diff_rgb, cmap='jet', vmin=0, vmax=diff_rgb.max())
        axs[0, 2].set_title("Diff RGB L1")
    diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)
    axs[1, 2].imshow(diff_depth_l1, cmap='jet', vmin=0, vmax=diff_depth_l1.max())
    axs[1, 2].set_title("Diff Depth L1")

    if opacities is not None:
        axs[0, 3].hist(opacities, bins=50, range=(0,1))
        axs[0, 3].set_title('Histogram of Opacities')
        axs[0, 3].set_xlabel('Opacity')
        axs[0, 3].set_ylabel('Frequency')
    if scales is not None:
        axs[1, 3].hist(scales, bins=50, range=(0, scales.max()))
        axs[1, 3].set_title('Histogram of Scales')
        axs[1, 3].set_xlabel('Scale')
        axs[1, 3].set_ylabel('Frequency')
        axs[1, 3].locator_params(axis='x', nbins=6)

    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[0, 2].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    axs[1, 2].axis('off')
    fig.suptitle("frame: " + str(idx), y=0.95, fontsize=16)
    fig.tight_layout()
    if save_plot:
        save_path = os.path.join(plot_dir, f"{idx}.png")
        plt.savefig(save_path, bbox_inches='tight')

    plt.close()


def create_gif_from_directory(directory_path, output_filename, duration=100, online=True):
    """
    Creates a GIF from all PNG images in a given directory.

    :param directory_path: Path to the directory containing PNG images.
    :param output_filename: Output filename for the GIF.
    :param duration: Duration of each frame in the GIF (in milliseconds).
    """

    from PIL import Image
    import re
    # Function to extract the number from the filename
    def extract_number(filename):
        # Pattern to find a number followed by '.png'
        match = re.search(r'(\d+)\.png$', filename)
        if match:
            return int(match.group(1))
        else:
            return None


    if online:
        # Get all PNG files in the directory
        image_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.png')]

        # Sort the files based on the number in the filename
        image_files.sort(key=extract_number)
    else:
        # Get all PNG files in the directory
        image_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.png')]

        # Sort the files based on the number in the filename
        image_files.sort()

    # Load images
    images = [Image.open(file) for file in image_files]

    # Convert images to the same mode and size for consistency
    images = [img.convert('RGBA') for img in images]
    base_size = images[0].size
    resized_images = [img.resize(base_size, Image.LANCZOS) for img in images]

    # Save as GIF
    resized_images[0].save(output_filename, save_all=True, append_images=resized_images[1:], optimize=False, duration=duration, loop=0)


def clean_mesh(mesh):
    mesh_tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(
        mesh.triangles), vertex_colors=np.asarray(mesh.vertex_colors))
    components = trimesh.graph.connected_components(
        edges=mesh_tri.edges_sorted)

    min_len = 100
    components_to_keep = [c for c in components if len(c) >= min_len]

    new_vertices = []
    new_faces = []
    new_colors = []
    vertex_count = 0
    for component in components_to_keep:
        vertices = mesh_tri.vertices[component]
        colors = mesh_tri.visual.vertex_colors[component]

        # Create a mapping from old vertex indices to new vertex indices
        index_mapping = {old_idx: vertex_count +
                         new_idx for new_idx, old_idx in enumerate(component)}
        vertex_count += len(vertices)

        # Select faces that are part of the current connected component and update vertex indices
        faces_in_component = mesh_tri.faces[np.any(
            np.isin(mesh_tri.faces, component), axis=1)]
        reindexed_faces = np.vectorize(index_mapping.get)(faces_in_component)

        new_vertices.extend(vertices)
        new_faces.extend(reindexed_faces)
        new_colors.extend(colors)

    cleaned_mesh_tri = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    cleaned_mesh_tri.visual.vertex_colors = np.array(new_colors)

    cleaned_mesh_tri.remove_degenerate_faces()
    cleaned_mesh_tri.remove_duplicate_faces()
    print(
        f'Mesh cleaning (before/after), vertices: {len(mesh_tri.vertices)}/{len(cleaned_mesh_tri.vertices)}, faces: {len(mesh_tri.faces)}/{len(cleaned_mesh_tri.faces)}')

    cleaned_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cleaned_mesh_tri.vertices),
        o3d.utility.Vector3iVector(cleaned_mesh_tri.faces)
    )
    vertex_colors = np.asarray(cleaned_mesh_tri.visual.vertex_colors)[
        :, :3] / 255.0
    cleaned_mesh.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors.astype(np.float64))

    return cleaned_mesh