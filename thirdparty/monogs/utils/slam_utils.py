# Copyright 2024 The MonoGS Authors.

# Licensed under the License issued by the MonoGS Authors
# available here: https://github.com/muskie82/MonoGS/blob/main/LICENSE.md

import torch
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
import os
import cv2
from thirdparty.monogs.utils.pose_utils import get_new_RT, slerp
from thirdparty.monogs.utils.rotation_conv import quaternion_angle_difference, quaternion_to_matrix, matrix_to_quaternion, quaternion_multiply, quaternion_invert, quaternion_angle_difference_dot
import torch.nn as nn
import torchvision.transforms.functional as TF
from thirdparty.monogs.utils.pose_utils import slerp, SO3_exp

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


# Not used, but kept for reference
def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


# Not used, but kept for reference
def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    loss = 0
    if config["Training"]["ssim_loss"]:
        ssim_loss = 1.0 - ssim(image, gt_image)
        
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    if config["Training"]["ssim_loss"]:
        hyperparameter = config["opt_params"]["lambda_dssim"]
        loss += (1.0 - hyperparameter) * l1_rgb + hyperparameter * ssim_loss
    else:
        loss += l1_rgb

    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * loss.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

def BAD_mapping_loss(config, image, gt_image, images, depths, viewpoint, seen=True, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    
    alpha = config["mapping"]["Training"]["alpha"] if "alpha" in config["mapping"]["Training"] else 0.95
    lambda_dssim = config["mapping"]["opt_params"]["lambda_dssim"]
    lambda_total_variation = 0.0 #config["opt_params"]["lambda_total_variation"]
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["mapping"]["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    
    save_dir = config['data']['output'] + '/' + config['scene']
    opacity_path = os.path.join(save_dir,f"opacity_image_replica_mapping_{viewpoint.uid}.jpg")
   
    l1 = (torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)).mean()


    loss_edges = torch.tensor(0.0).to(image.device)

    loss = (1.0 - lambda_dssim) * (
        l1
    ) + lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_edges #+ total_variation_loss
    #print("l1: ", l1.item(), "loss_edges", loss_edges.item())
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth = depths.mean(dim=0)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    
    return alpha * loss.mean() + (1 - alpha) * l1_depth.mean()

def BAD_tracking_loss(config, image, gt_image, images, opacities, viewpoint, prev, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    
    lambda_dssim = config['mapping']["opt_params"]["lambda_dssim"]
    lambda_total_variation = config['mapping']["opt_params"]["lambda_total_variation"]
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config['mapping']["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    
    l1 = (opacities.mean(0) * torch.abs(image_ab * rgb_pixel_mask - gt_image * rgb_pixel_mask)).mean()

    lambda_rot_smooth = config["mapping"]["opt_params"]["lambda_rot_smooth"]    
    lambda_trans_smooth = config["mapping"]["opt_params"]["lambda_trans_smooth"]

    q_cur_0, t_cur_0 = get_new_RT(viewpoint, 0)
    q_cur_1, t_cur_1  = get_new_RT(viewpoint, 1)

    if prev.uid != viewpoint.uid:

        with torch.no_grad():
            q_prev_0, t_prev_0 = get_new_RT(prev, 0)  # Start of previous frame
            q_prev_1, t_prev_1 = get_new_RT(prev, 1)  # End of previous frame

        rot_dir_loss = lambda_rot_smooth * torch.norm( quaternion_angle_difference(q_cur_0, q_cur_1) ** 2  + quaternion_angle_difference(q_prev_1, q_cur_0) ** 2)
        trans_dir_loss = lambda_trans_smooth * ( torch.norm(t_cur_1 - t_cur_0) ** 2 + torch.norm(t_cur_0 - t_prev_1) ** 2 )

    else:
        rot_dir_loss = 0.0
        trans_dir_loss = 0.0

    save_dir = config['data']['output'] + '/' + config['scene']

    loss = l1 + trans_dir_loss  + rot_dir_loss  #+ total_variation_loss
    #print(f"Loss: {loss.shape}")
    return loss

def render_video(path, frames, framerate, frames_idx = None):
    if len(frames) == 0:
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, framerate, (width, height))

    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame.numpy().astype('uint8'), cv2.COLOR_BGR2RGB)
        if frames_idx is not None:
            cv2.putText(frame, f"Frame: {frames_idx[i]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"Iteration: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        video.write(frame)
    video.release()

def plot_tensor(tensor, path):
    import torchvision.transforms.functional as TF

    tensor = tensor.float()
    tensor = TF.to_pil_image(tensor)

    tensor.save(path)

def BAD_tracking_loss_gap(config, avg_images_list, gt_images_list, images_tensor_list, opacities_tensor_list, viewpoints, vp_prev, vp_cur, vp_next, initialization=False):
    # Initialize total loss
    total_loss = 0.0

    viewpoints = [viewpoint for viewpoint in viewpoints if viewpoint is not None]

    # Loop over each viewpoint
    for idx, vp in enumerate(viewpoints):
        # Retrieve the average image and ground truth image for this viewpoint
        avg_image = avg_images_list[idx]
        gt_image = gt_images_list[idx]
        opacities = opacities_tensor_list[idx]

        if initialization:
            image_ab = avg_image
        else:
            # Apply exposure adjustment
            image_ab = torch.exp(vp.exposure_a) * avg_image + vp.exposure_b

        # Retrieve configuration parameters
        _, h, w = gt_image.shape
        mask_shape = (1, h, w)
        rgb_boundary_threshold = config['mapping']["Training"]["rgb_boundary_threshold"]

        # Compute the RGB pixel mask
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
        rgb_pixel_mask = rgb_pixel_mask * vp.grad_mask

        # Compute L1 loss
        l1 = (opacities.mean(0) * torch.abs(image_ab * rgb_pixel_mask - gt_image * rgb_pixel_mask)).mean()

        # Accumulate the loss
        total_loss += l1

    # Initialize regularization losses
    trans_dir_loss = 0.0
    rot_dir_loss = 0.0

    lambda_trans_smooth = 0.001
    lambda_rot_smooth = 0.001

    # Regularization between previous and current viewpoints
    if vp_prev is not None:
        # Get mid rotations and translations
        R_prev_mid, t_prev_mid, _, _ = vp_prev.get_mid_extrinsic()
        R_cur_mid, t_cur_mid, _, _ = vp_cur.get_mid_extrinsic()

        mid_q_prev = matrix_to_quaternion(R_prev_mid)
        mid_q_cur = matrix_to_quaternion(R_cur_mid)

        # Fraction for interpolation between prev and cur
        fraction_prev = 0.5 + vp_cur.prev_gap

        # Expected start translation and rotation
        t_expected_start = torch.lerp(t_prev_mid, t_cur_mid, fraction_prev)
        q_expected_start = slerp(torch.tensor(fraction_prev, device=mid_q_prev.device), mid_q_prev, mid_q_cur)

        # Get actual start rotation and translation of current viewpoint
        q_cur_0, t_cur_0 = get_new_RT(vp_cur, 0)

        # Compute translation and rotation differences
        trans_dir_loss_prev = torch.norm(t_cur_0 - t_expected_start) ** 2
        rot_dir_loss_prev = torch.norm(quaternion_angle_difference(q_cur_0, q_expected_start)) ** 2

        # Add weighted losses to total
        trans_dir_loss += lambda_trans_smooth * trans_dir_loss_prev
        rot_dir_loss += lambda_rot_smooth * rot_dir_loss_prev

    # Regularization between current and next viewpoints
    if vp_next is not None:
        # Get mid rotations and translations
        R_cur_mid, t_cur_mid, _, _ = vp_cur.get_mid_extrinsic()
        R_next_mid, t_next_mid, _, _ = vp_next.get_mid_extrinsic()

        mid_q_cur = matrix_to_quaternion(R_cur_mid)
        mid_q_next = matrix_to_quaternion(R_next_mid)

        # Fraction for interpolation between cur and next
        fraction_next = 0.5 + vp_cur.next_gap

        # Expected end translation and rotation
        t_expected_end = torch.lerp(t_cur_mid, t_next_mid, fraction_next)
        q_expected_end = slerp(torch.tensor(fraction_next, device=mid_q_cur.device), mid_q_cur, mid_q_next)

        # Get actual end rotation and translation of current viewpoint
        q_cur_1, t_cur_1 = get_new_RT(vp_cur, 1)

        # Compute translation and rotation differences
        trans_dir_loss_next = torch.norm(t_cur_1 - t_expected_end) ** 2
        rot_dir_loss_next = torch.norm(quaternion_angle_difference(q_cur_1, q_expected_end)) ** 2

        # Add weighted losses to total
        trans_dir_loss += lambda_trans_smooth * trans_dir_loss_next
        rot_dir_loss += lambda_rot_smooth * rot_dir_loss_next

    # Add regularization losses to total loss
    total_loss += trans_dir_loss + rot_dir_loss

    # Optionally, print loss components for debugging
    #print(f"Viewpoint {vp_cur.uid} - L1 Loss: {l1.item()}, Trans Loss: {trans_dir_loss.item()}, Rot Loss: {rot_dir_loss.item()}")


    # Return the total loss across all viewpoints
    return total_loss