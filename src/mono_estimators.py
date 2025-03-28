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

import torch
from torchvision import transforms
import torch.nn.functional as F
import cv2
import numpy as np

from thirdparty.monogs.utils.slam_utils import plot_tensor
from torchvision.transforms import Compose
from src.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

def get_mono_depth_estimator(cfg):
    device = cfg["device"]
    depth_model = cfg["mono_prior"]["depth"]
    depth_pretrained = cfg["mono_prior"]["depth_pretrained"]
    if depth_model == "omnidata":
        model = get_omnidata_model(depth_pretrained, device, 1)
    elif depth_model == "anydepth_v2":
        model = get_anydepth_model(depth_pretrained, device)
    else:
        # If use other mono depth estimator as prior, load it here
        raise NotImplementedError 
    return model

def get_omnidata_model(pretrained_path, device, num_channels):
    from thirdparty.mono_priors.omnidata.modules.midas.dpt_depth import DPTDepthModel
    model = DPTDepthModel(backbone='vitb_rn50_384',num_channels=num_channels)
    checkpoint = torch.load(pretrained_path)
    
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)

    return model.to(device).eval()

def get_anydepth_model(pretrained_path, device):
    from src.depth_anything_v2.dpt import DepthAnythingV2
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vitl'
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(pretrained_path, map_location=device))
    return depth_anything.to(device).eval()

@torch.no_grad()
def predict_mono_depth(model,idx,input,cfg,device):
    '''
    input: tensor (1,3,H,W)
    '''
    depth_model = cfg["mono_prior"]["depth"]
    output_dir = f"{cfg['data']['output']}/{cfg['scene']}"
    if depth_model == "omnidata":
        # s = cfg["cam"]["H_out"]
        # image_size = (s,s)
        image_size = (512,512)
        input_size = input.shape[-2:]
        trans_totensor = transforms.Compose([transforms.Resize(image_size),
                                            transforms.Normalize(mean=0.5, std=0.5)])
        img_tensor = trans_totensor(input).to(device)
        output = model(img_tensor).clamp(min=0, max=1)
        output = F.interpolate(output.unsqueeze(0), input_size, mode='bicubic').squeeze(0)
        output = output.clamp(0,1).squeeze() #[H,W]
    elif depth_model == "anydepth_v2":        
        input_size = input.shape[-2:]
        image_size = 518
        trans_totensor = transforms.Compose([
            transforms.Resize(
                size=(image_size, image_size),
                interpolation=cv2.INTER_CUBIC,
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = trans_totensor(input).to(device)
        output = model(img_tensor)
        output = F.interpolate(output[:, None], input_size, mode="bilinear", align_corners=True)[0, 0]

        output = (1 / output)
        #normalize the depth
        output = (output - output.min()) / (output.max() - output.min())

    else:
        # If use other mono depth estimator as prior, predict the mono depth here
        raise NotImplementedError
    
    output_path_np = f"{output_dir}/mono_priors/depths/{idx:05d}.npy"
    final_depth = output.detach().cpu().float().numpy()
    np.save(output_path_np, final_depth)

    return output