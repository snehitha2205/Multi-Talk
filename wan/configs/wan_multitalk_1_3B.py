# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan MultiTalk 1.3B ------------------------#

multitalk_1_3B = EasyDict(__name__='Config: Wan MultiTalk AI2V 1.3B')
multitalk_1_3B.update(wan_shared_cfg)
multitalk_1_3B.sample_neg_prompt = 'bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards'

# t5
multitalk_1_3B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'  # can be updated if smaller version exists
multitalk_1_3B.t5_tokenizer = 'google/umt5-xxl'

# clip
multitalk_1_3B.clip_model = 'clip_xlm_roberta_vit_h_14'  # update if different clip for 1.3B
multitalk_1_3B.clip_dtype = torch.float16
multitalk_1_3B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-base-1_3B.pth'  # your 1.3B clip checkpoint path
multitalk_1_3B.clip_tokenizer = 'xlm-roberta-large'

# vae
multitalk_1_3B.vae_checkpoint = 'Wan2.1_1.3B_VAE.pth'  # 1.3B model-specific VAE checkpoint
multitalk_1_3B.vae_stride = (4, 8, 8)

# transformer model architecture parameters (scaled down for 1.3B)
multitalk_1_3B.patch_size = (1, 2, 2)
multitalk_1_3B.dim = 1536
multitalk_1_3B.ffn_dim = 8960
multitalk_1_3B.freq_dim = 256
multitalk_1_3B.num_heads = 12
multitalk_1_3B.num_layers = 30
multitalk_1_3B.window_size = (-1, -1)
multitalk_1_3B.qk_norm = True
multitalk_1_3B.cross_attn_norm = True
multitalk_1_3B.eps = 1e-6
