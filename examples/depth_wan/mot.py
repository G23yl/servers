import os
from copy import deepcopy
from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file
from tqdm import tqdm
import numpy as np

from diffsynth import ModelManager
from diffsynth.models.wan_video_image_encoder import WanImageEncoder
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.prompters import WanPrompter
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from diffsynth.models.utils import load_state_dict
from examples.depth_wan.depth_dit import WanModel as DepthWanModel
from examples.depth_wan.video_dit import WanModel as VideoWanModel

AdaModulationType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float32).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class VideoModule(nn.Module):
    def __init__(self, video_model: VideoWanModel):
        super().__init__()
        self.model = video_model

    def prepare_input(
        self,
        noisy_video_latent: torch.Tensor,
        y: torch.Tensor,
        context: torch.Tensor,
        clip_feat: torch.Tensor,
    ):
        context = self.model.text_embedding(context)
        if self.model.has_image_input:
            x = torch.cat([noisy_video_latent, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.model.img_emb(clip_feat)
            context = torch.cat([clip_embdding, context], dim=1)
        x, grid_size = self.model.patchify(x)
        f, h, w = grid_size
        freqs = (
            torch.cat(
                [
                    self.model.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.model.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.model.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, freqs, grid_size

    def get_time_embedding(self, t: torch.Tensor, seq_len: int):
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)

        bt = t.size(0)
        t_flat = t.flatten()

        # [B, seq_len, dim]
        t_embedding = self.model.time_embedding(
            sinusoidal_embedding_1d(self.model.freq_dim, t_flat).unflatten(
                0, (bt, seq_len)
            )
        )
        # [B, seq_len, 6, dim]
        t_mod = self.model.time_projection(t_embedding).unflatten(
            2, (6, self.model.dim)
        )

        return t_embedding, t_mod

    def compute_adaln_modulation(
        self, video_t_mod: torch.Tensor, layer_idx: int
    ) -> AdaModulationType:
        """
        计算6个modulation参数
        """
        video_layer = self.model.blocks[layer_idx]
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp - [B, seq_len, 1, dim]
        modulation = (
            video_layer.modulation.unsqueeze(0).to(
                dtype=video_t_mod.dtype, device=video_t_mod.device
            )
            + video_t_mod
        ).chunk(6, dim=2)
        return modulation

    def process_cross_attn(
        self,
        video_latent: torch.Tensor,
        context: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_layer = self.model.blocks[layer_idx]
        ca_out, video_cross_attn_map = video_layer.cross_attn(
            video_layer.norm3(video_latent), context
        )
        return video_latent + ca_out, video_cross_attn_map

    def process_ffn(
        self,
        video_latent: torch.Tensor,
        adaln_modulation: AdaModulationType,
        layer_idx: int,
    ):
        video_layer = self.model.blocks[layer_idx]
        input_x = modulate(
            video_layer.norm2(video_latent),
            adaln_modulation[3].squeeze(2),
            adaln_modulation[4].squeeze(2),
        )
        ffn_out = video_layer.ffn(input_x)
        x = video_latent + ffn_out * adaln_modulation[5].squeeze(2)
        return x

    def apply_output_head(self, video_latent: torch.Tensor, t: torch.Tensor, grid_size):
        x = self.model.head(video_latent, t)
        x = self.model.unpatchify(x, grid_size)
        return x

    def process_joint_self_attn(
        self,
        video_latent: torch.Tensor,
        depth_latent: torch.Tensor,
        video_adaln_modulation: AdaModulationType,
        depth_adaln_modulation: AdaModulationType,
        depth_layer: torch.nn.Module,
        video_freqs: torch.Tensor,
        depth_freqs: torch.Tensor,
        layer_idx: int,
    ):
        video_layer = self.model.blocks[layer_idx]

        # pre-norm modulation
        norm_video_latent = modulate(
            video_layer.norm1(video_latent),
            video_adaln_modulation[0].squeeze(2),
            video_adaln_modulation[1].squeeze(2),
        )
        norm_depth_latent = modulate(
            depth_layer.norm1(depth_latent),
            depth_adaln_modulation[0].squeeze(2),
            depth_adaln_modulation[1].squeeze(2),
        )

        # prepare depth latent
        depth_q, depth_k, depth_v = (
            depth_layer.self_attn.norm_q(depth_layer.self_attn.q(norm_depth_latent)),
            depth_layer.self_attn.norm_k(depth_layer.self_attn.k(norm_depth_latent)),
            depth_layer.self_attn.v(norm_depth_latent),
        )
        depth_q = rope_apply(depth_q, depth_freqs, depth_layer.num_heads)
        depth_k = rope_apply(depth_k, depth_freqs, depth_layer.num_heads)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        video_out, depth_out_h = torch_checkpoint(
            create_custom_forward(video_layer.self_attn),
            norm_video_latent,
            video_freqs,
            depth_q,
            depth_k,
            depth_v,
            use_reentrant=False,
        )

        depth_out = depth_layer.self_attn.o(depth_out_h)
        video_latent = video_latent + video_out * video_adaln_modulation[2].squeeze(2)
        depth_latent = depth_latent + depth_out * depth_adaln_modulation[2].squeeze(2)

        return video_latent, depth_latent


class DepthModule(nn.Module):
    def __init__(self, model: DepthWanModel):
        super().__init__()
        self.model = model

    def prepare_input(
        self,
        noisy_depth_latent: torch.Tensor,
        y: torch.Tensor,
        context: torch.Tensor,
        clip_feat: torch.Tensor,
    ):
        context = self.model.text_embedding(context)
        if self.model.has_image_input:
            x = torch.cat([noisy_depth_latent, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.model.img_emb(clip_feat)
            context = torch.cat([clip_embdding, context], dim=1)
        x, grid_size = self.model.patchify(x)
        f, h, w = grid_size
        freqs = (
            torch.cat(
                [
                    self.model.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.model.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.model.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, freqs, grid_size

    def get_time_embedding(self, t: torch.Tensor, seq_len: int):
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)

        bt = t.size(0)
        t_flat = t.flatten()

        # [B, seq_len, dim]
        t_embedding = self.model.time_embedding(
            sinusoidal_embedding_1d(self.model.freq_dim, t_flat).unflatten(
                0, (bt, seq_len)
            )
        )
        # [B, seq_len, 6, dim]
        t_mod = self.model.time_projection(t_embedding).unflatten(
            2, (6, self.model.dim)
        )

        return t_embedding, t_mod

    def compute_adaln_modulation(
        self, depth_t_mod: torch.Tensor, layer_idx: int
    ) -> AdaModulationType:
        """
        计算6个modulation参数
        """
        depth_layer = self.model.blocks[layer_idx]
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        modulation = (
            depth_layer.modulation.unsqueeze(0).to(
                dtype=depth_t_mod.dtype, device=depth_t_mod.device
            )
            + depth_t_mod
        ).chunk(6, dim=2)
        return modulation

    def process_cross_attn(
        self,
        depth_latent: torch.Tensor,
        context: torch.Tensor,
        layer_idx: int,
        cross_attn_map: Optional[torch.Tensor] = None,
    ):
        depth_layer = self.model.blocks[layer_idx]
        ca_out = depth_layer.cross_attn(
            depth_layer.norm3(depth_latent), context, cross_attn_map
        )
        return depth_latent + ca_out

    def process_ffn(
        self,
        depth_latent: torch.Tensor,
        adaln_modulation: AdaModulationType,
        layer_idx: int,
    ):
        depth_layer = self.model.blocks[layer_idx]
        input_x = modulate(
            depth_layer.norm2(depth_latent),
            adaln_modulation[3].squeeze(2),
            adaln_modulation[4].squeeze(2),
        )
        ffn_out = depth_layer.ffn(input_x)
        x = depth_latent + ffn_out * adaln_modulation[5].squeeze(2)
        return x

    def apply_output_head(self, depth_latent: torch.Tensor, t: torch.Tensor, grid_size):
        x = self.model.head(depth_latent, t)
        x = self.model.unpatchify(x, grid_size)
        return x


class DepthWanMoT(nn.Module):
    def __init__(self, is_training_mode: bool):
        super().__init__()
        if not is_training_mode:
            # inference mode needs to initialize video_dit and depth_dit first
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
            self.video_dit = VideoWanModel(**config)
            self.depth_dit = DepthWanModel(**config)

    def load_pretrained_models(self, model_dir: str, weight_type, device="cpu"):
        self.dtype = weight_type
        self.device = device
        logger.info(f"Loading pretrained models from {model_dir}")
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=1000, shift=5, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(num_inference_steps=1000, training=True)
        # load utils
        self._load_utils(model_dir, weight_type)
        self.text_encoder.to(self.device)
        self.image_encoder.to(self.device)
        self.vae.to(self.device)
        # load dit backbone
        dit_state_dict = load_state_dict(f"{model_dir}/model.safetensors")
        video_state_dict_converter = VideoWanModel.state_dict_converter()
        depth_state_dict_converter = DepthWanModel.state_dict_converter()
        video_state_dict, video_extra_kwargs = video_state_dict_converter.from_civitai(
            dit_state_dict
        )
        depth_state_dict, depth_extra_kwargs = depth_state_dict_converter.from_civitai(
            dit_state_dict
        )
        self.video_dit = VideoWanModel(**video_extra_kwargs)
        self.depth_dit = DepthWanModel(**depth_extra_kwargs)
        self.video_dit.load_state_dict(video_state_dict, strict=False, assign=True)
        self.depth_dit.load_state_dict(depth_state_dict, strict=False, assign=True)
        self.video_dit.to(dtype=self.dtype, device=self.device)
        self.depth_dit.to(dtype=self.dtype, device=self.device)

        self.video_module = VideoModule(self.video_dit)
        self.depth_module = DepthModule(self.depth_dit)
        logger.info("Loaded video and depth backbone")

    def load_checkpoint(
        self,
        model_path: str,
        weight_type,
        device="cpu",
        utils_path: Optional[str] = None,
    ):
        """
        model_path(str): trainable video_dit and depth_dit
        utils_path(str): untrainable text_encoder, vae and image_encoder
        """
        self.dtype = weight_type
        self.device = device
        # load video_dit and depth_dit
        self._load_dit(model_path, weight_type, device)
        # TODO scheduler inference steps need to modify
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=1000, shift=5, sigma_min=0.0, extra_one_step=True
        )
        # load vae, text_encoder, image_encoder
        utils_path = utils_path or model_path
        self._load_utils(utils_path, weight_type)

    def _load_utils(self, utils_path: str, weight_type):
        manager_input_paths = [
            f"{utils_path}/Wan2.1_VAE.pth",
            f"{utils_path}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{utils_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ]
        model_manager = ModelManager(torch_dtype=weight_type, device="cpu")
        model_manager.load_models(manager_input_paths, torch_dtype=weight_type)

        # load utils: text_encoder vae image_encoder
        self.prompter = WanPrompter()
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.vae: WanVideoVAE = model_manager.fetch_model("wan_video_vae")
        self.image_encoder: WanImageEncoder = model_manager.fetch_model(
            "wan_video_image_encoder"
        )
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        logger.info("Loaded and freezed utils: text_encoder | vae | image_encoder")

    def _load_dit(self, model_path: str, weight_type, device="cpu"):
        ckpt_path = Path(model_path)
        if ckpt_path.is_dir():
            ckpt_files = [
                ckpt_path.joinpath("model.safetensors"),
                ckpt_path.with_name("pytorch_model_0.bin"),
            ]
            for f in ckpt_files:
                if f.exists():
                    state_dict = (
                        load_file(f)
                        if str(f).endswith(".safetensors")
                        else torch.load(f)
                    )
                    self.load_state_dict(state_dict)
                    logger.info(f"Loaded checkpoint from: {f}")
                    break
        else:
            state_dict = (
                load_file(ckpt_path)
                if str(ckpt_path).endswith(".safetensors")
                else torch.load(ckpt_path)
            )
            self.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint from: {ckpt_path}")
        self.video_dit.to(dtype=weight_type, device=device)
        self.video_dit.requires_grad_(False)
        self.depth_dit.to(dtype=weight_type, device=device)
        self.depth_dit.requires_grad_(False)
        self.video_module = VideoModule(self.video_dit)
        self.depth_module = DepthModule(self.depth_dit)

    def forward(
        self,
        prompt,
        first_image: torch.Tensor,
        first_depth: torch.Tensor,
        video: torch.Tensor,
        depth: torch.Tensor,
        return_dict: bool = False,
    ):
        return self.training_step(
            prompt, first_image, first_depth, video, depth, return_dict
        )

    def training_step(
        self,
        prompt,
        first_image: torch.Tensor,
        first_depth: torch.Tensor,
        video: torch.Tensor,
        depth: torch.Tensor,
        return_dict: bool = False,
    ):
        batch_size, channels, frames, H, W = video.shape
        prompt_emb = self.prompter.encode_prompt(prompt).to(
            dtype=self.dtype, device=self.device
        )
        with torch.no_grad():
            rgb_clip_feat = self.image_encoder.encode_image(
                first_image.transpose(1, 2)
            ).to(dtype=self.dtype, device=self.device)
            depth_clip_feat = self.image_encoder.encode_image(
                first_depth.transpose(1, 2)
            ).to(dtype=self.dtype, device=self.device)
        # prepare mask
        rgb_msk = torch.ones(batch_size, 1, frames, H // 8, W // 8, device=self.device)
        rgb_msk[:, :, 1:] = 0
        rgb_msk = torch.concat(
            [
                torch.repeat_interleave(rgb_msk[:, :, 0:1], repeats=4, dim=2),
                rgb_msk[:, :, 1:],
            ],
            dim=2,
        )
        rgb_msk = rgb_msk.view(batch_size, rgb_msk.shape[2] // 4, 4, H // 8, W // 8)
        # [B, 4, 13, h, w]
        rgb_msk = rgb_msk.transpose(1, 2)
        depth_msk = deepcopy(rgb_msk)

        rgb_vae_input = torch.concat(
            [
                first_image,
                torch.zeros(
                    batch_size,
                    channels,
                    frames - 1,
                    H,
                    W,
                    device=first_image.device,
                ),
            ],
            dim=2,
        ).to(dtype=self.dtype, device=self.device)
        with torch.no_grad():
            rgb_y = self.vae.encode(rgb_vae_input, device=self.device).to(
                dtype=self.dtype, device=self.device
            )
        # [B, 16+4, 13, h, w]
        rgb_y = torch.concat([rgb_msk, rgb_y], dim=1).to(
            dtype=self.dtype, device=self.device
        )

        depth_vae_input = torch.concat(
            [
                first_depth,
                torch.zeros(
                    batch_size,
                    channels,
                    frames - 1,
                    H,
                    W,
                    device=first_depth.device,
                ),
            ],
            dim=2,
        ).to(dtype=self.dtype, device=self.device)
        with torch.no_grad():
            depth_y = self.vae.encode(depth_vae_input, device=self.device).to(
                dtype=self.dtype, device=self.device
            )
        # [B, 16+4, 13, h, w]
        depth_y = torch.concat([depth_msk, depth_y], dim=1).to(
            dtype=self.dtype, device=self.device
        )

        with torch.no_grad():
            clean_video_latent = self.vae.encode(video, device=self.device).to(
                dtype=self.dtype, device=self.device
            )
            clean_depth_latent = self.vae.encode(depth, device=self.device).to(
                dtype=self.dtype, device=self.device
            )

        video_noise = torch.randn_like(clean_video_latent)
        depth_noise = torch.randn_like(clean_depth_latent)
        timestep_id = torch.randint(
            0, self.scheduler.num_train_timesteps, (batch_size,)
        )
        timestep = self.scheduler.timesteps[timestep_id].to(
            dtype=self.dtype, device=self.device
        )
        sigma = self.scheduler.sigmas[timestep_id].to(
            dtype=self.dtype, device=self.device
        )
        # prepare noisy latents
        noisy_video_latent = (1 - sigma) * clean_video_latent + sigma * video_noise
        noisy_depth_latent = (1 - sigma) * clean_depth_latent + sigma * depth_noise
        # video_latent: [B, S, C]
        video_latent, video_context, video_freqs, video_grid_size = (
            self.video_module.prepare_input(
                noisy_video_latent, rgb_y, prompt_emb, rgb_clip_feat
            )
        )

        depth_latent, depth_context, depth_freqs, depth_grid_size = (
            self.depth_module.prepare_input(
                noisy_depth_latent, depth_y, prompt_emb, depth_clip_feat
            )
        )
        # prepare target latents
        video_target = video_noise - clean_video_latent
        depth_target = depth_noise - clean_depth_latent

        # get time embedding
        video_t_embedding, video_t_mod = self.video_module.get_time_embedding(
            timestep, video_latent.shape[1]
        )
        depth_t_embedding, depth_t_mod = self.depth_module.get_time_embedding(
            timestep, depth_latent.shape[1]
        )

        # MoT
        num_layers = len(self.video_dit.blocks)
        for layer_idx in range(num_layers):
            # adaln modulation
            with torch.autograd.graph.save_on_cpu():
                video_adaln_modulation = torch_checkpoint(
                    self.video_module.compute_adaln_modulation,
                    video_t_mod,
                    layer_idx,
                    use_reentrant=False,
                )
            with torch.autograd.graph.save_on_cpu():
                depth_adaln_modulation = torch_checkpoint(
                    self.depth_module.compute_adaln_modulation,
                    depth_t_mod,
                    layer_idx,
                    use_reentrant=False,
                )

            # joint self_attention
            with torch.autograd.graph.save_on_cpu():
                video_latent, depth_latent = torch_checkpoint(
                    self.video_module.process_joint_self_attn,
                    video_latent,
                    depth_latent,
                    video_adaln_modulation,
                    depth_adaln_modulation,
                    self.depth_dit.blocks[layer_idx],
                    video_freqs,
                    depth_freqs,
                    layer_idx,
                    use_reentrant=False,
                )

            # cross attention
            with torch.autograd.graph.save_on_cpu():
                video_latent, cross_attn_map = torch_checkpoint(
                    self.video_module.process_cross_attn,
                    video_latent,
                    video_context,
                    layer_idx,
                    use_reentrant=False,
                )
            cross_attn_map = cross_attn_map.detach()
            with torch.autograd.graph.save_on_cpu():
                depth_latent = torch_checkpoint(
                    self.depth_module.process_cross_attn,
                    depth_latent,
                    depth_context,
                    layer_idx,
                    cross_attn_map,
                    use_reentrant=False,
                )

            # ffn
            with torch.autograd.graph.save_on_cpu():
                video_latent = torch_checkpoint(
                    self.video_module.process_ffn,
                    video_latent,
                    video_adaln_modulation,
                    layer_idx,
                    use_reentrant=False,
                )
            with torch.autograd.graph.save_on_cpu():
                depth_latent = torch_checkpoint(
                    self.depth_module.process_ffn,
                    depth_latent,
                    depth_adaln_modulation,
                    layer_idx,
                    use_reentrant=False,
                )

        # output head and loss
        video_out = self.video_module.apply_output_head(
            video_latent, video_t_embedding, video_grid_size
        )
        depth_out = self.depth_module.apply_output_head(
            depth_latent, depth_t_embedding, depth_grid_size
        )
        assert video_out.shape == video_target.shape
        assert depth_out.shape == depth_target.shape

        video_loss = F.mse_loss(
            video_out.float(), video_target.float(), reduction="mean"
        )
        depth_loss = F.mse_loss(
            depth_out.float(), depth_target.float(), reduction="mean"
        )
        tot_loss = video_loss + depth_loss
        if return_dict:
            return {
                "video_loss": video_loss,
                "depth_loss": depth_loss,
                "total_loss": tot_loss,
            }
        return video_loss, depth_loss, tot_loss

    def inference(
        self,
        num_inference_steps: int,
        prompt: str,
        first_image: torch.Tensor,
        first_depth: torch.Tensor,
        num_frames: int = 49,
        shift: float = 5.0,
        height: int = 544,
        width: int = 960,
    ):
        """
        first_image: [B, C, H, W]
        first_depth: [B, C, H, W]

        return:
            rgb_videos: [B, F, H, W, C] (0..255)
            depth_videos: [B, F, H, W, C] (0..255)
        """
        batch_size = first_image.shape[0]
        # load util modules to device
        self.text_encoder.to(self.device)
        self.image_encoder.to(self.device)
        self.vae.to(self.device)
        self.scheduler.set_timesteps(num_inference_steps, shift=shift)
        # [B, F, C, H, W]
        first_image = first_image.unsqueeze(1).to(device=self.device)
        first_depth = first_depth.unsqueeze(1).to(device=self.device)

        video_latent = torch.randn(
            batch_size,
            16,
            (num_frames - 1) // 4 + 1,
            height // 8,
            width // 8,
            dtype=self.dtype,
            device=self.device,
        )
        depth_latent = torch.randn(
            batch_size,
            16,
            (num_frames - 1) // 4 + 1,
            height // 8,
            width // 8,
            dtype=self.dtype,
            device=self.device,
        )
        prompt_emb = self.prompter.encode_prompt(prompt).to(
            dtype=self.dtype, device=self.device
        )
        rgb_clip_feat = self.image_encoder.encode_image(first_image).to(
            dtype=self.dtype, device=self.device
        )
        depth_clip_feat = self.image_encoder.encode_image(first_depth).to(
            dtype=self.dtype, device=self.device
        )
        # prepare mask
        rgb_msk = torch.ones(
            batch_size, 1, num_frames, height // 8, width // 8, device=self.device
        )
        rgb_msk[:, :, 1:] = 0
        rgb_msk = torch.concat(
            [
                torch.repeat_interleave(rgb_msk[:, :, 0:1], repeats=4, dim=2),
                rgb_msk[:, :, 1:],
            ],
            dim=2,
        )
        rgb_msk = rgb_msk.view(
            batch_size, rgb_msk.shape[2] // 4, 4, height // 8, width // 8
        )
        # [B, 4, 13, h, w]
        rgb_msk = rgb_msk.transpose(1, 2)
        depth_msk = deepcopy(rgb_msk)

        rgb_vae_input = torch.concat(
            [
                first_image.transpose(1, 2),  # [B, C, F, H, W]
                torch.zeros(
                    batch_size,
                    3,
                    num_frames - 1,
                    height,
                    width,
                    device=first_image.device,
                ),
            ],
            dim=2,
        ).to(dtype=self.dtype, device=self.device)
        rgb_y = self.vae.encode(rgb_vae_input, device=self.device).to(
            dtype=self.dtype, device=self.device
        )
        # [B, 16+4, 13, h, w]
        rgb_y = torch.concat([rgb_msk, rgb_y], dim=1).to(
            dtype=self.dtype, device=self.device
        )

        depth_vae_input = torch.concat(
            [
                first_depth.transpose(1, 2),
                torch.zeros(
                    batch_size,
                    3,
                    num_frames - 1,
                    height,
                    width,
                    device=first_depth.device,
                ),
            ],
            dim=2,
        ).to(dtype=self.dtype, device=self.device)
        depth_y = self.vae.encode(depth_vae_input, device=self.device).to(
            dtype=self.dtype, device=self.device
        )
        # [B, 16+4, 13, h, w]
        depth_y = torch.concat([depth_msk, depth_y], dim=1).to(
            dtype=self.dtype, device=self.device
        )

        for i in tqdm(range(num_inference_steps)):
            timestep = (
                self.scheduler.timesteps[i]
                .expand(batch_size)
                .to(dtype=self.dtype, device=self.device)
            )
            dt = (
                self.scheduler.sigmas[i + 1]
                if i + 1 < len(self.scheduler.timesteps)
                else 0
            ) - self.scheduler.sigmas[i]

            video_tokens, video_context, video_freqs, video_grid_size = (
                self.video_module.prepare_input(
                    video_latent, rgb_y, prompt_emb, rgb_clip_feat
                )
            )
            depth_tokens, depth_context, depth_freqs, depth_grid_size = (
                self.depth_module.prepare_input(
                    depth_latent, depth_y, prompt_emb, depth_clip_feat
                )
            )
            # get time embedding
            video_t_embedding, video_t_mod = self.video_module.get_time_embedding(
                timestep, video_tokens.shape[1]
            )
            depth_t_embedding, depth_t_mod = self.depth_module.get_time_embedding(
                timestep, depth_tokens.shape[1]
            )
            # MoT
            num_layers = len(self.video_dit.blocks)
            for layer_idx in range(num_layers):
                # adaln modulation
                video_adaln_modulation = self.video_module.compute_adaln_modulation(
                    video_t_mod, layer_idx
                )
                depth_adaln_modulation = self.depth_module.compute_adaln_modulation(
                    depth_t_mod, layer_idx
                )

                # joint self_attention
                video_tokens, depth_tokens = self.video_module.process_joint_self_attn(
                    video_tokens,
                    depth_tokens,
                    video_adaln_modulation,
                    depth_adaln_modulation,
                    self.depth_dit.blocks[layer_idx],
                    video_freqs,
                    depth_freqs,
                    layer_idx,
                )

                # cross attention
                video_tokens, cross_attn_map = self.video_module.process_cross_attn(
                    video_tokens, video_context, layer_idx
                )
                depth_tokens = self.depth_module.process_cross_attn(
                    depth_tokens, depth_context, layer_idx, cross_attn_map
                )

                # ffn
                video_tokens = self.video_module.process_ffn(
                    video_tokens, video_adaln_modulation, layer_idx
                )
                depth_tokens = self.depth_module.process_ffn(
                    depth_tokens, depth_adaln_modulation, layer_idx
                )

            # output head and loss
            video_v_pred = self.video_module.apply_output_head(
                video_tokens, video_t_embedding, video_grid_size
            )
            depth_v_pred = self.depth_module.apply_output_head(
                depth_tokens, depth_t_embedding, depth_grid_size
            )
            # step
            video_latent = video_latent + video_v_pred * dt
            depth_latent = depth_latent + depth_v_pred * dt
        # decode
        decoded_videos = self.vae.decode(video_latent, device=self.device, tiled=False)
        decoded_depths = self.vae.decode(depth_latent, device=self.device, tiled=False)

        decoded_videos = rearrange(decoded_videos, "B C F H W -> B F H W C")
        decoded_depths = rearrange(decoded_depths, "B C F H W -> B F H W C")
        decoded_videos = (
            ((decoded_videos / 2 + 0.5).clamp(0, 1) * 255.0).clamp(0, 255).float()
        )
        decoded_depths = (
            ((decoded_depths / 2 + 0.5).clamp(0, 1) * 255.0).clamp(0, 255).float()
        )
        decoded_videos = [
            video.cpu().numpy().astype(np.uint8) for video in decoded_videos
        ]
        decoded_depths = [
            video.cpu().numpy().astype(np.uint8) for video in decoded_depths
        ]
        return decoded_videos, decoded_depths
