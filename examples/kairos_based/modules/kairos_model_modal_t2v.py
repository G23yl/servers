from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.models.utils import Cache as fla_Cahce
from loguru import logger
from PIL import Image
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tqdm import tqdm

from examples.kairos_based.modules.dits import (
    KairosModalDiT,
    KairosVideoDiT,
    sinusoidal_embedding_1d,
)
from examples.kairos_based.modules.schedulers import FlowMatchScheduler
from examples.kairos_based.modules.text_encoders import QwenVLTextEncoder
from examples.kairos_based.modules.utils import load_state_dict
from examples.kairos_based.modules.vaes import WanVideoVAE


def create_custom_forward(module):
    def custom_forward(*inputs):
        return module(*inputs)

    return custom_forward


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    x = x * (1 + scale) + shift
    return x


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float16).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class VideoModule(nn.Module):
    def __init__(self, model: KairosVideoDiT):
        super().__init__()
        self.model = model

    def prepare_input(
        self,
        noisy_video_latent: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor | None = None,
        clip_feat: torch.Tensor | None = None,
    ):
        context = self.model.text_embedding(context)
        x = noisy_video_latent
        if y is not None and self.model.require_vae_embedding:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        if clip_feat is None and self.model.require_clip_embedding:
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

    def get_time_embedding(
        self, t: torch.Tensor, latent_f: int, latent_h: int, latent_w: int
    ):
        seq_len = (
            latent_f
            * latent_h
            * latent_w
            // (self.model.patch_size[1] * self.model.patch_size[2])
        )
        ## first frame is always clean so its timestep should be 0 and the rest keeps unchanged
        if self.model.seperated_timestep and self.model.fuse_vae_embedding_in_latents:
            t = torch.concat(
                [
                    torch.zeros(
                        (
                            1,
                            latent_h
                            * latent_w
                            // (self.model.patch_size[1] * self.model.patch_size[2]),
                        ),
                        dtype=t.dtype,
                        device=t.device,
                    ),
                    torch.ones(
                        (
                            latent_f - 1,
                            latent_h
                            * latent_w
                            // (self.model.patch_size[1] * self.model.patch_size[2]),
                        ),
                        dtype=t.dtype,
                        device=t.device,
                    )
                    * t,
                ]
            ).unsqueeze(0)
        else:
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

    def apply_output_head(self, video_latent: torch.Tensor, t: torch.Tensor, grid_size):
        x = self.model.head(video_latent, t)
        x = self.model.unpatchify(x, grid_size)
        return x

    def compute_adaln_modulation(self, video_t_mod: torch.Tensor, layer_idx: int):
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

    def process_cross_attn_ffn(
        self,
        video_latent: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        adaln_modulation,
        layer_idx: int,
    ):
        video_layer = self.model.blocks[layer_idx]
        ca_out = video_layer.cross_attn(
            video_layer.cross_attn_norm(video_latent), context, attn_mask=context_mask
        )
        video_latent = video_latent + ca_out
        input_x = modulate(
            video_layer.ffn_norm(video_latent),
            adaln_modulation[3].squeeze(2),
            adaln_modulation[4].squeeze(2),
        )
        ffn_out = video_layer.ffn(input_x)
        x = video_latent + ffn_out * adaln_modulation[5].squeeze(2)
        return x

    def process_joint_attention(
        self,
        video_latent: torch.Tensor,
        modal_latent: torch.Tensor,
        video_adaln_modulation,
        modal_adaln_modulation,
        modal_layer: torch.nn.Module,
        video_freqs: torch.Tensor,
        modal_freqs: torch.Tensor,
        layer_idx: int,
        grid_size,
    ):
        video_layer = self.model.blocks[layer_idx]
        (f, h, w) = grid_size
        L = h * w
        # pre-norm modulation
        norm_video_latent = modulate(
            video_layer.self_attn_norm(video_latent),
            video_adaln_modulation[0].squeeze(2),
            video_adaln_modulation[1].squeeze(2),
        )
        norm_modal_latent = modulate(
            modal_layer.self_attn_norm(modal_latent),
            modal_adaln_modulation[0].squeeze(2),
            modal_adaln_modulation[1].squeeze(2),
        )
        if video_layer.use_linear_attn:
            ## here not use MoT
            chunk_size = 40768 // 7
            ## video branch
            if norm_video_latent.shape[1] > chunk_size:
                cache = fla_Cahce.from_legacy_cache()
                outputs = []
                for start in range(0, norm_video_latent.shape[1], chunk_size):
                    x_chunk = norm_video_latent[:, start : start + chunk_size, :]
                    out_chunk, _, cache = video_layer.gated_delta(
                        x_chunk, past_key_values=cache, use_cache=True
                    )
                    outputs.append(out_chunk)
                video_out = torch.cat(outputs, dim=1)
            else:
                video_out, _, _ = video_layer.gated_delta(norm_video_latent)
            ## modal branch
            if norm_modal_latent.shape[1] > chunk_size:
                cache = fla_Cahce.from_legacy_cache()
                outputs = []
                for start in range(0, norm_modal_latent.shape[1], chunk_size):
                    x_chunk = norm_modal_latent[:, start : start + chunk_size, :]
                    out_chunk, _, cache = modal_layer.gated_delta(
                        x_chunk, past_key_values=cache, use_cache=True
                    )
                    outputs.append(out_chunk)
                modal_out = torch.cat(outputs, dim=1)
            else:
                modal_out, _, _ = modal_layer.gated_delta(norm_modal_latent)
        else:
            # MoT
            video_out, modal_out = video_layer.self_attn(
                norm_video_latent,
                norm_modal_latent,
                modal_layer,
                video_freqs,
                modal_freqs,
                f,
                L,
            )
        video_latent = video_latent + video_out * video_adaln_modulation[2].squeeze(2)
        modal_latent = modal_latent + modal_out * modal_adaln_modulation[2].squeeze(2)

        return video_latent, modal_latent


class ModalModule(nn.Module):
    def __init__(self, model: KairosModalDiT):
        super().__init__()
        self.model = model

    def prepare_input(
        self,
        noisy_modal_latent: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor | None = None,
        clip_feat: torch.Tensor | None = None,
    ):
        context = self.model.text_embedding(context)
        x = noisy_modal_latent
        if y is not None and self.model.require_vae_embedding:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        if clip_feat is None and self.model.require_clip_embedding:
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

    def get_time_embedding(
        self, t: torch.Tensor, latent_f: int, latent_h: int, latent_w: int
    ):
        seq_len = (
            latent_f
            * latent_h
            * latent_w
            // (self.model.patch_size[1] * self.model.patch_size[2])
        )
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

    def apply_output_head(self, modal_latent: torch.Tensor, t: torch.Tensor, grid_size):
        x = self.model.head(modal_latent, t)
        x = self.model.unpatchify(x, grid_size)
        return x

    def compute_adaln_modulation(self, modal_t_mod: torch.Tensor, layer_idx: int):
        """
        计算6个modulation参数
        """
        modal_layer = self.model.blocks[layer_idx]
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp - [B, seq_len, 1, dim]
        modulation = (
            modal_layer.modulation.unsqueeze(0).to(
                dtype=modal_t_mod.dtype, device=modal_t_mod.device
            )
            + modal_t_mod
        ).chunk(6, dim=2)
        return modulation

    def process_cross_attn_ffn(
        self,
        modal_latent: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        adaln_modulation,
        layer_idx: int,
    ):
        modal_layer = self.model.blocks[layer_idx]
        ca_out = modal_layer.cross_attn(
            modal_layer.cross_attn_norm(modal_latent), context, attn_mask=context_mask
        )
        modal_latent = modal_latent + ca_out
        input_x = modulate(
            modal_layer.ffn_norm(modal_latent),
            adaln_modulation[3].squeeze(2),
            adaln_modulation[4].squeeze(2),
        )
        ffn_out = modal_layer.ffn(input_x)
        x = modal_latent + ffn_out * adaln_modulation[5].squeeze(2)
        return x


class KairosMotModel(nn.Module):
    def __init__(
        self,
        device="cpu",
        torch_dtype=torch.bfloat16,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.config = {
            "dim": 2560,
            "in_dim": 16,
            "ffn_dim": 10240,
            "out_dim": 16,
            "text_dim": 3584,
            "freq_dim": 256,
            "eps": 1e-6,
            "patch_size": (1, 2, 2),
            "num_heads": 20,
            "num_layers": 32,
            "has_image_input": False,
            "seperated_timestep": True,
            "require_vae_embedding": False,
            "require_clip_embedding": False,
            "fuse_vae_embedding_in_latents": True,
            "dilated_lengths": [1, 1, 4, 1],
            "use_seq_parallel": False,
            "use_tp_in_getaeddeltanet": False,
            "use_tp_in_self_attn": False,
        }
        video_dit = KairosVideoDiT(**self.config)
        modal_dit = KairosModalDiT(**self.config)
        self.video_module = VideoModule(video_dit)
        self.modal_module = ModalModule(modal_dit)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device="cpu", torch_dtype=torch.bfloat16, **kwargs
    ):
        """
        Load model from a checkpoint file containing all trainable parameters.

        Args:
            checkpoint_path: Path to the checkpoint file (.pt, .pth, .bin, or .safetensors)
            device: Device to load the model on
            torch_dtype: Data type for model weights
            **kwargs: Additional arguments passed to model constructor

        Returns:
            KairosMotModel: Loaded model instance
        """
        # Create model instance
        model = cls(device=device, torch_dtype=torch_dtype, **kwargs)

        # Load checkpoint
        state_dict = load_state_dict(
            checkpoint_path, torch_dtype=torch_dtype, device=device
        )

        # Load state dict into model
        model.load_state_dict(state_dict, strict=True)
        model.to(device=device, dtype=torch_dtype)

        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        return model

    def from_pretrained(self, ckpt_path_manager: Dict[str, str]):
        state_dict = load_state_dict(ckpt_path_manager["dit"])
        self.video_module.model.load_state_dict(state_dict, strict=True)
        self.modal_module.model.load_state_dict(state_dict, strict=True)
        self.video_module.to(device=self.device, dtype=self.torch_dtype)
        self.modal_module.to(device=self.device, dtype=self.torch_dtype)
        logger.info(f"Loaded dit from: {ckpt_path_manager['dit']}")

        self.video_scheduler = FlowMatchScheduler(
            num_train_timesteps=1000,
            shift=5,
            sigma_min=0.0,
            extra_one_step=True,
            exponential_shift=True,
            exponential_shift_mu=1.609,
        )
        dynamic_shift_len = ((480 // 8 + 2 - 1) // 2) * ((640 // 8 + 2 - 1) // 2)
        self.video_scheduler.set_timesteps(
            num_inference_steps=1000,
            training=True,
            dynamic_shift_len=dynamic_shift_len,
            num_frames=(49 - 1) // 4 + 1,
        )
        self.modal_scheduler = deepcopy(self.video_scheduler)

    def _modal_fn(
        self,
        video_latent,
        modal_latent,
        video_prompt_emb,
        modal_prompt_emb,
        video_attn_mask,
        modal_attn_mask,
        video_timestep,
        modal_timestep,
        latent_video_shape: Tuple[int, int, int],
        latent_modal_shape: Tuple[int, int, int],
    ):
        # video_latent: [B, S, C]
        video_latent, video_context, video_freqs, video_grid_size = (
            self.video_module.prepare_input(video_latent, video_prompt_emb)
        )
        modal_latent, modal_context, modal_freqs, modal_grid_size = (
            self.modal_module.prepare_input(modal_latent, modal_prompt_emb)
        )
        # prepare time_embeddings
        video_t_embedding, video_t_mod = self.video_module.get_time_embedding(
            video_timestep, *latent_video_shape
        )
        modal_t_embedding, modal_t_mod = self.modal_module.get_time_embedding(
            modal_timestep, *latent_modal_shape
        )

        num_layers = len(self.video_module.model.blocks)
        for layer_idx in range(num_layers):
            video_modulation = self.video_module.compute_adaln_modulation(
                video_t_mod, layer_idx
            )
            modal_modulation = self.modal_module.compute_adaln_modulation(
                modal_t_mod, layer_idx
            )
            ## joint attention
            video_latent, modal_latent = self.video_module.process_joint_attention(
                video_latent,
                modal_latent,
                video_modulation,
                modal_modulation,
                self.modal_module.model.blocks[layer_idx],
                video_freqs,
                modal_freqs,
                layer_idx,
                video_grid_size,
            )
            ## cross_attention and ffn
            video_latent = self.video_module.process_cross_attn_ffn(
                video_latent,
                video_context,
                video_attn_mask,
                video_modulation,
                layer_idx,
            )
            modal_latent = self.modal_module.process_cross_attn_ffn(
                modal_latent,
                modal_context,
                modal_attn_mask,
                modal_modulation,
                layer_idx,
            )
        # output head
        video_out = self.video_module.apply_output_head(
            video_latent, video_t_embedding, video_grid_size
        )
        modal_out = self.modal_module.apply_output_head(
            modal_latent, modal_t_embedding, modal_grid_size
        )
        return video_out, modal_out

    @torch.no_grad()
    def inference(
        self,
        prompt: str,
        negative_prompt: str,
        modal_type: str,
        image: Image.Image,
        vae: WanVideoVAE,
        prompter: QwenVLTextEncoder,
        num_inference_steps: int = 50,
        num_frames: int = 49,
        shift: float = 5.0,
        cfg_scale: float = 5.0,
        height: int = 480,
        width: int = 640,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ):
        """
        Inference function
        Args:
            prompt: lowercase, no space and dot around
            modal_type: lowercase, currently only be depth or flow
        """
        # scheduler
        scheduler = FlowMatchScheduler(
            shift=shift,
            sigma_min=0.0,
            extra_one_step=True,
            exponential_shift=True,
            exponential_shift_mu=1.609,
        )
        ph, pw = (
            self.video_module.model.patch_size[1],
            self.video_module.model.patch_size[2],
        )
        dynamic_shift_len = ((height // vae.upsampling_factor + ph - 1) // ph) * (
            (width // vae.upsampling_factor + pw - 1) // pw
        )
        scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            training=False,
            dynamic_shift_len=dynamic_shift_len,
            num_frames=(num_frames - 1) // 4 + 1,
        )

        # generate noise latents
        batch_size = 1
        latent_frames = (num_frames - 1) // 4 + 1
        shape = (
            batch_size,
            vae.z_dim,
            latent_frames,
            height // vae.upsampling_factor,
            width // vae.upsampling_factor,
        )
        video_latent = torch.randn(shape, dtype=self.torch_dtype, device=self.device)
        modal_latent = torch.randn(shape, dtype=self.torch_dtype, device=self.device)

        # encode prompt
        video_prompt_emb, video_attn_mask = prompter.encode_prompt(
            prompt, images=image, positive=True, device=self.device
        )
        modal_prompt = f"[{modal_type.upper()}] {prompt}"
        modal_prompt_emb, modal_attn_mask = prompter.encode_prompt(
            modal_prompt, images=None, positive=True, device=self.device
        )
        video_prompt_emb = video_prompt_emb.to(
            dtype=self.torch_dtype, device=self.device
        )
        modal_prompt_emb = modal_prompt_emb.to(
            dtype=self.torch_dtype, device=self.device
        )
        negative_prompt_emb, negative_attn_mask = prompter.encode_prompt(
            negative_prompt, images=None, positive=False, device=self.device
        )

        # process image
        # [1, C, 1, H, W]
        image_tensor = (
            torch.from_numpy(
                (np.array(image.resize((width, height))) / 255.0 - 0.5) * 2
            )
            .permute(2, 0, 1)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(dtype=self.torch_dtype, device=self.device)
        )
        first_fused_latent = vae.encode(
            image_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video_latent[:, :, 0:1] = first_fused_latent

        latent_f_video, latent_h_video, latent_w_video = video_latent.shape[2:]
        latent_f_modal, latent_h_modal, latent_w_modal = modal_latent.shape[2:]

        for idx in tqdm(range(num_inference_steps)):
            timestep = (
                scheduler.timesteps[idx]
                .unsqueeze(0)
                .to(dtype=self.torch_dtype, device=self.device)
            )
            dt = (
                scheduler.sigmas[idx + 1] if idx + 1 < len(scheduler.timesteps) else 0
            ) - scheduler.sigmas[idx]

            video_pred, modal_pred = self._modal_fn(
                video_latent,
                modal_latent,
                video_prompt_emb,
                modal_prompt_emb,
                video_attn_mask,
                modal_attn_mask,
                timestep,
                timestep,
                (latent_f_video, latent_h_video, latent_w_video),
                (latent_f_modal, latent_h_modal, latent_w_modal),
            )
            if cfg_scale != 1.0:
                video_pred_nega, modal_pred_nega = self._modal_fn(
                    video_latent,
                    modal_latent,
                    negative_prompt_emb,
                    negative_prompt_emb,
                    negative_attn_mask,
                    negative_attn_mask,
                    timestep,
                    timestep,
                    (latent_f_video, latent_h_video, latent_w_video),
                    (latent_f_modal, latent_h_modal, latent_w_modal),
                )
                video_pred = cfg_scale * video_pred + (1 - cfg_scale) * video_pred_nega
                modal_pred = cfg_scale * modal_pred + (1 - cfg_scale) * modal_pred_nega
            video_latent = video_latent + video_pred * dt
            modal_latent = modal_latent + modal_pred * dt
            video_latent[:, :, 0:1] = first_fused_latent
        # decoding
        decoded_video = vae.decode(
            video_latent,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        decoded_modal = vae.decode(
            modal_latent,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]

        decoded_video = rearrange(decoded_video, "C F H W -> F H W C")
        decoded_modal = rearrange(decoded_modal, "C F H W -> F H W C")
        decoded_video = (
            ((decoded_video / 2 + 0.5).clamp(0, 1) * 255.0).clamp(0, 255).float()
        )
        decoded_modal = (
            ((decoded_modal / 2 + 0.5).clamp(0, 1) * 255.0).clamp(0, 255).float()
        )
        decoded_video = [
            video.cpu().numpy().astype(np.uint8) for video in decoded_video
        ]
        decoded_modal = [
            video.cpu().numpy().astype(np.uint8) for video in decoded_modal
        ]
        return decoded_video, decoded_modal

    def forward(
        self,
        video_prompt,
        modal_prompt,
        first_image: torch.Tensor,
        video: torch.Tensor,
        modal: torch.Tensor,
        prompter: QwenVLTextEncoder,
        vae: WanVideoVAE,
        return_dict: bool = False,
    ):
        return self.training_step1(
            video_prompt,
            modal_prompt,
            first_image,
            video,
            modal,
            prompter,
            vae,
            return_dict,
        )

    def training_step(
        self,
        video_prompt,
        modal_prompt,
        first_image: torch.Tensor,
        video: torch.Tensor,
        modal: torch.Tensor,
        prompter: QwenVLTextEncoder,
        vae: WanVideoVAE,
        return_dict: bool = False,
    ):
        """
        Args:
            video_prompt: List[str]
            modal_prompt: List[str]
            first_image: Tensor [B, H, W, C] 0..255
            video: Tensor [B, C, F, H, W] -1..1
            modal: Tensor [B, C, F, H, W] -1..1
        """
        batch_size = video.shape[0]
        ## get video and modal prompt_emb
        ## Note: images must be Image type, so need change Tensor to List[Image]
        first_image_list = [
            Image.fromarray(image.cpu().numpy().astype(np.uint8))
            for image in first_image
        ]
        video_prompt_emb, video_attn_mask = prompter.encode_prompt(
            video_prompt, images=first_image_list, positive=True, device=self.device
        )
        modal_prompt_emb, modal_attn_mask = prompter.encode_prompt(
            modal_prompt, images=None, positive=True, device=self.device
        )
        video_prompt_emb = video_prompt_emb.to(
            dtype=self.torch_dtype, device=self.device
        )
        modal_prompt_emb = modal_prompt_emb.to(
            dtype=self.torch_dtype, device=self.device
        )

        with torch.no_grad():
            clean_video_latent = vae.encode(video, device=self.device).to(
                dtype=self.torch_dtype, device=self.device
            )
            clean_modal_latent = vae.encode(modal, device=self.device).to(
                dtype=self.torch_dtype, device=self.device
            )
            first_image_latent = deepcopy(clean_video_latent[:, :, 0:1])

        video_noise = torch.randn_like(
            clean_video_latent, device=clean_video_latent.device
        )
        modal_noise = torch.randn_like(
            clean_modal_latent, device=clean_video_latent.device
        )
        timestep_id = torch.randint(
            0, self.video_scheduler.num_train_timesteps, (batch_size,)
        )
        timestep_id_modal = torch.randint(
            0, self.modal_scheduler.num_train_timesteps, (batch_size,)
        )
        timestep = self.video_scheduler.timesteps[timestep_id].to(
            dtype=self.torch_dtype, device=self.device
        )
        timestep_modal = self.modal_scheduler.timesteps[timestep_id_modal].to(
            dtype=self.torch_dtype, device=self.device
        )
        sigma = self.video_scheduler.sigmas[timestep_id].to(
            dtype=self.torch_dtype, device=self.device
        )
        sigma_modal = self.modal_scheduler.sigmas[timestep_id_modal].to(
            dtype=self.torch_dtype, device=self.device
        )
        noisy_video_latent = (1 - sigma) * clean_video_latent + sigma * video_noise
        noisy_modal_latent = (
            1 - sigma_modal
        ) * clean_modal_latent + sigma_modal * modal_noise
        # replace the noisy first frame with the clean one for only video branch
        noisy_video_latent[:, :, 0:1] = first_image_latent

        latent_f_video, latent_h_video, latent_w_video = noisy_video_latent.shape[2:]
        latent_f_modal, latent_h_modal, latent_w_modal = noisy_modal_latent.shape[2:]

        # video_latent: [B, S, C]
        video_latent, video_context, video_freqs, video_grid_size = (
            self.video_module.prepare_input(noisy_video_latent, video_prompt_emb)
        )
        modal_latent, modal_context, modal_freqs, modal_grid_size = (
            self.modal_module.prepare_input(noisy_modal_latent, modal_prompt_emb)
        )
        # training target
        video_target = video_noise - clean_video_latent
        modal_target = modal_noise - clean_modal_latent
        # prepare time_embeddings
        video_t_embedding, video_t_mod = self.video_module.get_time_embedding(
            timestep, latent_f_video, latent_h_video, latent_w_video
        )
        modal_t_embedding, modal_t_mod = self.modal_module.get_time_embedding(
            timestep_modal, latent_f_modal, latent_h_modal, latent_w_modal
        )

        num_layers = len(self.video_module.model.blocks)
        for layer_idx in range(num_layers):
            if self.use_gradient_checkpointing_offload:
                video_modulation = torch_checkpoint(
                    self.video_module.compute_adaln_modulation,
                    video_t_mod,
                    layer_idx,
                    use_reentrant=False,
                )
                modal_modulation = torch_checkpoint(
                    self.modal_module.compute_adaln_modulation,
                    modal_t_mod,
                    layer_idx,
                    use_reentrant=False,
                )
                with torch.autograd.graph.save_on_cpu():
                    ## joint attention
                    video_latent, modal_latent = torch_checkpoint(
                        self.video_module.process_joint_attention,
                        video_latent,
                        modal_latent,
                        video_modulation,
                        modal_modulation,
                        self.modal_module.model.blocks[layer_idx],
                        video_freqs,
                        modal_freqs,
                        layer_idx,
                        video_grid_size,
                        use_reentrant=False,
                    )
                    ## cross_attention and ffn
                    video_latent = torch_checkpoint(
                        self.video_module.process_cross_attn_ffn,
                        video_latent,
                        video_context,
                        video_attn_mask,
                        video_modulation,
                        layer_idx,
                        use_reentrant=False,
                    )
                    modal_latent = torch_checkpoint(
                        self.modal_module.process_cross_attn_ffn,
                        modal_latent,
                        modal_context,
                        modal_attn_mask,
                        modal_modulation,
                        layer_idx,
                        use_reentrant=False,
                    )
            elif self.use_gradient_checkpointing:
                video_modulation = torch_checkpoint(
                    self.video_module.compute_adaln_modulation,
                    video_t_mod,
                    layer_idx,
                    use_reentrant=False,
                )
                modal_modulation = torch_checkpoint(
                    self.modal_module.compute_adaln_modulation,
                    modal_t_mod,
                    layer_idx,
                    use_reentrant=False,
                )
                ## joint attention
                video_latent, modal_latent = torch_checkpoint(
                    self.video_module.process_joint_attention,
                    video_latent,
                    modal_latent,
                    video_modulation,
                    modal_modulation,
                    self.modal_module.model.blocks[layer_idx],
                    video_freqs,
                    modal_freqs,
                    layer_idx,
                    video_grid_size,
                    use_reentrant=False,
                )
                ## cross_attention and ffn
                video_latent = torch_checkpoint(
                    self.video_module.process_cross_attn_ffn,
                    video_latent,
                    video_context,
                    video_attn_mask,
                    video_modulation,
                    layer_idx,
                    use_reentrant=False,
                )
                modal_latent = torch_checkpoint(
                    self.modal_module.process_cross_attn_ffn,
                    modal_latent,
                    modal_context,
                    modal_attn_mask,
                    modal_modulation,
                    layer_idx,
                    use_reentrant=False,
                )
            else:
                video_modulation = self.video_module.compute_adaln_modulation(
                    video_t_mod, layer_idx
                )
                modal_modulation = self.modal_module.compute_adaln_modulation(
                    modal_t_mod, layer_idx
                )
                ## joint attention
                video_latent, modal_latent = self.video_module.process_joint_attention(
                    video_latent,
                    modal_latent,
                    video_modulation,
                    modal_modulation,
                    self.modal_module.model.blocks[layer_idx],
                    video_freqs,
                    modal_freqs,
                    layer_idx,
                    video_grid_size,
                )
                ## cross_attention and ffn
                video_latent = self.video_module.process_cross_attn_ffn(
                    video_latent,
                    video_context,
                    video_attn_mask,
                    video_modulation,
                    layer_idx,
                )
                modal_latent = self.modal_module.process_cross_attn_ffn(
                    modal_latent,
                    modal_context,
                    modal_attn_mask,
                    modal_modulation,
                    layer_idx,
                )
        # output head
        video_out = self.video_module.apply_output_head(
            video_latent, video_t_embedding, video_grid_size
        )
        modal_out = self.modal_module.apply_output_head(
            modal_latent, modal_t_embedding, modal_grid_size
        )

        assert video_out.shape == video_target.shape
        assert modal_out.shape == modal_target.shape

        video_loss = F.mse_loss(
            video_out.float(), video_target.float(), reduction="mean"
        )
        modal_loss = F.mse_loss(
            modal_out.float(), modal_target.float(), reduction="mean"
        )
        tot_loss = video_loss + modal_loss
        if return_dict:
            return {
                "video_loss": video_loss,
                "modal_loss": modal_loss,
                "total_loss": tot_loss,
            }
        return video_loss, modal_loss, tot_loss
    
    def training_step1(
        self,
        video_prompt,
        modal_prompt,
        first_image: torch.Tensor,
        video: torch.Tensor,
        modal: torch.Tensor,
        prompter: QwenVLTextEncoder,
        vae: WanVideoVAE,
        return_dict: bool = False,
    ):
        """
        Args:
            video_prompt: List[str]
            modal_prompt: List[str]
            first_image: Tensor [B, H, W, C] 0..255
            video: Tensor [B, C, F, H, W] -1..1
            modal: Tensor [B, C, F, H, W] -1..1
        """
        batch_size = video.shape[0]
        ## get video and modal prompt_emb
        ## Note: images must be Image type, so need change Tensor to List[Image]
        first_image_list = [
            Image.fromarray(image.cpu().numpy().astype(np.uint8))
            for image in first_image
        ]
        video_prompt_emb, video_attn_mask = prompter.encode_prompt(
            video_prompt, images=first_image_list, positive=True, device=self.device
        )
        modal_prompt_emb, modal_attn_mask = prompter.encode_prompt(
            modal_prompt, images=None, positive=True, device=self.device
        )
        video_prompt_emb = video_prompt_emb.to(
            dtype=self.torch_dtype, device=self.device
        )
        modal_prompt_emb = modal_prompt_emb.to(
            dtype=self.torch_dtype, device=self.device
        )

        with torch.no_grad():
            clean_video_latent = vae.encode(video, device=self.device).to(
                dtype=self.torch_dtype, device=self.device
            )
            clean_modal_latent = vae.encode(modal, device=self.device).to(
                dtype=self.torch_dtype, device=self.device
            )
            first_image_latent = deepcopy(clean_video_latent[:, :, 0:1])

        video_noise = torch.randn_like(
            clean_video_latent, device=clean_video_latent.device
        )
        modal_noise = torch.randn_like(
            clean_modal_latent, device=clean_video_latent.device
        )
        timestep_id = torch.randint(
            0, self.video_scheduler.num_train_timesteps, (batch_size,)
        )
        timestep_id_modal = torch.randint(
            0, self.modal_scheduler.num_train_timesteps, (batch_size,)
        )
        timestep = self.video_scheduler.timesteps[timestep_id].to(
            dtype=self.torch_dtype, device=self.device
        )
        timestep_modal = self.modal_scheduler.timesteps[timestep_id_modal].to(
            dtype=self.torch_dtype, device=self.device
        )
        sigma = self.video_scheduler.sigmas[timestep_id].to(
            dtype=self.torch_dtype, device=self.device
        )
        sigma_modal = self.modal_scheduler.sigmas[timestep_id_modal].to(
            dtype=self.torch_dtype, device=self.device
        )
        noisy_video_latent = (1 - sigma) * clean_video_latent + sigma * video_noise
        noisy_modal_latent = (
            1 - sigma_modal
        ) * clean_modal_latent + sigma_modal * modal_noise
        # replace the noisy first frame with the clean one for only video branch
        noisy_video_latent[:, :, 0:1] = first_image_latent

        latent_f_video, latent_h_video, latent_w_video = noisy_video_latent.shape[2:]
        latent_f_modal, latent_h_modal, latent_w_modal = noisy_modal_latent.shape[2:]

        # video_latent: [B, S, C]
        video_latent, video_context, video_freqs, video_grid_size = (
            self.video_module.prepare_input(noisy_video_latent, video_prompt_emb)
        )
        modal_latent, modal_context, modal_freqs, modal_grid_size = (
            self.modal_module.prepare_input(noisy_modal_latent, modal_prompt_emb)
        )
        # training target
        video_target = video_noise - clean_video_latent
        modal_target = modal_noise - clean_modal_latent
        # prepare time_embeddings
        video_t_embedding, video_t_mod = self.video_module.get_time_embedding(
            timestep, latent_f_video, latent_h_video, latent_w_video
        )
        modal_t_embedding, modal_t_mod = self.modal_module.get_time_embedding(
            timestep_modal, latent_f_modal, latent_h_modal, latent_w_modal
        )

        num_layers = len(self.video_module.model.blocks)
        interval = 2
        for layer_idx in range(num_layers):
            if layer_idx % interval == 0:
                if self.use_gradient_checkpointing_offload:
                    video_modulation = torch_checkpoint(
                        self.video_module.compute_adaln_modulation,
                        video_t_mod,
                        layer_idx,
                        use_reentrant=False,
                    )
                    modal_modulation = torch_checkpoint(
                        self.modal_module.compute_adaln_modulation,
                        modal_t_mod,
                        layer_idx,
                        use_reentrant=False,
                    )
                    with torch.autograd.graph.save_on_cpu():
                        ## joint attention
                        video_latent, modal_latent = torch_checkpoint(
                            self.video_module.process_joint_attention,
                            video_latent,
                            modal_latent,
                            video_modulation,
                            modal_modulation,
                            self.modal_module.model.blocks[layer_idx],
                            video_freqs,
                            modal_freqs,
                            layer_idx,
                            video_grid_size,
                            use_reentrant=False,
                        )
                        ## cross_attention and ffn
                        video_latent = torch_checkpoint(
                            self.video_module.process_cross_attn_ffn,
                            video_latent,
                            video_context,
                            video_attn_mask,
                            video_modulation,
                            layer_idx,
                            use_reentrant=False,
                        )
                        modal_latent = torch_checkpoint(
                            self.modal_module.process_cross_attn_ffn,
                            modal_latent,
                            modal_context,
                            modal_attn_mask,
                            modal_modulation,
                            layer_idx,
                            use_reentrant=False,
                        )
                elif self.use_gradient_checkpointing:
                    video_modulation = torch_checkpoint(
                        self.video_module.compute_adaln_modulation,
                        video_t_mod,
                        layer_idx,
                        use_reentrant=False,
                    )
                    modal_modulation = torch_checkpoint(
                        self.modal_module.compute_adaln_modulation,
                        modal_t_mod,
                        layer_idx,
                        use_reentrant=False,
                    )
                    ## joint attention
                    video_latent, modal_latent = torch_checkpoint(
                        self.video_module.process_joint_attention,
                        video_latent,
                        modal_latent,
                        video_modulation,
                        modal_modulation,
                        self.modal_module.model.blocks[layer_idx],
                        video_freqs,
                        modal_freqs,
                        layer_idx,
                        video_grid_size,
                        use_reentrant=False,
                    )
                    ## cross_attention and ffn
                    video_latent = torch_checkpoint(
                        self.video_module.process_cross_attn_ffn,
                        video_latent,
                        video_context,
                        video_attn_mask,
                        video_modulation,
                        layer_idx,
                        use_reentrant=False,
                    )
                    modal_latent = torch_checkpoint(
                        self.modal_module.process_cross_attn_ffn,
                        modal_latent,
                        modal_context,
                        modal_attn_mask,
                        modal_modulation,
                        layer_idx,
                        use_reentrant=False,
                    )
            else:
                video_modulation = self.video_module.compute_adaln_modulation(
                    video_t_mod, layer_idx
                )
                modal_modulation = self.modal_module.compute_adaln_modulation(
                    modal_t_mod, layer_idx
                )
                ## joint attention
                video_latent, modal_latent = self.video_module.process_joint_attention(
                    video_latent,
                    modal_latent,
                    video_modulation,
                    modal_modulation,
                    self.modal_module.model.blocks[layer_idx],
                    video_freqs,
                    modal_freqs,
                    layer_idx,
                    video_grid_size,
                )
                ## cross_attention and ffn
                video_latent = self.video_module.process_cross_attn_ffn(
                    video_latent,
                    video_context,
                    video_attn_mask,
                    video_modulation,
                    layer_idx,
                )
                modal_latent = self.modal_module.process_cross_attn_ffn(
                    modal_latent,
                    modal_context,
                    modal_attn_mask,
                    modal_modulation,
                    layer_idx,
                )
        # output head
        video_out = self.video_module.apply_output_head(
            video_latent, video_t_embedding, video_grid_size
        )
        modal_out = self.modal_module.apply_output_head(
            modal_latent, modal_t_embedding, modal_grid_size
        )

        assert video_out.shape == video_target.shape
        assert modal_out.shape == modal_target.shape

        video_loss = F.mse_loss(
            video_out.float(), video_target.float(), reduction="mean"
        )
        modal_loss = F.mse_loss(
            modal_out.float(), modal_target.float(), reduction="mean"
        )
        tot_loss = video_loss + modal_loss
        if return_dict:
            return {
                "video_loss": video_loss,
                "modal_loss": modal_loss,
                "total_loss": tot_loss,
            }
        return video_loss, modal_loss, tot_loss
