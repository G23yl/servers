import torch
import os
from peft import PeftModel, PeftConfig
from PIL.Image import Image
from PIL import Image as ImageModule
from tqdm import tqdm
from typing import Optional, Tuple
from einops import rearrange
import copy
import cv2
import numpy as np

from examples.depth_wan.transformer_depth import DepthAIDNs, TrajectoryFPNHead

from diffusers.video_processor import VideoProcessor

from diffsynth.pipelines.base import BasePipeline
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from diffsynth.prompters import WanPrompter
from diffsynth.models.wan_video_text_encoder import WanTextEncoder
from diffsynth.models.wan_video_image_encoder import WanImageEncoder
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.models import ModelManager


class WanDepthPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.rgb_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.depth_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.rgb_dit: WanModel = None
        self.depth_dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.aidns_modules: DepthAIDNs = None
        self.video_processor = VideoProcessor()
        self.model_names = [
            "text_encoder",
            "rgb_dit",
            "depth_dit",
            "vae",
            "aidns_modules",
        ]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.target_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanDepthPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        self.rgb_dit = model_manager.fetch_model("wan_video_dit")
        self.depth_dit = copy.deepcopy(self.rgb_dit)

    def load_lora(self, model_path: str):
        rgb_transformer_path = os.path.join(model_path, "rgb_transformer_lora")
        depth_transformer_path = os.path.join(model_path, "depth_transformer_lora")
        aidns_path = os.path.join(model_path, "aidns_modules")

        print(f"load rgb_dit lora ckpt from: {rgb_transformer_path}")
        self.rgb_dit = PeftModel.from_pretrained(self.rgb_dit, rgb_transformer_path)

        print(f"load depth_dit lora ckpt from: {depth_transformer_path}")
        self.depth_dit = PeftModel.from_pretrained(
            self.depth_dit, depth_transformer_path
        )

        print(f"load aidns_module ckpt from: {aidns_path}")
        self.aidns_modules = DepthAIDNs.from_pretrained(aidns_path)

    def load_no_lora(self, model_path: str):
        rgb_transformer_path = os.path.join(model_path, "rgb_transformer", "model.pth")
        depth_transformer_path = os.path.join(model_path, "depth_transformer", "model.pth")
        aidns_path = os.path.join(model_path, "aidns_modules")

        print(f"load rgb_dit ckpt from: {rgb_transformer_path}")
        
        self.rgb_dit.load_state_dict(torch.load(rgb_transformer_path))

        print(f"load depth_dit ckpt from: {depth_transformer_path}")
        self.depth_dit.load_state_dict(torch.load(depth_transformer_path))

        print(f"load aidns_module ckpt from: {aidns_path}")
        self.aidns_modules = DepthAIDNs.from_pretrained(aidns_path)

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [
                image.transpose(0, 1),
                torch.zeros(3, num_frames - 1, height, width).to(image.device),
            ],
            dim=1,
        )
        y = self.vae.encode(
            [vae_input.to(dtype=self.torch_dtype, device=self.device)],
            device=self.device,
        )[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    def decode_video(
        self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        frames = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return frames

    def _prepare_input(
        self,
        transformer: WanModel,
        timestep: torch.Tensor,
        context: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
    ):
        t = transformer.time_embedding(
            sinusoidal_embedding_1d(transformer.freq_dim, timestep)
        )
        t_mod = transformer.time_projection(t).unflatten(1, (6, transformer.dim))
        context = transformer.text_embedding(context)

        if transformer.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = transformer.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = transformer.patchify(x)

        freqs = (
            torch.cat(
                [
                    transformer.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, t, t_mod, freqs, (f, h, w)

    def _output(
        self,
        transformer: WanModel,
        hidden_state: torch.Tensor,
        embed_t: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ):
        x = transformer.head(hidden_state, embed_t)
        x = transformer.unpatchify(x, grid_size)
        return x

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: Image,
        depth: Image,
        seed=None,
        height: int = 544,
        width: int = 960,
        num_frames: int = 49,
        denoising_strength=1.0,
        num_inference_steps: int = 50,
        time_shift: float = 5.0,
        guidance_scale: float = 5.0,
    ):
        do_cfg = False
        if guidance_scale > 1.0:
            do_cfg = True
        self.text_encoder.to(device=self.device)
        self.image_encoder.to(device=self.device)
        self.rgb_dit.to(device=self.device)
        self.depth_dit.to(device=self.device)
        self.vae.to(device=self.device)
        self.aidns_modules.to(device=self.device, dtype=self.torch_dtype)
        ## set timesteps
        self.rgb_scheduler.set_timesteps(
            num_inference_steps, denoising_strength, shift=time_shift
        )
        self.depth_scheduler.set_timesteps(
            num_inference_steps, denoising_strength, shift=time_shift
        )

        rgb_latent = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=self.device,
            dtype=self.torch_dtype,
        )
        depth_latent = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=self.device,
            dtype=self.torch_dtype,
        )

        prompt_embed = self.encode_prompt(prompt)["context"].to(
            dtype=self.torch_dtype, device=self.device
        )
        if do_cfg:
            negative_prompt_embed = self.encode_prompt("")["context"].to(
                dtype=self.torch_dtype, device=self.device
            )
            ## 在batch维度连接 1->2
            prompt_embed = torch.cat([negative_prompt_embed, prompt_embed], dim=0)
        ## image encode
        image_emb = self.encode_image(image, num_frames, height, width)
        if do_cfg:
            image_emb = {
                "clip_feature": torch.cat([image_emb["clip_feature"]] * 2, dim=0),
                "y": torch.cat([image_emb["y"]] * 2, dim=0),
            }
        ## depth encode
        depth_emb = self.encode_image(depth, num_frames, height, width)
        if do_cfg:
            depth_emb = {
                "clip_feature": torch.cat([depth_emb["clip_feature"]] * 2, dim=0),
                "y": torch.cat([depth_emb["y"]] * 2, dim=0),
            }

        for idx, timestep in enumerate(tqdm(self.rgb_scheduler.timesteps)):
            if do_cfg:
                rgb_latent = torch.cat([rgb_latent] * 2, dim=0)
            if do_cfg:
                depth_latent = torch.cat([depth_latent] * 2, dim=0)
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            if do_cfg:
                timestep = timestep.expand(rgb_latent.shape[0])

            (
                rgb_hidden_state,
                rgb_context,
                rgb_embed_t,
                rgb_t_mod,
                rgb_freqs,
                rgb_grid_size,
            ) = self._prepare_input(
                self.rgb_dit,
                timestep,
                prompt_embed,
                rgb_latent,
                image_emb["y"],
                image_emb["clip_feature"],
            )

            (
                depth_hidden_state,
                depth_context,
                depth_embed_t,
                depth_t_mod,
                depth_freqs,
                depth_grid_size,
            ) = self._prepare_input(
                self.depth_dit,
                timestep,
                prompt_embed,
                depth_latent,
                depth_emb["y"],
                depth_emb["clip_feature"],
            )

            for block_idx in range(len(self.rgb_dit.blocks)):
                rgb_block = self.rgb_dit.blocks[block_idx]
                rgb_hidden_state, rgb_cross_attn_map = rgb_block(
                    rgb_hidden_state, rgb_context, rgb_t_mod, rgb_freqs
                )

                depth_block = self.depth_dit.blocks[block_idx]
                depth_hidden_state, _ = depth_block(
                    depth_hidden_state,
                    depth_context,
                    depth_t_mod,
                    depth_freqs,
                    rgb_cross_attn_map,
                )

                ## AIDN: depth to image
                if block_idx in self.target_idxs:
                    id = self.target_idxs.index(block_idx)
                    rgb_hidden_state = self.aidns_modules.D2I_aidn_blocks[id](
                        rgb_hidden_state, rgb_t_mod.flatten(1), depth_hidden_state
                    )

            rgb_noise_pred = self._output(
                self.rgb_dit, rgb_hidden_state, rgb_embed_t, rgb_grid_size
            )
            if do_cfg:
                rgb_noise_pred_cond, rgb_noise_pred_uncond = rgb_noise_pred.chunk(
                    2, dim=0
                )
                rgb_noise_pred = rgb_noise_pred_uncond + guidance_scale * (
                    rgb_noise_pred_cond - rgb_noise_pred_uncond
                )

            depth_noise_pred = self._output(
                self.depth_dit, depth_hidden_state, depth_embed_t, depth_grid_size
            )
            if do_cfg:
                depth_noise_pred_cond, depth_noise_pred_uncond = depth_noise_pred.chunk(
                    2, dim=0
                )
                depth_noise_pred = depth_noise_pred_uncond + guidance_scale * (
                    depth_noise_pred_cond - depth_noise_pred_uncond
                )

            ## scheduler step
            if idx + 1 >= len(self.rgb_scheduler.timesteps):
                rgb_dt = 0 - self.rgb_scheduler.sigmas[idx]
            else:
                rgb_dt = self.rgb_scheduler.sigmas[idx + 1] - self.rgb_scheduler.sigmas[idx]
            rgb_latent = (rgb_latent.chunk(2, dim=0)[0] if do_cfg else rgb_latent) + rgb_noise_pred * rgb_dt
            # rgb_latent = self.rgb_scheduler.step(
            #     rgb_noise_pred,
            #     self.rgb_scheduler.timesteps[idx],
            #     rgb_latent.chunk(2, dim=0)[0] if do_cfg else rgb_latent
            # )
            if idx + 1 >= len(self.depth_scheduler.timesteps):
                depth_dt = 0 - self.depth_scheduler.sigmas[idx]
            else:
                depth_dt = self.depth_scheduler.sigmas[idx + 1] - self.depth_scheduler.sigmas[idx]
            depth_latent = (depth_latent.chunk(2, dim=0)[0] if do_cfg else depth_latent) + depth_noise_pred * depth_dt
            # depth_latent = self.depth_scheduler.step(
            #     depth_noise_pred,
            #     self.depth_scheduler.timesteps[idx],
            #     depth_latent.chunk(2, dim=0)[0] if do_cfg else depth_latent
            # )

        rgb_frames = self.decode_video(rgb_latent, tiled=False)
        depth_frames = self.decode_video(depth_latent, tiled=False)

        rgb_frames = rearrange(rgb_frames[0], "C T H W -> T H W C")
        rgb_frames = (rgb_frames / 2 + 0.5).clamp(0, 1).float()
        rgb_frames = [f.cpu().numpy() for f in rgb_frames]  # [0, 1]

        depth_frames = rearrange(depth_frames[0], "C T H W -> T H W C")
        depth_frames = (depth_frames / 2 + 0.5).clamp(0, 1).float()
        depth_frames = [f.cpu().numpy() for f in depth_frames]  # [0, 1]

        return rgb_frames, depth_frames

class WanDepthTry1Pipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.rgb_dit: WanModel = None
        self.depth_dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.aidns_modules: DepthAIDNs = None
        self.model_names = [
            "text_encoder",
            "rgb_dit",
            "depth_dit",
            "vae",
            "aidns_modules",
        ]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.target_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanDepthTry1Pipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        self.rgb_dit = model_manager.fetch_model("wan_video_dit")
        self.depth_dit = copy.deepcopy(self.rgb_dit)

    def load_lora(self, model_path: str):
        rgb_transformer_path = os.path.join(model_path, "rgb_transformer_lora")
        depth_transformer_path = os.path.join(model_path, "depth_transformer_lora")
        aidns_path = os.path.join(model_path, "aidns_modules")

        print(f"load rgb_dit lora ckpt from: {rgb_transformer_path}")
        self.rgb_dit = PeftModel.from_pretrained(self.rgb_dit, rgb_transformer_path)

        print(f"load depth_dit lora ckpt from: {depth_transformer_path}")
        self.depth_dit = PeftModel.from_pretrained(
            self.depth_dit, depth_transformer_path
        )

        print(f"load aidns_module ckpt from: {aidns_path}")
        self.aidns_modules = DepthAIDNs.from_pretrained(aidns_path)

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [
                image.transpose(0, 1),
                torch.zeros(3, num_frames - 1, height, width).to(image.device),
            ],
            dim=1,
        )
        y = self.vae.encode(
            [vae_input.to(dtype=self.torch_dtype, device=self.device)],
            device=self.device,
        )[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    def decode_video(
        self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        frames = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return frames

    def _prepare_input(
        self,
        transformer: WanModel,
        timestep: torch.Tensor,
        context: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
    ):
        t = transformer.time_embedding(
            sinusoidal_embedding_1d(transformer.freq_dim, timestep)
        )
        t_mod = transformer.time_projection(t).unflatten(1, (6, transformer.dim))
        context = transformer.text_embedding(context)

        if transformer.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = transformer.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = transformer.patchify(x)

        freqs = (
            torch.cat(
                [
                    transformer.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, t, t_mod, freqs, (f, h, w)

    def _output(
        self,
        transformer: WanModel,
        hidden_state: torch.Tensor,
        embed_t: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ):
        x = transformer.head(hidden_state, embed_t)
        x = transformer.unpatchify(x, grid_size)
        return x

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: Image,
        depth: Image,
        seed=None,
        height: int = 544,
        width: int = 960,
        num_frames: int = 49,
        denoising_strength=1.0,
        num_inference_steps: int = 50,
        time_shift: float = 5.0,
        guidance_scale: float = 5.0,
    ):
        do_cfg = False
        if guidance_scale > 1.0:
            do_cfg = True
        self.text_encoder.to(device=self.device)
        self.image_encoder.to(device=self.device)
        self.rgb_dit.to(device=self.device)
        self.depth_dit.to(device=self.device)
        self.vae.to(device=self.device)
        self.aidns_modules.to(device=self.device, dtype=self.torch_dtype)
        ## set timesteps
        self.scheduler.set_timesteps(
            num_inference_steps, denoising_strength, shift=time_shift
        )

        rgb_latent = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=self.device,
            dtype=self.torch_dtype,
        )
        depth_latent = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=self.device,
            dtype=self.torch_dtype,
        )

        prompt_embed = self.encode_prompt(prompt)["context"].to(
            dtype=self.torch_dtype, device=self.device
        )
        if do_cfg:
            negative_prompt_embed = self.encode_prompt("")["context"].to(
                dtype=self.torch_dtype, device=self.device
            )
            ## 在batch维度连接 1->2
            prompt_embed = torch.cat([negative_prompt_embed, prompt_embed], dim=0)
        ## image encode
        image_emb = self.encode_image(image, num_frames, height, width)
        if do_cfg:
            image_emb = {
                "clip_feature": torch.cat([image_emb["clip_feature"]] * 2, dim=0),
                "y": torch.cat([image_emb["y"]] * 2, dim=0),
            }
        ## depth encode
        depth_emb = self.encode_image(depth, num_frames, height, width)
        if do_cfg:
            depth_emb = {
                "clip_feature": torch.cat([depth_emb["clip_feature"]] * 2, dim=0),
                "y": torch.cat([depth_emb["y"]] * 2, dim=0),
            }

        for idx, timestep in enumerate(tqdm(self.scheduler.timesteps)):
            if do_cfg:
                rgb_latent = torch.cat([rgb_latent] * 2, dim=0)
            if do_cfg:
                depth_latent = torch.cat([depth_latent] * 2, dim=0)
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            if do_cfg:
                timestep = timestep.expand(rgb_latent.shape[0])

            (
                rgb_hidden_state,
                rgb_context,
                rgb_embed_t,
                rgb_t_mod,
                rgb_freqs,
                rgb_grid_size,
            ) = self._prepare_input(
                self.rgb_dit,
                timestep,
                prompt_embed,
                rgb_latent,
                image_emb["y"],
                image_emb["clip_feature"],
            )

            (
                depth_hidden_state,
                depth_context,
                depth_embed_t,
                depth_t_mod,
                depth_freqs,
                depth_grid_size,
            ) = self._prepare_input(
                self.depth_dit,
                timestep,
                prompt_embed,
                depth_latent,
                depth_emb["y"],
                depth_emb["clip_feature"],
            )

            for block_idx in range(len(self.rgb_dit.blocks)):
                rgb_block = self.rgb_dit.blocks[block_idx]
                rgb_hidden_state, rgb_cross_attn_map = rgb_block(
                    rgb_hidden_state, rgb_context, rgb_t_mod, rgb_freqs
                )

                ## AIDN: image to depth
                if block_idx in self.target_idxs:
                    id = self.target_idxs.index(block_idx)
                    depth_hidden_state = self.aidns_modules.D2I_aidn_blocks[id](
                        depth_hidden_state, depth_t_mod.flatten(1), rgb_hidden_state
                    )

                depth_block = self.depth_dit.blocks[block_idx]
                depth_hidden_state, _ = depth_block(
                    depth_hidden_state,
                    depth_context,
                    depth_t_mod,
                    depth_freqs,
                    rgb_cross_attn_map,
                )

            rgb_noise_pred = self._output(
                self.rgb_dit, rgb_hidden_state, rgb_embed_t, rgb_grid_size
            )
            if do_cfg:
                rgb_noise_pred_cond, rgb_noise_pred_uncond = rgb_noise_pred.chunk(
                    2, dim=0
                )
                rgb_noise_pred = rgb_noise_pred_uncond + guidance_scale * (
                    rgb_noise_pred_cond - rgb_noise_pred_uncond
                )

            depth_noise_pred = self._output(
                self.depth_dit, depth_hidden_state, depth_embed_t, depth_grid_size
            )
            if do_cfg:
                depth_noise_pred_cond, depth_noise_pred_uncond = depth_noise_pred.chunk(
                    2, dim=0
                )
                depth_noise_pred = depth_noise_pred_uncond + guidance_scale * (
                    depth_noise_pred_cond - depth_noise_pred_uncond
                )

            ## scheduler step
            rgb_latent = self.scheduler.step(
                rgb_noise_pred,
                self.scheduler.timesteps[idx],
                rgb_latent.chunk(2, dim=0)[0] if do_cfg else rgb_latent,
            )

            depth_latent = self.scheduler.step(
                depth_noise_pred,
                self.scheduler.timesteps[idx],
                depth_latent.chunk(2, dim=0)[0] if do_cfg else depth_latent,
            )

        rgb_frames = self.decode_video(rgb_latent, tiled=False)
        depth_frames = self.decode_video(depth_latent, tiled=False)

        rgb_frames = rearrange(rgb_frames[0], "C T H W -> T H W C")
        rgb_frames = (rgb_frames / 2 + 0.5).clamp(0, 1).float()
        rgb_frames = [f.cpu().numpy() for f in rgb_frames]  # [0, 1]

        depth_frames = rearrange(depth_frames[0], "C T H W -> T H W C")
        depth_frames = (depth_frames / 2 + 0.5).clamp(0, 1).float()
        depth_frames = [f.cpu().numpy() for f in depth_frames]  # [0, 1]

        return rgb_frames, depth_frames


class WanDepthNoAIDNPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.rgb_dit: WanModel = None
        self.depth_dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = [
            "text_encoder",
            "rgb_dit",
            "depth_dit",
            "vae",
        ]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.target_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanDepthNoAIDNPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        self.rgb_dit = model_manager.fetch_model("wan_video_dit")
        self.depth_dit = model_manager.fetch_model("wan_video_dit")

    def load_lora(self, model_path: str):
        rgb_transformer_path = os.path.join(model_path, "rgb_transformer_lora")
        depth_transformer_path = os.path.join(model_path, "depth_transformer_lora")

        self.rgb_dit = PeftModel.from_pretrained(self.rgb_dit, rgb_transformer_path)
        print(f"load rgb_dit lora ckpt from: {rgb_transformer_path}")
        self.depth_dit = PeftModel.from_pretrained(
            self.depth_dit, depth_transformer_path
        )
        print(f"load depth_dit lora ckpt from: {depth_transformer_path}")

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [
                image.transpose(0, 1),
                torch.zeros(3, num_frames - 1, height, width).to(image.device),
            ],
            dim=1,
        )
        y = self.vae.encode(
            [vae_input.to(dtype=self.torch_dtype, device=self.device)],
            device=self.device,
        )[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    def decode_video(
        self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        frames = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return frames

    def _prepare_input(
        self,
        transformer: WanModel,
        timestep: torch.Tensor,
        context: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
    ):
        t = transformer.time_embedding(
            sinusoidal_embedding_1d(transformer.freq_dim, timestep)
        )
        t_mod = transformer.time_projection(t).unflatten(1, (6, transformer.dim))
        context = transformer.text_embedding(context)

        if transformer.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = transformer.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = transformer.patchify(x)

        freqs = (
            torch.cat(
                [
                    transformer.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, t, t_mod, freqs, (f, h, w)

    def _output(
        self,
        transformer: WanModel,
        hidden_state: torch.Tensor,
        embed_t: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ):
        x = transformer.head(hidden_state, embed_t)
        x = transformer.unpatchify(x, grid_size)
        return x

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: Image,
        depth: Image,
        seed=None,
        height: int = 544,
        width: int = 960,
        num_frames: int = 49,
        denoising_strength=1.0,
        num_inference_steps: int = 50,
        time_shift: float = 5.0,
        guidance_scale: float = 5.0,
    ):
        do_cfg = False
        if guidance_scale > 1.0:
            do_cfg = True
        self.text_encoder.to(device=self.device)
        self.image_encoder.to(device=self.device)
        self.rgb_dit.to(device=self.device)
        self.depth_dit.to(device=self.device)
        self.vae.to(device=self.device)
        ## set timesteps
        self.scheduler.set_timesteps(
            num_inference_steps, denoising_strength, shift=time_shift
        )

        rgb_latent = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=self.device,
            dtype=self.torch_dtype,
        )
        depth_latent = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=self.device,
            dtype=self.torch_dtype,
        )

        prompt_embed = self.encode_prompt(prompt, positive=True)["context"].to(
            dtype=self.torch_dtype, device=self.device
        )
        if do_cfg:
            negative_prompt_embed = self.encode_prompt("", positive=False)[
                "context"
            ].to(dtype=self.torch_dtype, device=self.device)
            ## 在batch维度连接 1->2
            prompt_embed = torch.cat([negative_prompt_embed, prompt_embed], dim=0)
        ## image encode
        image_emb = self.encode_image(image, num_frames, height, width)
        if do_cfg:
            image_emb = {
                "clip_feature": torch.cat([image_emb["clip_feature"]] * 2, dim=0),
                "y": torch.cat([image_emb["y"]] * 2, dim=0),
            }
        ## depth encode
        depth_emb = self.encode_image(depth, num_frames, height, width)
        if do_cfg:
            depth_emb = {
                "clip_feature": torch.cat([depth_emb["clip_feature"]] * 2, dim=0),
                "y": torch.cat([depth_emb["y"]] * 2, dim=0),
            }

        for idx, timestep in enumerate(tqdm(self.scheduler.timesteps)):
            if do_cfg:
                rgb_latent = torch.cat([rgb_latent] * 2, dim=0)
            if do_cfg:
                depth_latent = torch.cat([depth_latent] * 2, dim=0)
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            if do_cfg:
                timestep = timestep.expand(rgb_latent.shape[0])

            (
                rgb_hidden_state,
                rgb_context,
                rgb_embed_t,
                rgb_t_mod,
                rgb_freqs,
                rgb_grid_size,
            ) = self._prepare_input(
                self.rgb_dit,
                timestep,
                prompt_embed,
                rgb_latent,
                image_emb["y"],
                image_emb["clip_feature"],
            )

            (
                depth_hidden_state,
                depth_context,
                depth_embed_t,
                depth_t_mod,
                depth_freqs,
                depth_grid_size,
            ) = self._prepare_input(
                self.depth_dit,
                timestep,
                prompt_embed,
                depth_latent,
                depth_emb["y"],
                depth_emb["clip_feature"],
            )

            for block_idx in range(len(self.rgb_dit.blocks)):
                rgb_block = self.rgb_dit.blocks[block_idx]
                rgb_hidden_state, rgb_cross_attn_map = rgb_block(
                    rgb_hidden_state, rgb_context, rgb_t_mod, rgb_freqs
                )

                depth_block = self.depth_dit.blocks[block_idx]
                depth_hidden_state, _ = depth_block(
                    depth_hidden_state,
                    depth_context,
                    depth_t_mod,
                    depth_freqs,
                    rgb_cross_attn_map,
                )

            rgb_noise_pred = self._output(
                self.rgb_dit, rgb_hidden_state, rgb_embed_t, rgb_grid_size
            )
            if do_cfg:
                rgb_noise_pred_cond, rgb_noise_pred_uncond = rgb_noise_pred.chunk(
                    2, dim=0
                )
                rgb_noise_pred = rgb_noise_pred_uncond + guidance_scale * (
                    rgb_noise_pred_cond - rgb_noise_pred_uncond
                )

            depth_noise_pred = self._output(
                self.depth_dit, depth_hidden_state, depth_embed_t, depth_grid_size
            )
            if do_cfg:
                depth_noise_pred_cond, depth_noise_pred_uncond = depth_noise_pred.chunk(
                    2, dim=0
                )
                depth_noise_pred = depth_noise_pred_uncond + guidance_scale * (
                    depth_noise_pred_cond - depth_noise_pred_uncond
                )

            ## scheduler step
            rgb_latent = self.scheduler.step(
                rgb_noise_pred,
                self.scheduler.timesteps[idx],
                rgb_latent.chunk(2, dim=0)[0] if do_cfg else rgb_latent,
            )

            depth_latent = self.scheduler.step(
                depth_noise_pred,
                self.scheduler.timesteps[idx],
                depth_latent.chunk(2, dim=0)[0] if do_cfg else depth_latent,
            )

        rgb_frames = self.decode_video(rgb_latent, tiled=False)
        depth_frames = self.decode_video(depth_latent, tiled=False)

        rgb_frames = rearrange(rgb_frames[0], "C T H W -> T H W C")
        rgb_frames = (rgb_frames / 2 + 0.5).clamp(0, 1).float()
        rgb_frames = [f.cpu().numpy() for f in rgb_frames]  # [0, 1]

        depth_frames = rearrange(depth_frames[0], "C T H W -> T H W C")
        depth_frames = (depth_frames / 2 + 0.5).clamp(0, 1).float()
        depth_frames = [f.cpu().numpy() for f in depth_frames]  # [0, 1]

        return rgb_frames, depth_frames


class WanDepthTrajPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.rgb_dit: WanModel = None
        self.depth_dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.aidns_modules: DepthAIDNs = None
        self.trajectory_head: TrajectoryFPNHead = None
        self.model_names = [
            "text_encoder",
            "rgb_dit",
            "depth_dit",
            "vae",
            "aidns_modules",
        ]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.target_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanDepthTrajPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        self.rgb_dit = model_manager.fetch_model("wan_video_dit")
        self.depth_dit = copy.deepcopy(self.rgb_dit)

    def load_lora(self, model_path: str):
        rgb_transformer_path = os.path.join(model_path, "rgb_transformer_lora")
        depth_transformer_path = os.path.join(model_path, "depth_transformer_lora")
        aidns_path = os.path.join(model_path, "aidns_modules")
        trajectory_path = os.path.join(model_path, "traj_modules")

        self.rgb_dit = PeftModel.from_pretrained(self.rgb_dit, rgb_transformer_path)
        print(f"load rgb_dit lora ckpt from: {rgb_transformer_path}")

        self.depth_dit = PeftModel.from_pretrained(
            self.depth_dit, depth_transformer_path
        )
        print(f"load depth_dit lora ckpt from: {depth_transformer_path}")

        self.aidns_modules = DepthAIDNs.from_pretrained(aidns_path)
        print(f"load aidns_module ckpt from: {aidns_path}")

        self.trajectory_head = TrajectoryFPNHead.from_pretrained(trajectory_path)
        print(f"load trajectory_head ckpt from: {trajectory_path}")

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def encode_image(self, image, num_frames, height, width):
        ##BUG should't use Image.resize because this api will stretch the image, use crop_and_resize_frames instead
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [
                image.transpose(0, 1),
                torch.zeros(3, num_frames - 1, height, width).to(image.device),
            ],
            dim=1,
        )
        y = self.vae.encode(
            [vae_input.to(dtype=self.torch_dtype, device=self.device)],
            device=self.device,
        )[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    def decode_video(
        self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        frames = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return frames

    def _prepare_input(
        self,
        transformer: WanModel,
        timestep: torch.Tensor,
        context: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
    ):
        t = transformer.time_embedding(
            sinusoidal_embedding_1d(transformer.freq_dim, timestep)
        )
        t_mod = transformer.time_projection(t).unflatten(1, (6, transformer.dim))
        context = transformer.text_embedding(context)

        if transformer.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = transformer.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = transformer.patchify(x)

        freqs = (
            torch.cat(
                [
                    transformer.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, t, t_mod, freqs, (f, h, w)

    def _output(
        self,
        transformer: WanModel,
        hidden_state: torch.Tensor,
        embed_t: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ):
        x = transformer.head(hidden_state, embed_t)
        x = transformer.unpatchify(x, grid_size)
        return x

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: Image,
        depth: Image,
        seed=None,
        height: int = 544,
        width: int = 960,
        num_frames: int = 49,
        denoising_strength=1.0,
        num_inference_steps: int = 50,
        time_shift: float = 5.0,
        guidance_scale: float = 5.0,
    ):
        do_cfg = False
        if guidance_scale > 1.0:
            do_cfg = True
        self.text_encoder.to(device=self.device)
        self.image_encoder.to(device=self.device)
        self.rgb_dit.to(device=self.device)
        self.depth_dit.to(device=self.device)
        self.vae.to(device=self.device)
        self.aidns_modules.to(device=self.device, dtype=self.torch_dtype)
        self.trajectory_head.to(device=self.device, dtype=self.torch_dtype)
        ## set timesteps
        self.scheduler.set_timesteps(
            num_inference_steps, denoising_strength, shift=time_shift
        )

        rgb_latent = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=self.device,
            dtype=self.torch_dtype,
        )
        depth_latent = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=self.device,
            dtype=self.torch_dtype,
        )

        prompt_embed = self.encode_prompt(prompt)["context"].to(
            dtype=self.torch_dtype, device=self.device
        )
        if do_cfg:
            negative_prompt_embed = self.encode_prompt("")["context"].to(
                dtype=self.torch_dtype, device=self.device
            )
            ## 在batch维度连接 1->2
            prompt_embed = torch.cat([negative_prompt_embed, prompt_embed], dim=0)
        ## image encode
        image_emb = self.encode_image(image, num_frames, height, width)
        if do_cfg:
            image_emb = {
                "clip_feature": torch.cat([image_emb["clip_feature"]] * 2, dim=0),
                "y": torch.cat([image_emb["y"]] * 2, dim=0),
            }
        ## depth encode
        depth_emb = self.encode_image(depth, num_frames, height, width)
        if do_cfg:
            depth_emb = {
                "clip_feature": torch.cat([depth_emb["clip_feature"]] * 2, dim=0),
                "y": torch.cat([depth_emb["y"]] * 2, dim=0),
            }

        diff_features = []

        for idx, timestep in enumerate(tqdm(self.scheduler.timesteps)):
            if do_cfg:
                rgb_latent = torch.cat([rgb_latent] * 2, dim=0)
            if do_cfg:
                depth_latent = torch.cat([depth_latent] * 2, dim=0)
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            if do_cfg:
                timestep = timestep.expand(rgb_latent.shape[0])

            (
                rgb_hidden_state,
                rgb_context,
                rgb_embed_t,
                rgb_t_mod,
                rgb_freqs,
                rgb_grid_size,
            ) = self._prepare_input(
                self.rgb_dit,
                timestep,
                prompt_embed,
                rgb_latent,
                image_emb["y"],
                image_emb["clip_feature"],
            )

            (
                depth_hidden_state,
                depth_context,
                depth_embed_t,
                depth_t_mod,
                depth_freqs,
                depth_grid_size,
            ) = self._prepare_input(
                self.depth_dit,
                timestep,
                prompt_embed,
                depth_latent,
                depth_emb["y"],
                depth_emb["clip_feature"],
            )

            for block_idx in range(len(self.rgb_dit.blocks)):
                rgb_block = self.rgb_dit.blocks[block_idx]
                rgb_hidden_state, rgb_cross_attn_map = rgb_block(
                    rgb_hidden_state, rgb_context, rgb_t_mod, rgb_freqs
                )

                depth_block = self.depth_dit.blocks[block_idx]
                depth_hidden_state, _ = depth_block(
                    depth_hidden_state,
                    depth_context,
                    depth_t_mod,
                    depth_freqs,
                    rgb_cross_attn_map,
                )

                ## AIDN: depth to image
                if block_idx in self.target_idxs:
                    id = self.target_idxs.index(block_idx)
                    rgb_hidden_state = self.aidns_modules.D2I_aidn_blocks[id](
                        rgb_hidden_state, rgb_t_mod.flatten(1), depth_hidden_state
                    )
                if idx == len(self.scheduler.timesteps) - 1:
                    diff_features.append(rgb_hidden_state)

            rgb_noise_pred = self._output(
                self.rgb_dit, rgb_hidden_state, rgb_embed_t, rgb_grid_size
            )
            if do_cfg:
                rgb_noise_pred_cond, rgb_noise_pred_uncond = rgb_noise_pred.chunk(
                    2, dim=0
                )
                rgb_noise_pred = rgb_noise_pred_uncond + guidance_scale * (
                    rgb_noise_pred_cond - rgb_noise_pred_uncond
                )

            depth_noise_pred = self._output(
                self.depth_dit, depth_hidden_state, depth_embed_t, depth_grid_size
            )
            if do_cfg:
                depth_noise_pred_cond, depth_noise_pred_uncond = depth_noise_pred.chunk(
                    2, dim=0
                )
                depth_noise_pred = depth_noise_pred_uncond + guidance_scale * (
                    depth_noise_pred_cond - depth_noise_pred_uncond
                )

            ## scheduler step
            rgb_latent = self.scheduler.step(
                rgb_noise_pred,
                self.scheduler.timesteps[idx],
                rgb_latent.chunk(2, dim=0)[0] if do_cfg else rgb_latent,
            )

            depth_latent = self.scheduler.step(
                depth_noise_pred,
                self.scheduler.timesteps[idx],
                depth_latent.chunk(2, dim=0)[0] if do_cfg else depth_latent,
            )

        ## postprocess diff_features
        features = []
        for i in range(len(diff_features)):
            t_feature = rearrange(
                diff_features[i],
                "B (F H W) C -> (B F) C H W",
                F = rgb_grid_size[0],
                H = rgb_grid_size[1],
                W = rgb_grid_size[2]
            )
            features.append(t_feature)
        trajectory_out = self.trajectory_head(features)
        trajectory_out = rearrange(
            trajectory_out,
            "(B F) C H W -> B C F H W",
            F = rgb_grid_size[0]
        )

        rgb_frames = self.decode_video(rgb_latent, tiled=False)
        depth_frames = self.decode_video(depth_latent, tiled=False)
        trajectory_frames = self.decode_video(trajectory_out[:1], tiled=False)

        rgb_frames = rearrange(rgb_frames[0], "C T H W -> T H W C")
        rgb_frames = (rgb_frames / 2 + 0.5).clamp(0, 1).float()
        rgb_frames = [f.cpu().numpy() for f in rgb_frames]  # [0, 1]

        depth_frames = rearrange(depth_frames[0], "C T H W -> T H W C")
        depth_frames = (depth_frames / 2 + 0.5).clamp(0, 1).float()
        depth_frames = [f.cpu().numpy() for f in depth_frames]  # [0, 1]

        trajectory_frames = rearrange(trajectory_frames[0], "C T H W -> T H W C")
        trajectory_frames = (trajectory_frames / 2 + 0.5).clamp(0, 1).float()
        trajectory_frames = [f.cpu().numpy() for f in trajectory_frames]  # [0, 1]

        return rgb_frames, depth_frames, trajectory_frames
