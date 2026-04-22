import argparse
import copy
import os

import torch
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import wandb
from diffsynth import ModelManager, WanVideoPipeline
from examples.depth_wan.transformer_depth import DepthAIDNs, DepthWanFullModel
from examples.depth_wan.robodataset import RoboDepth


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str, help="Path to save model")
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--wandb_key", type=str, default="", help="wandb key to use wandb"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="", help="wandb entity name to use wandb"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="", help="wandb project name to use wandb"
    )
    parser.add_argument("--save_steps", type=int, default=50, help="saving interval")
    parser.add_argument("--lr_scheduler_name", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=200)
    parser.add_argument("--max_training_steps", type=int, default=5000)
    parser.add_argument("--max_norm", type=float, default=10.0)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    accelerator = Accelerator()
    weight_type = torch.float32
    if accelerator.mixed_precision == "bf16":
        weight_type = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        weight_type = torch.float16

    ## set up wandb for logger
    if args.use_wandb and accelerator.is_local_main_process:
        for key in ["wandb_key", "wandb_entity", "wandb_project"]:
            if getattr(args, key) == "":
                raise ValueError(f"Need {key} to use wandb")
        wandb.login(key=args.wandb_key)
        wandb.init(
            mode="online",
            name="wan_depth_full_train",
            entity=args.wandb_entity,
            project=args.wandb_project,
        )

    model_manager = ModelManager(torch_dtype=weight_type, device="cpu")
    if os.path.exists(args.model_path):
        manager_input_paths = [
            f"{args.model_path}/model.safetensors",
            f"{args.model_path}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{args.model_path}/Wan2.1_VAE.pth",
            f"{args.model_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ]
        model_manager.load_models(manager_input_paths, torch_dtype=weight_type)
    ## for encode
    pipe = WanVideoPipeline.from_model_manager(model_manager)
    pipe.text_encoder.to(device=accelerator.device)
    pipe.image_encoder.to(device=accelerator.device)
    pipe.vae.to(device=accelerator.device)
    pipe.scheduler.set_timesteps(1000, training=True)
    pipe.requires_grad_(False)
    ## rgb dit
    rgb_transformer = pipe.dit
    rgb_transformer.requires_grad_(True)
    ## depth dit
    depth_transformer = copy.deepcopy(rgb_transformer)
    depth_transformer.requires_grad_(True)
    ##BUG 因为depth分支的cross_map是rgb分支的，因此depth分支的k和norm_k层没用，但是却是可训练的，所以冻结它们的参数. 这也是为什么lora微调的时候不能加k和norm_k的原因
    for dit_block in depth_transformer.blocks:
        dit_block.cross_attn.k.requires_grad_(False)
        dit_block.cross_attn.norm_k.requires_grad_(False)

    if accelerator.is_local_main_process:
        logger.info("Load rgb_transformer and depth_transformer")

    aidns = DepthAIDNs(num_layers=10, time_adaptive=True)
    aidns.requires_grad_(True)
    aidns.to(accelerator.device)
    aidns.train()

    rgb_transformer.to(accelerator.device)
    depth_transformer.to(accelerator.device)
    rgb_transformer.train()
    depth_transformer.train()

    model_transformer = DepthWanFullModel(rgb_transformer, depth_transformer, aidns)
    params_optim = filter(lambda p: p.requires_grad, model_transformer.parameters())

    ## prepare dataset
    dataset = RoboDepth(data_root=args.data_path, is_train=True, max_num_frames=49)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    if accelerator.is_local_main_process:
        logger.info("Load dataset")

    ## optimizer and lr_scheduler
    optimizer = torch.optim.AdamW(params_optim, lr=args.lr, weight_decay=0.001)
    lr_scheduler = get_scheduler(
        args.lr_scheduler_name,
        optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_training_steps * accelerator.num_processes,
    )

    model_transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model_transformer, optimizer, dataloader, lr_scheduler
    )
    logger.info(
        f"[rank{accelerator.process_index}] dataloader length: {len(dataloader)}"
    )

    if accelerator.is_local_main_process:
        logger.info("Start training")
    progress_bar = tqdm(
        range(0, args.max_training_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    global_step = 0
    while True:
        model_transformer.train()
        for step, batch in enumerate(dataloader):
            prompt = batch["prompts"]
            first_image = batch["first_image"].to(dtype=weight_type)
            first_depth = batch["first_depth"].to(dtype=weight_type)
            video = batch["frames"].to(dtype=weight_type)
            depth = batch["depth"].to(dtype=weight_type)

            batch_size, channels, frames, H, W = video.shape
            prompt_embed = pipe.encode_prompt(prompt)["context"].to(
                dtype=weight_type, device=accelerator.device
            )
            rgb_clip_feat = pipe.image_encoder.encode_image(
                first_image.transpose(1, 2)
            ).to(dtype=weight_type, device=accelerator.device)
            depth_clip_feat = pipe.image_encoder.encode_image(
                first_depth.transpose(1, 2)
            ).to(dtype=weight_type, device=accelerator.device)
            # prepare mask
            rgb_msk = torch.ones(
                batch_size, 1, frames, H // 8, W // 8, device=accelerator.device
            )
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
            depth_msk = copy.deepcopy(rgb_msk)

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
            ).to(dtype=weight_type, device=accelerator.device)
            rgb_y = pipe.vae.encode(rgb_vae_input, device=accelerator.device).to(
                dtype=weight_type, device=accelerator.device
            )
            # [B, 16+4, 13, h, w]
            rgb_y = torch.concat([rgb_msk, rgb_y], dim=1).to(
                dtype=weight_type, device=accelerator.device
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
            ).to(dtype=weight_type, device=accelerator.device)
            depth_y = pipe.vae.encode(depth_vae_input, device=accelerator.device).to(
                dtype=weight_type, device=accelerator.device
            )
            # [B, 16+4, 13, h, w]
            depth_y = torch.concat([depth_msk, depth_y], dim=1).to(
                dtype=weight_type, device=accelerator.device
            )

            rgb_latent = pipe.vae.encode(video, device=accelerator.device).to(
                dtype=weight_type, device=accelerator.device
            )
            depth_latent = pipe.vae.encode(depth, device=accelerator.device).to(
                dtype=weight_type, device=accelerator.device
            )
            rgb_noise = torch.randn_like(
                rgb_latent, dtype=weight_type, device=accelerator.device
            )
            depth_noise = torch.randn_like(
                depth_latent, dtype=weight_type, device=accelerator.device
            )
            timestep_id = torch.randint(0, pipe.scheduler.num_train_timesteps, (1,))
            timestep = pipe.scheduler.timesteps[timestep_id].to(
                dtype=weight_type, device=accelerator.device
            )
            noisy_rgb_latents = pipe.scheduler.add_noise(
                rgb_latent, rgb_noise, timestep
            ).to(dtype=weight_type, device=accelerator.device)
            noisy_depth_latents = pipe.scheduler.add_noise(
                depth_latent, depth_noise, timestep
            ).to(dtype=weight_type, device=accelerator.device)
            rgb_target = pipe.scheduler.training_target(rgb_latent, rgb_noise, timestep)
            depth_target = pipe.scheduler.training_target(
                depth_latent, depth_noise, timestep
            )

            ## predict rgb_noise and depth_noise
            rgb_out, depth_out = model_transformer(
                noisy_rgb_latents,
                noisy_depth_latents,
                timestep,
                prompt_embed,
                rgb_clip_feat,
                depth_clip_feat,
                rgb_y,
                depth_y,
                args.use_gradient_checkpointing,
                args.use_gradient_checkpointing_offload,
            )

            assert rgb_target.shape == rgb_out.shape
            assert depth_target.shape == depth_out.shape

            rgb_loss = F.mse_loss(rgb_out.float(), rgb_target.float(), reduction="mean")
            depth_loss = F.mse_loss(
                depth_out.float(), depth_target.float(), reduction="mean"
            )
            tot_loss = rgb_loss + depth_loss

            ## gather rgb_loss, depth_loss, loss between all processes
            avg_rgb_loss = accelerator.gather(rgb_loss).mean()
            avg_depth_loss = accelerator.gather(depth_loss).mean()
            avg_tot_loss = accelerator.gather(tot_loss).mean()

            accelerator.backward(tot_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_optim, args.max_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            ## log
            if accelerator.is_local_main_process and args.use_wandb:
                wandb.log(
                    {
                        "total_loss": avg_tot_loss.item(),
                        "rgb_loss": avg_rgb_loss.item(),
                        "depth_loss": avg_depth_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            ## save ckpt
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                if global_step % args.save_steps == 0:
                    out_dir = os.path.join(args.output_path, f"step_{global_step:08d}")
                    os.makedirs(out_dir, exist_ok=True)
                    logger.info(f"Save to {out_dir}")

                    mtransformer = accelerator.unwrap_model(model_transformer)
                    rgb_transformer_path = os.path.join(out_dir, "rgb_transformer")
                    os.makedirs(rgb_transformer_path, exist_ok=True)
                    torch.save(
                        mtransformer.rgb_transformer.state_dict(),
                        os.path.join(rgb_transformer_path, "model.pth"),
                    )

                    depth_transformer_path = os.path.join(out_dir, "depth_transformer")
                    os.makedirs(depth_transformer_path, exist_ok=True)
                    torch.save(
                        mtransformer.depth_transformer.state_dict(),
                        os.path.join(depth_transformer_path, "model.pth"),
                    )

                    mtransformer.aidns.save_pretrained(
                        os.path.join(out_dir, "aidns_modules")
                    )

            if global_step >= args.max_training_steps:
                break
        if global_step >= args.max_training_steps:
            break

    accelerator.end_training()
