import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import save_file

import wandb
from examples.depth_wan.mot import DepthWanMoT
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
            name="wan_depth_mot",
            entity=args.wandb_entity,
            project=args.wandb_project,
        )

    model = DepthWanMoT(is_training_mode=True)
    model.load_pretrained_models(
        args.model_path, weight_type, device=accelerator.device
    )

    parameters_optim = [p for p in model.parameters() if p.requires_grad]
    # with open("trainable_params.txt", "w") as f:
    #     trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    #     f.write("\n".join(trainable_params))
    num_params = sum(p.numel() for p in parameters_optim)
    if accelerator.is_local_main_process:
        logger.info(f"Number of trainable parameters: {num_params / 1e9:.2f}B")

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
        logger.info("Loaded dataset")

    ## optimizer and lr_scheduler
    optimizer = torch.optim.AdamW(parameters_optim, lr=args.lr, weight_decay=0.001)
    lr_scheduler = get_scheduler(
        args.lr_scheduler_name,
        optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_training_steps * accelerator.num_processes,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    logger.info(
        f"[rank{accelerator.process_index}] dataloader length: {len(dataloader)}"
    )

    progress_bar = tqdm(
        range(0, args.max_training_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    global_step = 0
    while True:
        model.train()
        for step, batch in enumerate(dataloader):
            prompt = batch["prompts"]
            first_image = batch["first_image"].to(dtype=weight_type)
            first_depth = batch["first_depth"].to(dtype=weight_type)
            video = batch["frames"].to(dtype=weight_type)
            depth = batch["depth"].to(dtype=weight_type)

            # deal with DDP wrap
            # model1 = model.module if hasattr(model, "module") else model
            # ret = model1.training_step(
                # prompt, first_image, first_depth, video, depth, return_dict=True
            # )
            try:
                ret = model(prompt, first_image, first_depth, video, depth, return_dict=True)
            except Exception as e:
                if accelerator.is_local_main_process:
                    logger.debug(f"[Custom] Failed: {e}")
            video_loss, depth_loss, total_loss = (
                ret["video_loss"],
                ret["depth_loss"],
                ret["total_loss"],
            )
            ## gather rgb_loss, depth_loss, loss between all processes
            avg_video_loss = accelerator.gather(video_loss).mean()
            avg_depth_loss = accelerator.gather(depth_loss).mean()
            avg_total_loss = accelerator.gather(total_loss).mean()

            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(parameters_optim, args.max_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            ## log
            if accelerator.is_local_main_process and args.use_wandb:
                wandb.log(
                    {
                        "total_loss": avg_total_loss.item(),
                        "video_loss": avg_video_loss.item(),
                        "depth_loss": avg_depth_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            ## save ckpt
            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    out_dir = os.path.join(args.output_path, f"step_{global_step:08d}")
                    os.makedirs(out_dir, exist_ok=True)
                    # Unwrap the model if it's wrapped by DDP/DeepSpeed
                    unwrapped_model = accelerator.unwrap_model(model)
                    # get trainable params i.e. video_dit and depth_dit
                    trainable_state_dict = {}
                    for name, p in unwrapped_model.named_parameters():
                        if p.requires_grad:
                            trainable_state_dict[name] = p.contiguous()

                    model_save_path = os.path.join(out_dir, "model.safetensors")
                    save_file(trainable_state_dict, model_save_path)
                    logger.info(f"Model saved to {model_save_path}")

            if global_step >= args.max_training_steps:
                break
        if global_step >= args.max_training_steps:
            break

    accelerator.end_training()
