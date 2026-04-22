import argparse
import os
import time

import torch
from PIL import Image
import numpy as np
from safetensors.torch import load_file

from examples.depth_wan.pipeline import WanDepthPipeline, WanDepthTrajPipeline, WanDepthTry1Pipeline
from diffsynth import ModelManager
from diffusers.utils.export_utils import export_to_video


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--depth_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--prompt", type=str)
    # "normal" | "trajectory" | "flow" | "try1"
    parser.add_argument("--model_type", type=str, default="normal")
    parser.add_argument("--lora", action="store_true")

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--time_shift", type=float, default=5.0)
    parser.add_argument("--guidance_scale", type=float, default=5.0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    model_manager = ModelManager(device="cpu")
    if os.path.exists(args.base_model_path):
        manager_input_paths = [
            f"{args.base_model_path}/model.safetensors",
            f"{args.base_model_path}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{args.base_model_path}/Wan2.1_VAE.pth",
            f"{args.base_model_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ]
        model_manager.load_models(manager_input_paths, torch_dtype=torch.bfloat16)

    image = Image.open(args.image_path).convert("RGB")
    depth = Image.open(args.depth_path).convert("RGB")

    if args.model_type == "trajectory" or args.model_type == "flow":
        pipe = WanDepthTrajPipeline.from_model_manager(
            model_manager, torch_dtype=torch.bfloat16, device="cuda"
        )
        pipe.load_lora(args.model_path)
        video, depth_video, traj_video = pipe(
            prompt=args.prompt,
            image=image,
            depth=depth,
            num_inference_steps=args.num_inference_steps,
            seed=0,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            time_shift=args.time_shift,
            guidance_scale=args.guidance_scale,
        )
        combined_video = []
        for rgb, depth, traj in zip(video, depth_video, traj_video):
            # [H, W*3, C]
            comb = np.concat([rgb, depth, traj], axis=1)
            combined_video.append(comb)

        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        combined_out_path = os.path.join(
            args.output_path, f"{current_time}_{args.prompt[:100].replace('/', '')}_combined.mp4"
        )
        export_to_video(combined_video, combined_out_path, fps=8)
    elif args.model_type == "normal":
        ## load base model and lora ckpt
        pipe = WanDepthPipeline.from_model_manager(
            model_manager, torch_dtype=torch.bfloat16, device="cuda"
        )
        if args.lora:
            pipe.load_lora(args.model_path)
        else:
            pipe.load_no_lora(args.model_path)

        video, depth_video = pipe(
            prompt=args.prompt,
            image=image,
            depth=depth,
            num_inference_steps=args.num_inference_steps,
            seed=0,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            time_shift=args.time_shift,
            guidance_scale=args.guidance_scale,
        )
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        rgb_out_path = os.path.join(
            args.output_path, f"{current_time}_{args.prompt[:100].replace('/', '')}_rgb.mp4"
        )
        depth_out_path = os.path.join(
            args.output_path,
            f"{current_time}_{args.prompt[:100].replace('/', '')}_depth.mp4",
        )
        export_to_video(video, rgb_out_path, fps=8)
        export_to_video(depth_video, depth_out_path, fps=8)
    elif args.model_type == "try1":
        ## load base model and lora ckpt
        pipe = WanDepthTry1Pipeline.from_model_manager(
            model_manager, torch_dtype=torch.bfloat16, device="cuda"
        )
        pipe.load_lora(args.model_path)

        video, depth_video = pipe(
            prompt=args.prompt,
            image=image,
            depth=depth,
            num_inference_steps=args.num_inference_steps,
            seed=0,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            time_shift=args.time_shift,
            guidance_scale=args.guidance_scale,
        )
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        rgb_out_path = os.path.join(
            args.output_path, f"{current_time}_{args.prompt[:100].replace('/', '')}_rgb.mp4"
        )
        depth_out_path = os.path.join(
            args.output_path,
            f"{current_time}_{args.prompt[:100].replace('/', '')}_depth.mp4",
        )
        export_to_video(video, rgb_out_path, fps=8)
        export_to_video(depth_video, depth_out_path, fps=8)
    else:
        raise NotImplementedError(f"Unknown type: {args.model_type}")


"""
100 - moved blue cloth to the center of the table Trossen WidowX 250 robot arm
923: moved the can to the middle of the table Trossen WidowX 250 robot arm
1000: move the can to the bottom right of the table Trossen WidowX 250 robot arm
1500: transfer the passoir into its yellow container in the upper right Trossen WidowX 250 robot arm
"""
