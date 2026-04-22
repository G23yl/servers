import argparse
import os

import torch
from PIL import Image
from tqdm import tqdm
from accelerate import PartialState

from examples.depth_wan.pipeline import (
    WanDepthPipeline,
    WanDepthTry1Pipeline,
    WanDepthTrajPipeline,
)
from diffsynth import ModelManager
from diffusers.utils.export_utils import export_to_video


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--model_path", type=str)
    # "normal" | "trajectory" | "flow" | "try1"
    parser.add_argument("--model_type", type=str, default="normal")
    parser.add_argument("--lora", action="store_true")

    # parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_data_path", type=str, help="Need when testing")
    parser.add_argument("--test_out_path", type=str, help="Need when testing")
    parser.add_argument("--rgb_save_name", type=str)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--time_shift", type=float, default=5.0)
    parser.add_argument("--guidance_scale", type=float, default=5.0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    distributed_state = PartialState()
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

    tot_samples = []
    for folder in os.listdir(args.test_data_path):
        pp = os.path.join(args.test_data_path, folder)
        img_pp = os.path.join(pp, "image.png")
        depth_pp = os.path.join(pp, "depth.png")
        ins_pp = os.path.join(pp, "ins.txt")
        with open(ins_pp, "r") as f:
            prompt = f.read()
        image = Image.open(img_pp).convert("RGB")
        depth = Image.open(depth_pp).convert("RGB")
        tot_samples.append([prompt, image, depth, folder])

    if distributed_state.is_local_main_process:
        print(f"=== Using {args.model_type} mode ===")

    if args.model_type == "normal":
        ## load base model and lora ckpt
        pipe = WanDepthPipeline.from_model_manager(
            model_manager, torch_dtype=torch.bfloat16, device=distributed_state.device
        )
        if args.lora:
            pipe.load_lora(args.model_path)
        else:
            pipe.load_no_lora(args.model_path)

    elif args.model_type == "try1":
        ## load base model and lora ckpt
        pipe = WanDepthTry1Pipeline.from_model_manager(
            model_manager, torch_dtype=torch.bfloat16, device=distributed_state.device
        )
        pipe.load_lora(args.model_path)
    elif args.model_type == "trajectory" or args.model_type == "flow":
        pipe = WanDepthTrajPipeline.from_model_manager(
            model_manager, torch_dtype=torch.bfloat16, device=distributed_state.device
        )
        pipe.load_lora(args.model_path)

    with distributed_state.split_between_processes(tot_samples) as samples:
        for sample in tqdm(
            samples, desc="Total: ", disable=not distributed_state.is_local_main_process
        ):
            prompt, image, depth, folder = sample[0], sample[1], sample[2], sample[-1]
            video_path = os.path.join(args.test_out_path, folder)
            os.makedirs(video_path, exist_ok=True)
            res = pipe(
                prompt=prompt,
                image=image,
                depth=depth,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                time_shift=args.time_shift,
                guidance_scale=args.guidance_scale,
            )
            video = res[0]
            depth_video = res[1]
            # rgb_out_path = os.path.join(video_path, "b_6k_pred_traj_rgb_10000.mp4")
            # depth_out_path = os.path.join(video_path, "b_6k_pred_traj_depth_10000.mp4")
            rgb_out_path = os.path.join(video_path, args.rgb_save_name)
            depth_out_path = os.path.join(
                video_path, args.rgb_save_name.replace("rgb", "depth")
            )
            export_to_video(video, rgb_out_path, fps=8)
            export_to_video(depth_video, depth_out_path, fps=8)
