import argparse
import os
import time
import random

import imageio
import numpy as np
import torch
from loguru import logger
from PIL import Image

from examples.depth_wan.mot import DepthWanMoT


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--utils_path", type=str, default=None)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--depth_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--save_combined", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--shift", type=float, default=5.0)

    args = parser.parse_args()
    return args

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def save_video(path, video, fps=16, quality=5):
    # video: [F, H, W, C] - 0..255
    imageio.mimwrite(path, video, fps=fps, quality=quality)

if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)

    model = DepthWanMoT(is_training_mode=False)
    model.load_checkpoint(
        args.model_path,
        weight_type=torch.bfloat16,
        device="cuda",
        utils_path=args.utils_path,
    )
    logger.info("[Inference] Loaded model successfully!!!")

    image = Image.open(args.image_path).convert("RGB")
    depth = Image.open(args.depth_path).convert("RGB")
    # [B, C, H, W]
    image_tensor = torch.from_numpy((np.array(image.resize((args.width, args.height))) / 255.0 - 0.5) * 2).permute(2, 0, 1).unsqueeze(0)
    depth_tensor = torch.from_numpy((np.array(depth.resize((args.width, args.height))) / 255.0 - 0.5) * 2).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        # [B, F, H, W, C]
        rgb_videos, depth_videos = model.inference(
            args.num_inference_steps,
            args.prompt,
            image_tensor,
            depth_tensor,
            args.num_frames,
            args.shift,
            args.height,
            args.width
        )
        # save_video
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        for i in range(len(rgb_videos)):
            rgb_out_path = os.path.join(
                args.output_path, f"{current_time}_{args.prompt[:50].replace('/', '')}_rgb_{i}.mp4"
            )
            depth_out_path = os.path.join(
                args.output_path, f"{current_time}_{args.prompt[:50].replace('/', '')}_depth_{i}.mp4"
            )
            save_video(rgb_out_path, rgb_videos[i], fps=16, quality=5)
            save_video(depth_out_path, depth_videos[i], fps=16, quality=5)
            logger.info(f"[Inference] Save videos to {rgb_out_path}")
            if args.save_combined:
                combined_out_path = os.path.join(
                    args.output_path, f"{current_time}_{args.prompt[:50].replace('/', '')}_combined_{i}.mp4"
                )
                rgb_video, depth_video = rgb_videos[i], depth_videos[i]
                combined_video = np.concat([rgb_video, depth_video], axis=-2)
                save_video(combined_out_path, combined_video, fps=16, quality=5)
                logger.info(f"[Inference] Save combined videos to {combined_out_path}")

"""
100 - moved blue cloth to the center of the table Trossen WidowX 250 robot arm
923: moved the can to the middle of the table Trossen WidowX 250 robot arm
1000: move the can to the bottom right of the table Trossen WidowX 250 robot arm
1500: transfer the passoir into its yellow container in the upper right Trossen WidowX 250 robot arm
"""
