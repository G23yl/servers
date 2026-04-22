import argparse
import os
import time
import random

import imageio
import numpy as np
import torch
from loguru import logger
from PIL import Image

from examples.kairos_based.modules.kairos_model_modal_t2v import KairosMotModel
from examples.kairos_based.modules.text_encoders import QwenVLTextEncoder
from examples.kairos_based.modules.utils import init_weights_on_device, load_state_dict
from examples.kairos_based.modules.vaes import WanVideoVAE


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", type=str, help="Real model checkpoint path"
    )
    parser.add_argument(
        "--model_path", type=str, help="VAE and text_encoder checkpoint path"
    )
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, contorted human joints, objects floating against natural forces, abrupt shot changes",
    )
    parser.add_argument("--modal_type", type=str)
    parser.add_argument("--save_combined", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--cfg_scale", type=float, default=5.0)

    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_video(path, video, fps=16, quality=5):
    # video: [F, H, W, C] - 0..255
    imageio.mimwrite(path, video, fps=fps, quality=quality)


if __name__ == "__main__":
    args = get_args()

    setup_seed(args.seed)

    text_encoder_path = f"{args.model_path}/Qwen2.5-VL-7B-Instruct-AWQ/"
    vae_path = f"{args.model_path}/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
    # text_encoder
    text_encoder = QwenVLTextEncoder(
        dtype=torch.bfloat16,
        device="cuda",
        from_pretrained=text_encoder_path,
    )
    text_encoder.requires_grad_(False)
    # vae
    vae_state_dict = load_state_dict(vae_path)
    vae_state_dict_converter = WanVideoVAE.state_dict_converter()
    state_dict_results = vae_state_dict_converter.from_civitai(vae_state_dict)
    extra_kwargs = {}
    with init_weights_on_device():
        vae = WanVideoVAE(**extra_kwargs)
    if hasattr(vae, "eval"):
        vae = vae.eval()
    vae.load_state_dict(state_dict_results, assign=True)
    vae = vae.to(dtype=torch.bfloat16, device="cuda")
    vae.requires_grad_(False)

    model = KairosMotModel.from_checkpoint(
        args.checkpoint_path, device="cuda", torch_dtype=torch.bfloat16
    )

    logger.info("[Inference] Loaded model successfully!!!")

    image = Image.open(args.image_path).convert("RGB")

    # [F, H, W, C]
    rgb_video, modal_video = model.inference(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        modal_type=args.modal_type,
        image=image,
        vae=vae,
        prompter=text_encoder,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        shift=args.shift,
        cfg_scale=args.cfg_scale,
        height=args.height,
        width=args.width,
        tiled=True,
    )

    # save_video
    ckpt_step = args.checkpoint_path.split("/")[-2]
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    rgb_out_path = os.path.join(
        args.output_path, f"{current_time}_{args.prompt[:50].replace('/', '')}_rgb_{ckpt_step}.mp4"
    )
    depth_out_path = os.path.join(
        args.output_path,
        f"{current_time}_{args.prompt[:50].replace('/', '')}_{args.modal_type}_{ckpt_step}.mp4",
    )
    save_video(rgb_out_path, rgb_video, fps=16, quality=5)
    save_video(depth_out_path, modal_video, fps=16, quality=5)
    logger.info(f"[Inference] Save videos to {rgb_out_path}")
    if args.save_combined:
        combined_out_path = os.path.join(
            args.output_path,
            f"{current_time}_{args.prompt[:50].replace('/', '')}_combined_{ckpt_step}.mp4",
        )
        combined_video = np.concat([rgb_video, modal_video], axis=-2)
        save_video(combined_out_path, combined_video, fps=16, quality=5)
        logger.info(f"[Inference] Save combined videos to {combined_out_path}")

"""
100 - moved blue cloth to the center of the table Trossen WidowX 250 robot arm
923: moved the can to the middle of the table Trossen WidowX 250 robot arm
1000: move the can to the bottom right of the table Trossen WidowX 250 robot arm
1500: transfer the passoir into its yellow container in the upper right Trossen WidowX 250 robot arm
"""
