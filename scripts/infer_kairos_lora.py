import argparse
import os
import random
import time

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
        "--model_path",
        type=str,
        required=True,
        help="Base model root path containing DiT, VAE, and text encoder weights.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the saved PEFT LoRA adapter directory.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Optional base DiT checkpoint path. Defaults to the Kairos robot DiT under model_path.",
    )
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, contorted human joints, objects floating against natural forces, abrupt shot changes",
    )
    parser.add_argument("--modal_type", type=str, required=True)
    parser.add_argument("--save_combined", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument(
        "--no_merge_lora",
        action="store_true",
        help="Keep the LoRA adapter unmerged after loading. By default it is merged for inference.",
    )

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


def load_text_encoder(model_path):
    text_encoder_path = f"{model_path}/Qwen2.5-VL-7B-Instruct-AWQ/"
    text_encoder = QwenVLTextEncoder(
        dtype=torch.bfloat16,
        device="cuda",
        from_pretrained=text_encoder_path,
    )
    text_encoder.requires_grad_(False)
    return text_encoder


def load_vae(model_path):
    vae_path = f"{model_path}/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
    vae_state_dict = load_state_dict(vae_path)
    vae_state_dict_converter = WanVideoVAE.state_dict_converter()
    state_dict_results = vae_state_dict_converter.from_civitai(vae_state_dict)

    with init_weights_on_device():
        vae = WanVideoVAE()
    if hasattr(vae, "eval"):
        vae = vae.eval()
    vae.load_state_dict(state_dict_results, assign=True)
    vae = vae.to(dtype=torch.bfloat16, device="cuda")
    vae.requires_grad_(False)
    return vae


def load_kairos_lora_model(args):
    dit_path = args.dit_path
    if dit_path is None:
        dit_path = f"{args.model_path}/models/robot/kairos-robot-4B-480P-16fps.safetensors"

    model = KairosMotModel(device="cuda", torch_dtype=torch.bfloat16)
    model.from_pretrained({"dit": dit_path})
    model.requires_grad_(False)

    model = model.load_lora_weights(
        args.lora_path,
        is_trainable=False,
        merge=not args.no_merge_lora,
    )
    if args.no_merge_lora and not hasattr(model, "inference"):
        model = model.get_base_model()
    model.eval()
    return model


if __name__ == "__main__":
    args = get_args()

    setup_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    text_encoder = load_text_encoder(args.model_path)
    vae = load_vae(args.model_path)
    model = load_kairos_lora_model(args)

    logger.info("[Inference] Loaded base model and LoRA adapter successfully!!!")

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

    lora_step = os.path.basename(os.path.normpath(args.lora_path))
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    prompt_name = args.prompt[:50].replace("/", "")
    rgb_out_path = os.path.join(
        args.output_path, f"{current_time}_{prompt_name}_rgb_{lora_step}.mp4"
    )
    modal_out_path = os.path.join(
        args.output_path,
        f"{current_time}_{prompt_name}_{args.modal_type}_{lora_step}.mp4",
    )
    if not args.save_combined:
        save_video(rgb_out_path, rgb_video, fps=16, quality=5)
        save_video(modal_out_path, modal_video, fps=16, quality=5)
        logger.info(f"[Inference] Save videos to {rgb_out_path} and {modal_out_path}")
    else:
        combined_out_path = os.path.join(
            args.output_path,
            f"{current_time}_{prompt_name}_combined_{lora_step}.mp4",
        )
        combined_video = np.concatenate([rgb_video, modal_video], axis=-2)
        save_video(combined_out_path, combined_video, fps=16, quality=5)
        logger.info(f"[Inference] Save combined videos to {combined_out_path}")
