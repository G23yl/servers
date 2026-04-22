import argparse
import os

import torch
from PIL import Image
from tqdm import tqdm
from accelerate import PartialState
import numpy as np
import random
import imageio
from loguru import logger

from examples.kairos_based.modules.kairos_model_modal_t2v import KairosMotModel
from examples.kairos_based.modules.text_encoders import QwenVLTextEncoder
from examples.kairos_based.modules.utils import init_weights_on_device, load_state_dict
from examples.kairos_based.modules.vaes import WanVideoVAE


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_video(path, video, fps=16, quality=5):
    # video: [F, H, W, C] - 0..255
    imageio.mimwrite(path, video, fps=fps, quality=quality)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, help="Real model checkpoint path"
    )
    parser.add_argument(
        "--model_path", type=str, help="VAE and text_encoder checkpoint path"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, contorted human joints, objects floating against natural forces, abrupt shot changes",
    )
    parser.add_argument("--modal_type", type=str)

    parser.add_argument("--test_data_path", type=str, help="Need when testing")
    parser.add_argument("--test_out_path", type=str, help="Need when testing")
    parser.add_argument("--rgb_save_name", type=str)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--cfg_scale", type=float, default=5.0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    distributed_state = PartialState()
    args = get_args()

    setup_seed(args.seed)

    tot_samples = []
    for folder in os.listdir(args.test_data_path):
        pp = os.path.join(args.test_data_path, folder)
        img_pp = os.path.join(pp, "image.png")
        ins_pp = os.path.join(pp, "ins.txt")
        with open(ins_pp, "r") as f:
            prompt = f.read()
        image = Image.open(img_pp).convert("RGB")
        tot_samples.append([prompt, image, folder])

    if distributed_state.is_local_main_process:
        logger.info(f"[Test] test_data lenght: {len(tot_samples)}")

    text_encoder_path = f"{args.model_path}/Qwen2.5-VL-7B-Instruct-AWQ/"
    vae_path = f"{args.model_path}/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
    # text_encoder
    text_encoder = QwenVLTextEncoder(
        dtype=torch.bfloat16,
        device=distributed_state.device,
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
    vae = vae.to(dtype=torch.bfloat16, device=distributed_state.device)
    vae.requires_grad_(False)

    model = KairosMotModel.from_checkpoint(
        args.checkpoint_path,
        device=distributed_state.device,
        torch_dtype=torch.bfloat16,
    )

    if distributed_state.is_local_main_process:
        logger.info("[Test] Loaded model successfully!!!")

    with distributed_state.split_between_processes(tot_samples) as samples:
        for sample in samples:
            prompt, image, folder = sample[0], sample[1], sample[2]
            video_path = os.path.join(args.test_out_path, folder)
            os.makedirs(video_path, exist_ok=True)

            # [F, H, W, C]
            rgb_video, modal_video = model.inference(
                prompt=prompt,
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

            rgb_out_path = os.path.join(video_path, args.rgb_save_name)
            modal_out_path = os.path.join(
                video_path, args.rgb_save_name.replace("rgb", str(args.modal_type))
            )
            save_video(rgb_out_path, rgb_video, fps=8)
            save_video(modal_out_path, modal_video, fps=8)
