import argparse
import os
import time

import torch
from PIL import Image

from diffsynth import ModelManager, WanVideoPipeline
from diffusers.utils.export_utils import export_to_video


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--dit_verbose", action="store_true")

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--data_path", type=str, default="", help="Need when testing")
    parser.add_argument("--out_path", type=str, default="", help="Need when testing")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    model_manager = ModelManager(device="cuda")
    if os.path.exists(args.model_path):
        manager_input_paths = [
            f"{args.model_path}/step-002800_resume.ckpt",
            f"{args.model_path}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{args.model_path}/Wan2.1_VAE.pth",
        ]
        model_manager.load_models(
            [
                f"{args.model_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
            ],
            torch_dtype=torch.float32,
        )
        model_manager.load_models(manager_input_paths, torch_dtype=torch.bfloat16)
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )

    if args.dit_verbose:
        for n, m in pipe.dit.named_modules():
            print(n, type(m))
        exit(0)

    if not args.test:
        image = Image.open(args.image_path).convert("RGB")

        # prompt: put the small can in the right top of the bleu cloth Trossen WidowX 250 robot arm
        video = pipe(
            prompt=args.prompt,
            input_image=image,
            num_inference_steps=50,
            seed=0,
            tiled=False,
            height=544,
            width=960,
            num_frames=49,
            cfg_scale=1.0,
            sigma_shift=5.0,
        )
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        out_path = os.path.join(
            args.output_path, f"{args.prompt[:100].replace('/', '')}_{current_time}.mp4"
        )
        export_to_video(video, out_path, fps=8)
    else:
        # infer with test_dataloader
        from tqdm import tqdm

        samples = []
        for folder in os.listdir(args.data_path):
            pp = os.path.join(args.data_path, folder)
            img_pp = os.path.join(pp, "image.png")
            ins_pp = os.path.join(pp, "ins.txt")
            with open(ins_pp, "r") as f:
                prompt = f.read()
            image = Image.open(img_pp).convert("RGB")
            samples.append([prompt, image, folder])

        for sample in tqdm(samples, desc="Total: "):
            image, prompt, folder = sample[1], sample[0], sample[2]
            video_path = os.path.join(args.out_path, folder)
            if os.path.exists(os.path.join(video_path, "pred1.mp4")):
                continue
            video = pipe(
                prompt=prompt,
                input_image=image,
                num_inference_steps=50,
                tiled=False,
                height=544,
                width=960,
                num_frames=49,
                cfg_scale=1.0,
                sigma_shift=5.0,
            )
            out_path = os.path.join(video_path, "pred1.mp4")
            export_to_video(video, out_path, fps=8)



"""
TOKENIZERS_PARALLELISM=False python infer.py \
    --model_path /mnt/zhouxin-mnt/ckpt/SkyReels-V2-I2V-1.3B-540P \
    --image_path data/input/first_image_5000.png \
    --output_path data/videos \
    --prompt "move the blue knife to the top right of the table Trossen WidowX 250 robot arm"

TOKENIZERS_PARALLELISM=False python infer.py \
    --model_path /mnt/zhouxin-mnt/ckpt/SkyReels-V2-I2V-1.3B-540P \
    --dit_verbose

0   - put the small can in the right top of the bleu cloth Trossen WidowX 250 robot arm
100 - moved blue cloth to the center of the table Trossen WidowX 250 robot arm
40000 - put banana in pot or pan Trossen WidowX 250 robot arm
5000 - move the blue knife to the top right of the table Trossen WidowX 250 robot arm
"""
