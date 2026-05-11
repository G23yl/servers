import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from calculate_psnr import calculate_psnr
import json
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
import argparse
from PIL import Image

# ps: pixel value should be in [0, 1]!


def adjust_num_frames(frames, target_num_frames):
    frame_count = len(frames)
    if frame_count < target_num_frames:
        extra = target_num_frames - frame_count
        if isinstance(frames, list):
            frames.extend([frames[-1]] * extra)
        elif isinstance(frames, torch.Tensor):
            frame_to_add = [frames[-1]] * extra
            frames = [f for f in frames] + frame_to_add
    elif frame_count > target_num_frames:
        indices = np.linspace(0, frame_count - 1, target_num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    return frames


parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", type=str)
parser.add_argument("--pred", type=str, help="Predicted video base name")
parser.add_argument("--target", type=str, help="Target video base name")
args = parser.parse_args()

path = args.test_dir
pred_video_name = f"{args.pred}.mp4"
target_video_name = f"{args.target}.mp4"
folders = os.listdir(path)
ssim_res = []
ssim_std = []
psnr_res = []
psnr_std = []
lpips_res = []
lpips_std = []

only_final = True
verbose = False

for folder in tqdm(folders):
    pred_video_path = os.path.join(path, folder, pred_video_name)
    if not os.path.exists(pred_video_path):
        continue
    cap = cv2.VideoCapture(pred_video_path)
    pred_frames = []  # [F, H, W, C]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_frames.append(frame)
    cap.release()

    target_video_path = os.path.join(path, folder, target_video_name)
    cap = cv2.VideoCapture(target_video_path)
    target_frames = []  # [F, H, W, C]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target_frames.append(frame)
    cap.release()

    pred_num_frames = len(pred_frames)
    target_num_frames = len(target_frames)
    # if pred_num_frames > target_num_frames:
    #     pred_frames = adjust_num_frames(pred_frames, target_num_frames)
    # elif pred_num_frames < target_num_frames:
    target_frames = adjust_num_frames(target_frames, pred_num_frames)

    target_frames = np.stack(target_frames, axis=0)
    target_frames = target_frames.transpose(0, 3, 1, 2)
    target_frames = (target_frames / 255.0).clip(0, 1)

    pred_frames = [Image.fromarray(f).resize((target_frames.shape[-1], target_frames.shape[-2])) for f in pred_frames]
    pred_frames = [np.array(f) for f in pred_frames]
    pred_frames = np.stack(pred_frames, axis=0)
    pred_frames = torch.from_numpy(pred_frames).permute(0, 3, 1, 2)
    pred_frames = (pred_frames / 255.0).clamp(0, 1)

    if not verbose:
        print(
            f"pred_frames shape: {pred_frames.shape}, target_frames shape: {target_frames.shape}"
        )
        verbose = True

    videos1, videos2 = np.array([pred_frames]), np.array([target_frames])
    ssim = calculate_ssim(videos1, videos2, only_final=only_final)
    ssim_res.append(ssim["value"][0])
    ssim_std.append(ssim["value_std"][0])
    psnr = calculate_psnr(videos1, videos2, only_final=only_final)
    psnr_res.append(psnr["value"][0])
    psnr_std.append(psnr["value_std"][0])
    videos1, videos2 = torch.from_numpy(videos1).float(), torch.from_numpy(videos2).float()
    lpips = calculate_lpips(videos1, videos2, "cuda", only_final=only_final)
    lpips_res.append(lpips["value"][0])
    lpips_std.append(lpips["value_std"][0])

    del pred_frames, target_frames, videos1, videos2

result = {}
ssim_res = np.array(ssim_res)
psnr_res = np.array(psnr_res)
lpips_res = np.array(lpips_res)
ssim_std = np.array(ssim_std)
psnr_std = np.array(psnr_std)
lpips_std = np.array(lpips_std)
result["ssim"] = np.mean(ssim_res)
result["ssim_std"] = np.mean(ssim_std)
result["psnr"] = np.mean(psnr_res)
result["psnr_std"] = np.mean(psnr_std)
result["lpips"] = np.mean(lpips_res)
result["lpips_std"] = np.mean(lpips_std)

print(result)

with open("ssim_psnr.json", "w") as f:
    json.dump(result, f, indent=4)
