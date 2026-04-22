import cv2
import numpy as np
import torch
from calc_absrel import (
    abs_relative_difference,
    crop_and_resize_frames,
    adjust_num_frames,
)
from diffusers.utils.export_utils import export_to_video
import os
import shutil
from tqdm import tqdm

# tesser_video_path = "/mnt/zhouxin-mnt/self_forcing/test_images/1/tesser_pred.mp4"

# cap = cv2.VideoCapture(tesser_video_path)
# pred_frames = []  # [F, H, W, C]
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pred_frames.append(frame)
# cap.release()
# pred_frames = np.stack(pred_frames, axis=0)
# W = pred_frames.shape[-2]
# pred_depth_videos = pred_frames[:, :, W // 3 : W // 3 * 2, 0]
# pred_depth_videos = torch.from_numpy(pred_depth_videos).float()

# # pred_depth_videos = (pred_depth_videos - pred_depth_videos.min()) / (
# #     pred_depth_videos.max() - pred_depth_videos.min() + 1e-8
# # )
# # pred_depth_videos *= 255.0

# target_depths = np.load(
#     tesser_video_path.replace("tesser_pred.mp4", "gen_target_depth.npz")
# )["arr_0"].astype(np.float32)
# # target_depths = target_depths[:, :, :W // 3]
# target_depths = torch.from_numpy(target_depths)
# # target_depths = (target_depths - target_depths.min()) / (
# #     target_depths.max() - target_depths.min() + 1e-8
# # )
# target_depths = target_depths / 2 + 0.5
# target_depths *= 255.0

# valid_mask = torch.logical_and(target_depths > 1e-5, target_depths < 256.0).bool()

# absrel = abs_relative_difference(pred_depth_videos, target_depths, valid_mask)
# print("Abs Rel:", absrel)

path = "/mnt/zhouxin-mnt/self_forcing/test_images"

folders = [f for f in os.listdir(path)]

tpath = "/mnt/zhouxin-mnt/self_forcing/tesser_test_images"
tfolders = [f for f in os.listdir(tpath)[:200]]

for i in range(len(folders)):
    p = os.path.join(path, folders[i], "ins.txt")
    pp = os.path.join(tpath, tfolders[i], "ins.txt")
    with open(p, "r") as f:
        ori_txt = f.read()
    with open(pp, "r") as f:
        tesser_txt = f.read()
    if ori_txt != tesser_txt:
        print(f"path: {p} | {pp}\nori: {ori_txt}\ntesser: {tesser_txt}")