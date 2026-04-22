import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
# from fvd.styleganv.fvd import get_fvd_feats, load_i3d_pretrained
from calculate_fvd import trans
import argparse
from PIL import Image


# ps: pixel value should be in [0, 1]!
method = "styleganv"
if method == 'styleganv':
    from fvd.styleganv.fvd import get_fvd_feats, load_i3d_pretrained
elif method == 'videogpt':
    from fvd.videogpt.fvd import load_i3d_pretrained
    from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats


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


def crop_and_resize_frames(frames, target_size, interpolation="bilinear"):
    # frames: [F, H, W, C]
    target_height, target_width = target_size
    original_height, original_width = frames[0].shape[:2]
    if original_height == target_height and original_width == target_width:
        return [frame for frame in frames]

    # ==== interpolation method ====
    if interpolation == "bilinear":
        interpolation = cv2.INTER_LINEAR
    elif interpolation == "nearest":
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR

    processed_frames = []
    for frame in frames:
        original_height, original_width = frame.shape[:2]
        aspect_ratio_target = target_width / target_height
        aspect_ratio_original = original_width / original_height

        if aspect_ratio_original > aspect_ratio_target:
            new_width = int(aspect_ratio_target * original_height)
            start_x = (original_width - new_width) // 2
            cropped_frame = frame[:, start_x : start_x + new_width]
        else:
            new_height = int(original_width / aspect_ratio_target)
            start_y = (original_height - new_height) // 2
            cropped_frame = frame[start_y : start_y + new_height, :]
        resized_frame = cv2.resize(
            cropped_frame, (target_width, target_height), interpolation=interpolation
        )
        processed_frames.append(resized_frame)

    return processed_frames


def get_feat(videos: torch.Tensor):
    video_len = len(videos.shape)
    if video_len == 4:
        videos = videos.unsqueeze(0)
    i3d = load_i3d_pretrained(device=device)
    videos = trans(videos)
    feat = get_fvd_feats(videos, i3d, device)
    if video_len == 4:
        return feat[0]
    return feat


parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", type=str)
parser.add_argument("--pred", type=str, help="Predicted video base name")
parser.add_argument("--target", type=str, help="Target video base name")
args = parser.parse_args()

path = args.test_dir
pred_video_name = f"{args.pred}.mp4"
target_video_name = f"{args.target}.mp4"

folders = os.listdir(path)
device = torch.device("cuda")
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
    pred_num_frames = len(pred_frames)

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
    target_frames = adjust_num_frames(target_frames, pred_num_frames)
    target_frames = np.stack(target_frames, axis=0)
    target_frames = torch.from_numpy(target_frames).permute(0, 3, 1, 2)
    target_frames = (target_frames / 255.0).clamp(0, 1)

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

    pred_feat = get_feat(pred_frames)
    np.save(f"preprocess_data/pred/{folder}.npy", pred_feat)

    target_feat = get_feat(target_frames)
    np.save(f"preprocess_data/target/{folder}.npy", target_feat)
