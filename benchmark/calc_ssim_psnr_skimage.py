import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import json
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# from calculate_lpips import calculate_lpips
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


def to_numpy_frames(frames):
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    return np.asarray(frames, dtype=np.float32)


def channel_first_to_last(frame):
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        return np.transpose(frame, (1, 2, 0))
    return frame


def prepare_frame_for_skimage(frame):
    frame = channel_first_to_last(frame)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return np.squeeze(frame, axis=-1), None
    channel_axis = -1 if frame.ndim == 3 else None
    return frame, channel_axis


def calculate_skimage_metrics(videos1, videos2, only_final=False):
    # videos: [batch_size, timestamps, channel, h, w], pixel values in [0, 1]
    assert videos1.shape == videos2.shape

    ssim_results = []
    psnr_results = []

    for video_num in range(videos1.shape[0]):
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        ssim_results_of_a_video = []
        psnr_results_of_a_video = []

        for clip_timestamp in range(len(video1)):
            img1, channel_axis = prepare_frame_for_skimage(video1[clip_timestamp])
            img2, _ = prepare_frame_for_skimage(video2[clip_timestamp])

            ssim_results_of_a_video.append(
                structural_similarity(
                    img1,
                    img2,
                    data_range=1.0,
                    channel_axis=channel_axis,
                )
            )
            psnr_results_of_a_video.append(
                peak_signal_noise_ratio(img1, img2, data_range=1.0)
            )

        ssim_results.append(ssim_results_of_a_video)
        psnr_results.append(psnr_results_of_a_video)

    ssim_results = np.array(ssim_results)
    psnr_results = np.array(psnr_results)

    if only_final:
        return {
            "ssim": {
                "value": [np.mean(ssim_results)],
                "value_std": [np.std(ssim_results)],
            },
            "psnr": {
                "value": [np.mean(psnr_results)],
                "value_std": [np.std(psnr_results)],
            },
        }

    ssim = []
    ssim_std = []
    psnr = []
    psnr_std = []
    for clip_timestamp in range(videos1.shape[1]):
        ssim.append(np.mean(ssim_results[:, clip_timestamp]))
        ssim_std.append(np.std(ssim_results[:, clip_timestamp]))
        psnr.append(np.mean(psnr_results[:, clip_timestamp]))
        psnr_std.append(np.std(psnr_results[:, clip_timestamp]))

    return {
        "ssim": {"value": ssim, "value_std": ssim_std},
        "psnr": {"value": psnr, "value_std": psnr_std},
    }


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
# lpips_res = []
# lpips_std = []

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

    pred_frames = [
        Image.fromarray(f).resize((target_frames.shape[-1], target_frames.shape[-2]))
        for f in pred_frames
    ]
    pred_frames = [np.array(f) for f in pred_frames]
    pred_frames = np.stack(pred_frames, axis=0)
    pred_frames = torch.from_numpy(pred_frames).permute(0, 3, 1, 2)
    pred_frames = (pred_frames / 255.0).clamp(0, 1)

    if not verbose:
        print(
            f"pred_frames shape: {pred_frames.shape}, target_frames shape: {target_frames.shape}"
        )
        verbose = True

    videos1 = to_numpy_frames(pred_frames)[None]
    videos2 = to_numpy_frames(target_frames)[None]
    metrics = calculate_skimage_metrics(videos1, videos2, only_final=only_final)
    ssim = metrics["ssim"]
    ssim_res.append(ssim["value"][0])
    ssim_std.append(ssim["value_std"][0])
    psnr = metrics["psnr"]
    psnr_res.append(psnr["value"][0])
    psnr_std.append(psnr["value_std"][0])
    # lpips = calculate_lpips(videos1, videos2, "cuda", only_final=only_final)
    # lpips_res.append(lpips["value"][0])
    # lpips_std.append(lpips["value_std"][0])

    del pred_frames, target_frames, videos1, videos2

result = {}
ssim_res = np.array(ssim_res)
psnr_res = np.array(psnr_res)
# lpips_res = np.array(lpips_res)
ssim_std = np.array(ssim_std)
psnr_std = np.array(psnr_std)
# lpips_std = np.array(lpips_std)
result["ssim"] = np.mean(ssim_res)
result["ssim_std"] = np.mean(ssim_std)
result["psnr"] = np.mean(psnr_res)
result["psnr_std"] = np.mean(psnr_std)
# result["lpips"] = np.mean(lpips_res)
# result["lpips_std"] = np.mean(lpips_std)

print(result)

with open("ssim_psnr.json", "w") as f:
    json.dump(result, f, indent=4)
