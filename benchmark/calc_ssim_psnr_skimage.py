import cv2
import os
import numpy as np
from tqdm import tqdm
import json
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# from calculate_lpips import calculate_lpips
import argparse

# ps: pixel value should be in [0, 255]!


def adjust_num_frames(frames, target_num_frames):
    frame_count = len(frames)
    if frame_count < target_num_frames:
        extra = target_num_frames - frame_count
        frames.extend([frames[-1]] * extra)
    elif frame_count > target_num_frames:
        indices = np.linspace(0, frame_count - 1, target_num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    return frames


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def resize_and_convert_to_gray(frames, resolution):
    width, height = resolution
    gray_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frames.append(frame)
    return np.stack(gray_frames, axis=0).astype(np.uint8)


def calculate_skimage_metrics(videos1, videos2, only_final=False):
    # videos: [timestamps, h, w], pixel values in [0, 255]
    assert videos1.shape == videos2.shape

    ssim_results = []
    psnr_results = []

    for clip_timestamp in range(len(videos1)):
        img1 = videos1[clip_timestamp]
        img2 = videos2[clip_timestamp]

        ssim_results.append(
            structural_similarity(
                img1,
                img2,
                data_range=255,
            )
        )
        psnr_results.append(peak_signal_noise_ratio(img1, img2, data_range=255))

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
    for clip_timestamp in range(videos1.shape[0]):
        ssim.append(ssim_results[clip_timestamp])
        ssim_std.append(0.0)
        psnr.append(psnr_results[clip_timestamp])
        psnr_std.append(0.0)

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
    pred_frames = read_video_frames(pred_video_path)

    target_video_path = os.path.join(path, folder, target_video_name)
    target_frames = read_video_frames(target_video_path)
    if not pred_frames or not target_frames:
        continue

    pred_num_frames = len(pred_frames)
    target_frames = adjust_num_frames(target_frames, pred_num_frames)

    target_height, target_width = target_frames[0].shape[:2]
    resolution = (target_width, target_height)
    pred_frames = resize_and_convert_to_gray(pred_frames, resolution)
    target_frames = resize_and_convert_to_gray(target_frames, resolution)

    if not verbose:
        print(
            f"pred_frames shape: {pred_frames.shape}, target_frames shape: {target_frames.shape}"
        )
        verbose = True

    metrics = calculate_skimage_metrics(pred_frames, target_frames, only_final=only_final)
    ssim = metrics["ssim"]
    ssim_res.append(ssim["value"][0])
    ssim_std.append(ssim["value_std"][0])
    psnr = metrics["psnr"]
    psnr_res.append(psnr["value"][0])
    psnr_std.append(psnr["value_std"][0])
    # lpips = calculate_lpips(videos1, videos2, "cuda", only_final=only_final)
    # lpips_res.append(lpips["value"][0])
    # lpips_std.append(lpips["value_std"][0])

    del pred_frames, target_frames

result = {}
ssim_res = np.array(ssim_res, dtype=np.float64)
psnr_res = np.array(psnr_res, dtype=np.float64)
# lpips_res = np.array(lpips_res)
ssim_std = np.array(ssim_std, dtype=np.float64)
psnr_std = np.array(psnr_std, dtype=np.float64)
# lpips_std = np.array(lpips_std)
result["ssim"] = np.float64(np.mean(ssim_res))
result["ssim_std"] = np.float64(np.mean(ssim_std))
result["psnr"] = np.float64(np.mean(psnr_res))
result["psnr_std"] = np.float64(np.mean(psnr_std))
# result["lpips"] = np.mean(lpips_res)
# result["lpips_std"] = np.mean(lpips_std)

print(result)

with open("ssim_psnr.json", "w") as f:
    json.dump(result, f, indent=4)
