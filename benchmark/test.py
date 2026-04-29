from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
import torch
from PIL import Image
from decord import VideoReader
import numpy as np
import cv2

def adjust_num_frames(frames, target_num_frames):
    """
    Adjust number of frames. Return the same type of input frames
    """
    frame_count = len(frames)
    if frame_count < target_num_frames:
        extra = target_num_frames - frame_count
        if isinstance(frames, list):
            frames.extend([frames[-1]] * extra)
        elif isinstance(frames, torch.Tensor):
            frame_to_add = [frames[-1]] * extra
            frames = [f for f in frames] + frame_to_add
            frames = torch.stack(frames)
        elif isinstance(frames, np.ndarray):
            frame_to_add = [frames[-1]] * extra
            frames = [f for f in frames] + frame_to_add
            frames = np.stack(frames)
    elif frame_count > target_num_frames:
        indices = np.linspace(0, frame_count - 1, target_num_frames, dtype=int)
        frames1 = [frames[i] for i in indices]
        if isinstance(frames, torch.Tensor):
            frames = torch.stack(frames1)
        elif isinstance(frames, np.ndarray):
            frames = np.stack(frames1)
    return frames


def resize_frames(frames, target_height: int, target_width: int):
    if isinstance(frames, list):
        frames = [
            cv2.resize(f, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            for f in frames
        ]
    elif isinstance(frames, np.ndarray):
        frames = [
            cv2.resize(f, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            for f in frames
        ]
        frames = np.stack(frames)
    elif isinstance(frames, torch.Tensor):
        frames = [
            cv2.resize(
                f.numpy(), (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
            for f in frames
        ]
        frames = torch.tensor(frames)
    return frames

if __name__ == "__main__":
    vid = VideoReader("/mnt/workspace/tusifan/Kairos_world_model/DriveGen/data_process/Video-Depth-Anything/outputs/target_rgb.mp4")
    vid1 = VideoReader("/mnt/workspace/tusifan/Kairos_world_model/DriveGen/data_process/Video-Depth-Anything/outputs/val_rgb.mp4")
    video = vid.get_batch(list(range(len(vid)))).asnumpy()
    video1 = vid1.get_batch(list(range(len(vid1)))).asnumpy()
    pred_num_frames = len(video)
    target_num_frames = len(video1)
    pred_frames = video
    target_frames = video1
    if pred_num_frames > target_num_frames:
        pred_frames = adjust_num_frames(pred_frames, target_num_frames)
    elif pred_num_frames < target_num_frames:
        target_frames = adjust_num_frames(target_frames, pred_num_frames)
    target_frames = target_frames.transpose(0, 3, 1, 2)
    target_frames = (target_frames / 255.0).clip(0, 1)

    pred_frames = resize_frames(pred_frames, target_frames.shape[-2], target_frames.shape[-1])
    pred_frames = torch.from_numpy(pred_frames).permute(0, 3, 1, 2)
    pred_frames = (pred_frames / 255.0).clamp(0, 1)

    print(target_frames.shape, pred_frames.shape)

    videos1, videos2 = np.array([pred_frames]), np.array([target_frames])
    ssim = calculate_ssim(videos1, videos2, only_final=True)
    psnr = calculate_psnr(videos1, videos2, only_final=True)
    videos1, videos2 = torch.from_numpy(videos1).float(), torch.from_numpy(videos2).float()
    lpips = calculate_lpips(videos1, videos2, device="cuda", only_final=True)
    print(ssim)
    print(psnr)
    print(lpips)
