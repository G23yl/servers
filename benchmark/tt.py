import os
import shutil
import random
import cv2
import numpy as np
from PIL import Image
from diffusers.utils.export_utils import export_to_video
import torch

# path = "/mnt/zhouxin-mnt/self_forcing/test_images"
# path1 = "/mnt/zhouxin-mnt/fractal20220817_data/processed"
# target_dir = "/mnt/zhouxin-mnt/self_forcing/tesser_test_images"

# i = 0

# for folder in os.listdir(path):
#     folder_path = os.path.join(path, folder)
#     target_folder_path = os.path.join(target_dir, str(i))
#     os.makedirs(target_folder_path, exist_ok=True)
#     shutil.copyfile(os.path.join(folder_path, "target.mp4"), os.path.join(target_folder_path, "rgb.mp4"))
#     shutil.copyfile(os.path.join(folder_path, "ins.txt"), os.path.join(target_folder_path, "ins.txt"))
#     i += 1

# folders = os.listdir(path1)
# idxs = list(range(len(folders)))
# random.seed(42)
# random.shuffle(idxs)
# test_idxs = idxs[:200]
# test_folders = [folders[i] for i in test_idxs]

# for folder in test_folders:
#     folder_path = os.path.join(path1, folder)
#     target_folder_path = os.path.join(target_dir, str(i))
#     os.makedirs(target_folder_path, exist_ok=True)
#     shutil.copyfile(os.path.join(folder_path, "video", "rgb.mp4"), os.path.join(target_folder_path, "rgb.mp4"))
#     shutil.copyfile(os.path.join(folder_path, "instruction.txt"), os.path.join(target_folder_path, "ins.txt"))
#     with open(os.path.join(target_folder_path, "ins.txt"), "a") as f:
#         f.write(" google robot")
#     i += 1

# path = "/mnt/zhouxin-mnt/self_forcing/tesser_test_images"
# target_dir = "./test_images"
# for folder in os.listdir(path):
#     folder_path = os.path.join(path, folder, "rgb.mp4")
#     cap = cv2.VideoCapture(str(folder_path))
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame)
#     cap.release()
#     first_image = frames[0]
#     cv2.imwrite(os.path.join(target_dir, f"{folder}.png"), cv2.cvtColor(first_image, cv2.COLOR_RGB2BGR))

# target_path = "/mnt/zhouxin-mnt/self_forcing/tesser_test_images"
# path = "./test_images"
# for file in os.listdir(path):
#     if "depth" in file or "normal" in file:
#         folder = file.split("_")[0]
#         folder_path = os.path.join(target_path, folder)
#         shutil.copyfile(
#             os.path.join(path, file),
#             os.path.join(
#                 folder_path,
#                 "image_depth.npy" if "depth" in file else "image_normal.png",
#             ),
#         )
#     else:
#         folder = file.split(".")[0]
#         folder_path = os.path.join(target_path, folder)
#         shutil.copyfile(
#             os.path.join(path, file), os.path.join(folder_path, "image.png")
#         )

# for file in os.listdir("./test_images"):
#     if "depth" in file:
#         d = np.load(os.path.join("./test_images", file))
#         print(file, d.shape, d.min(), d.max())

# def crop_and_resize_frames(frames, target_size, interpolation="bilinear"):
#     # frames: [F, H, W, C]
#     target_height, target_width = target_size
#     original_height, original_width = frames[0].shape[:2]
#     if original_height == target_height and original_width == target_width:
#         return [frame for frame in frames]

#     # ==== interpolation method ====
#     if interpolation == "bilinear":
#         interpolation = cv2.INTER_LINEAR
#     elif interpolation == "nearest":
#         interpolation = cv2.INTER_NEAREST
#     else:
#         interpolation = cv2.INTER_LINEAR

#     processed_frames = []
#     for frame in frames:
#         original_height, original_width = frame.shape[:2]
#         aspect_ratio_target = target_width / target_height
#         aspect_ratio_original = original_width / original_height

#         if aspect_ratio_original > aspect_ratio_target:
#             new_width = int(aspect_ratio_target * original_height)
#             start_x = (original_width - new_width) // 2
#             cropped_frame = frame[:, start_x : start_x + new_width]
#         else:
#             new_height = int(original_width / aspect_ratio_target)
#             start_y = (original_height - new_height) // 2
#             cropped_frame = frame[start_y : start_y + new_height, :]
#         resized_frame = cv2.resize(
#             cropped_frame, (target_width, target_height), interpolation=interpolation
#         )
#         print(resized_frame.shape)
#         processed_frames.append(resized_frame)

#     return processed_frames

# d = np.load("/mnt/zhouxin-mnt/self_forcing/tesser_test_images/279/image_depth.npy")
# d = d.repeat(3, axis=-1)
# d = crop_and_resize_frames([d], (480, 640))[0]
# d = np.array(d)
# print(d.shape)

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

path1 = "/mnt/zhouxin-mnt/self_forcing/test_images/10033/target.mp4"
path2 = "/mnt/zhouxin-mnt/self_forcing/test_images/10033/new_pred_rgb_10000.mp4"

cap1 = cv2.VideoCapture(path1)
frames1 = []
while True:
    ret, frame = cap1.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames1.append(frame)
cap1.release()

cap2 = cv2.VideoCapture(path2)
frames2 = []
while True:
    ret, frame = cap2.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames2.append(frame)
cap2.release()

frames1 = adjust_num_frames(frames1, len(frames2))
frames1 = crop_and_resize_frames(frames1, (frames2[0].shape[0], frames2[0].shape[1]))
# frames1 = [Image.fromarray(f).resize((frames2[0].shape[1], frames2[0].shape[0])) for f in frames1]
# frames1 = [np.array(f) for f in frames1]

frames1, frames2 = np.stack(frames1, axis=0), np.stack(frames2, axis=0)
comb = np.concat([frames1, frames2], axis=2) / 255.0

export_to_video(comb, "./comb1.mp4", fps=8)
