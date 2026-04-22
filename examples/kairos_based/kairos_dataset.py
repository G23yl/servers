import json
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import random
from loguru import logger

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

DATASET2ROBOT = {
    "fractal20220817_data": "google robot",
    "bridge": "Trossen WidowX 250 robot arm",
}


def get_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    # [F, H, W, C]
    frames = np.stack(frames)
    return frames


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


class KairosMultiModalDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        modals: List[str],
        samples_path: str,
        target_size: Tuple[int, int] = (480, 640),
        target_frames: int = 49,
    ):
        self.data_root = data_root
        self.modals_list = modals
        self.target_size = target_size
        self.target_frames = target_frames
        if not samples_path.endswith(".json"):
            raise ValueError(
                f"samples_path must be a json file but got: {samples_path}"
            )
        with open(samples_path, "r") as f:
            json_content = json.load(f)
        train_samples = json_content["train_samples"]
        samples = []  # {"path", "modal", "ins"}
        for sub in tqdm(train_samples):
            sub_path = sub
            if not os.path.exists(os.path.join(sub_path, "video", "rgb.mp4")):
                logger.info(f"rgb video not exist in {sub_path}, skip")
                continue
            ins_file = os.path.join(sub_path, "instruction.txt")
            if os.path.exists(ins_file):
                ins = Path(ins_file).read_text().strip(". ").lower()
            else:
                logger.info(f"instruction file not exist in {sub_path}, use empty text")
                ins = ""
            for m in modals:
                modal_path = os.path.join(sub_path, m, "npz", f"{m}.npz")
                if not os.path.exists(modal_path):
                    logger.info(f"{modal_path} not exist, skip")
                    continue
                d = {"path": sub_path, "modal": m, "ins": ins}
                samples.append(d)
        self.samples = samples
        # transform
        transform_list = [
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
        self.transforms = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.samples)

    def get_prompt(self, sample) -> Tuple[str, str]:
        ##NOTE need cfg - 10% drop
        if random.random() >= 0.1:
            ins = sample["ins"]
            modal_name = sample["modal"]
            path = sample["path"]
            if "bridge" in path:
                prompt = ins + f" {DATASET2ROBOT['bridge']}"
            elif "fractal20220817_data" in path:
                prompt = ins + f" {DATASET2ROBOT['fractal20220817_data']}"
            modal_prompt = f"[{modal_name.upper()}] {prompt}"
        else:
            prompt = ""
            modal_prompt = ""
        return prompt, modal_prompt

    def __getitem__(self, idx: int):
        try:
            s = self.samples[idx]
            # get modal data
            modal_path = os.path.join(s["path"], s["modal"], "npz", f"{s['modal']}.npz")
            d = np.load(modal_path)
            modal_array = d.get("arr_0", d).astype(np.float32)

            prompt, modal_prompt = self.get_prompt(s)
            # get rgb video
            video_path = os.path.join(s["path"], "video", "rgb.mp4")
            video = get_video(video_path)
            video = adjust_num_frames(video, self.target_frames)
            video = resize_frames(video, self.target_size[0], self.target_size[1])
            # [H, W, C] 0..255
            first_image = torch.from_numpy(deepcopy(video[0]))
            video = torch.from_numpy(video).permute(0, 3, 1, 2) # [F, C, H, W]
            # [C, F, H, W] -1..1
            video = self.transforms(video).permute(1, 0, 2, 3)
        except Exception as e:
            logger.debug(f"Error loading sample {idx} with path {s['path']}, fallback to default sample: {e}")
            # if occur error, fallback to the first sample
            s = self.samples[0]
            # get modal data
            modal_path = os.path.join(s["path"], s["modal"], "npz", f"{s['modal']}.npz")
            d = np.load(modal_path)
            modal_array = d.get("arr_0", d).astype(np.float32)

            prompt, modal_prompt = self.get_prompt(s)
            # get rgb video
            video_path = os.path.join(s["path"], "video", "rgb.mp4")
            video = get_video(video_path)
            video = adjust_num_frames(video, self.target_frames)
            video = resize_frames(video, self.target_size[0], self.target_size[1])
            # [H, W, C] 0..255
            first_image = torch.from_numpy(deepcopy(video[0]))
            video = torch.from_numpy(video).permute(0, 3, 1, 2) # [F, C, H, W]
            # [C, F, H, W] -1..1
            video = self.transforms(video).permute(1, 0, 2, 3)
        
        if s["modal"] == "depth":
            # normalize
            modal_max, modal_min = modal_array.max(), modal_array.min()
            modal_array = (modal_array - modal_min) / (
                modal_max - modal_min + (1e-8 if modal_max == modal_min else 0)
            )
            modal_array *= 255.0
            if len(modal_array.shape) == 3 or (len(modal_array.shape) == 4 and modal_array.shape[-1] == 1):
                # [F, H, W, C]
                modal_array = np.stack([modal_array] * 3, axis=-1)
            modal_array = adjust_num_frames(modal_array, self.target_frames)
            modal_array = resize_frames(
                modal_array, self.target_size[0], self.target_size[1]
            )
            # [H, W, C] 0..255
            first_modal = torch.from_numpy(deepcopy(modal_array[0]))
            modal_array = torch.from_numpy(modal_array).permute(0, 3, 1, 2)
            # [C, F, H, W] -1..1
            modal_array = self.transforms(modal_array).permute(1, 0, 2, 3)
        elif s["modal"] == "flow":
            ## NOTE optical_flow是不是不合理啊？推理的时候谁能给出第一帧图片的光流图啊？除非把modal branch从i2v变成t2v，这样好像也挺合理的，看examples/kairos_based/modules/kairos_model_modal_t2v.py的实现
            modal_array = adjust_num_frames(modal_array, self.target_frames)
            modal_array = resize_frames(
                modal_array, self.target_size[0], self.target_size[1]
            )
            # [H, W, C] 0..255
            first_modal = torch.from_numpy(deepcopy(modal_array[0]))
            modal_array = torch.from_numpy(modal_array).permute(0, 3, 1, 2)
            # [C, F, H, W] -1..1
            modal_array = self.transforms(modal_array).permute(1, 0, 2, 3)
        elif s["modal"] == "normal":
            raise NotImplementedError("Normal modal not implemented!")
        else:
            raise ValueError(f"Not support modal: {s['modal']} yet!")

        return {
            "prompt": prompt,
            "modal_prompt": modal_prompt,
            "first_image": first_image,
            "first_modal": first_modal,
            "video": video,
            "modal": modal_array,
        }


class DummyKairosDataset(Dataset):
    def __init__(self):
        self.samples = [
            {
                "prompt": "move the can to the left of table",
                "modal_prompt": "[Depth] move the can to the left of table",
                "video": torch.randn(3, 49, 480, 640),
                "modal": torch.randn(3, 49, 480, 640),
                "first_image": torch.randn(480, 640, 3),
                "first_modal": torch.randn(480, 640, 3),
            },
            {
                "prompt": "pick up the apple on the table",
                "modal_prompt": "[Flow] pick up the apple on the table",
                "video": torch.randn(3, 49, 480, 640),
                "modal": torch.randn(3, 49, 480, 640),
                "first_image": torch.randn(480, 640, 3),
                "first_modal": torch.randn(480, 640, 3),
            },
            {
                "prompt": "pick up the apple on the table",
                "modal_prompt": "[Flow] pick up the apple on the table",
                "video": torch.randn(3, 49, 480, 640),
                "modal": torch.randn(3, 49, 480, 640),
                "first_image": torch.randn(480, 640, 3),
                "first_modal": torch.randn(480, 640, 3),
            },
            {
                "prompt": "pick up the apple on the table",
                "modal_prompt": "[Flow] pick up the apple on the table",
                "video": torch.randn(3, 49, 480, 640),
                "modal": torch.randn(3, 49, 480, 640),
                "first_image": torch.randn(480, 640, 3),
                "first_modal": torch.randn(480, 640, 3),
            },
            {
                "prompt": "pick up the apple on the table",
                "modal_prompt": "[Flow] pick up the apple on the table",
                "video": torch.randn(3, 49, 480, 640),
                "modal": torch.randn(3, 49, 480, 640),
                "first_image": torch.randn(480, 640, 3),
                "first_modal": torch.randn(480, 640, 3),
            },
            {
                "prompt": "pick up the apple on the table",
                "modal_prompt": "[Flow] pick up the apple on the table",
                "video": torch.randn(3, 49, 480, 640),
                "modal": torch.randn(3, 49, 480, 640),
                "first_image": torch.randn(480, 640, 3),
                "first_modal": torch.randn(480, 640, 3),
            },
            {
                "prompt": "pick up the apple on the table",
                "modal_prompt": "[Flow] pick up the apple on the table",
                "video": torch.randn(3, 49, 480, 640),
                "modal": torch.randn(3, 49, 480, 640),
                "first_image": torch.randn(480, 640, 3),
                "first_modal": torch.randn(480, 640, 3),
            },
            {
                "prompt": "pick up the apple on the table",
                "modal_prompt": "[Flow] pick up the apple on the table",
                "video": torch.randn(3, 49, 480, 640),
                "modal": torch.randn(3, 49, 480, 640),
                "first_image": torch.randn(480, 640, 3),
                "first_modal": torch.randn(480, 640, 3),
            },
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


if __name__ == "__main__":
    d = KairosMultiModalDataset(
        data_root="/mnt/zhouxin-mnt",
        modals=["depth", "flow"],
        samples_path="/home/zhouxin/data/DriveGen/UniAnimate-DiT/f_samples.json",
    )
    print(f"dataset length: {len(d)}")

    s = d[110000]
    print(
        f"prompt: {s['prompt']}, modal_prompt: {s['modal_prompt']}, first_image/modal shape: {s['first_image'].shape}/{s['first_modal'].shape}, first_image/modal min/max: {s['first_image'].min()},{s['first_image'].max()}/{s['first_modal'].min()},{s['first_modal'].max()}, video/modal shape: {s['video'].shape}/{s['modal'].shape}, video/modal min/max: {s['video'].min()},{s['video'].max()}/{s['modal'].min()},{s['modal'].max()}"
    )
    s = d[132001]
    print(
        f"prompt: {s['prompt']}, modal_prompt: {s['modal_prompt']}, first_image/modal shape: {s['first_image'].shape}/{s['first_modal'].shape}, first_image/modal min/max: {s['first_image'].min()},{s['first_image'].max()}/{s['first_modal'].min()},{s['first_modal'].max()}, video/modal shape: {s['video'].shape}/{s['modal'].shape}, video/modal min/max: {s['video'].min()},{s['video'].max()}/{s['modal'].min()},{s['modal'].max()}"
    )

    # path = "/mnt/zhouxin-mnt/bridge"
    # DEBUG = True
    # tot = 0

    # with open("/home/zhouxin/data/DriveGen/UniAnimate-DiT/samples.json", "r") as f:
    #     json_content = json.load(f)
    # train_samples = json_content["train_samples"]
    # sub_train_samples = [path.split("/")[-3] for path in train_samples[:6000]]

    # with open("./need.txt", "w") as f:
    #     for folder in tqdm(sub_train_samples):
    #         mask_path = os.path.join(path, folder, "video", "mask_rgb.mp4")
    #         if os.path.exists(mask_path):
    #             tot += 1
    #             save_folder = os.path.join(path, folder, "mask", "npz")
    #             os.makedirs(save_folder, exist_ok=True)
    #             video = get_video(mask_path)
    #             video = np.stack(video) # [F, H, W, C]
    #             if DEBUG:
    #                 print(f"shape: {video.shape}")
    #                 DEBUG = False
    #             np.savez_compressed(os.path.join(save_folder, "mask.npz"), video)
    #             os.remove(mask_path)
    #         else:
    #             if os.path.exists(os.path.join(path, folder, "mask", "npz", "mask.npz")):
    #                 tot += 1
    #             else:
    #                 f.write(f"{folder}\n")
    # print(f"Total: {tot}")
