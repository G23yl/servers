import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import cv2
import jsonlines
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import logging

import os
import decord
import torch
from diffusers.utils.export_utils import export_to_video
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import gc

decord.bridge.set_bridge("torch")

logging.basicConfig(level=logging.INFO)

DATASET2ROBOT = {
    "fractal20220817_data": "google robot",
    "bridge": "Trossen WidowX 250 robot arm",
    "ssv2": "human hand",
    "rlbench": "Franka Emika Panda",
}
DATASET2RES = {
    "fractal20220817_data": (544, 960),
    # "fractal20220817_data_superres": (512, 640),
    "bridge": (544, 960),
    "rlbench": (512, 512),
    # common resolutions
    # "480p": (480, 854),
    # "720p": (720, 1280),
}
HEIGHT_BUCKETS = [240, 256, 480, 720]
WIDTH_BUCKETS = [320, 426, 640, 854, 1280]
FRAME_BUCKETS = [9, 49, 100]


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
        logging.warning(
            f"Unsupported interpolation: {interpolation}. Using bilinear instead."
        )

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


class RoboDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 81,
        id_token: Optional[str] = None,
        height_buckets: List[int] | None = None,
        width_buckets: List[int] | None = None,
        frame_buckets: List[int] | None = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.resolutions = [
            (f, h, w)
            for h in self.height_buckets
            for w in self.width_buckets
            for f in self.frame_buckets
        ]

        self._init_transforms()
        self._load_samples()

    def _init_transforms(self):
        """Initialize video transforms based on class requirements"""
        transform_list = [
            transforms.Lambda(self.identity_transform),
            transforms.Lambda(self.scale_transform),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]

        if self.random_flip:
            transform_list.insert(0, transforms.RandomHorizontalFlip(self.random_flip))

        self.video_transforms = transforms.Compose(transform_list)

    def _load_samples(self):
        """Load samples from dataset file or local paths"""
        if self.dataset_file is None or not Path(self.dataset_file).exists():
            self.samples, test_samples = self._load_openx_dataset_from_local_path(
                "bridge"
            )
            logging.info(
                f"Loaded {len(self.samples)} train and {len(test_samples)} test samples from Bridge dataset."
            )

            # Save samples to dataset file
            random.shuffle(self.samples)
            with open(self.dataset_file, "w") as f:
                json.dump(self.samples, f)
            with open(self.dataset_file.replace(".json", "_test.json"), "w") as f:
                json.dump(test_samples, f)
        else:
            with open(self.dataset_file, "r") as f:
                self.samples = json.load(f)
            self._get_rlbench_instructions()

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for i in range(3):
            try:
                return self.getitem(index)
            except Exception as e:
                logging.error(f"Error loading sample {self.samples[index][1]}: {e}")
                index = random.randint(0, len(self.samples) - 1)
        return self.getitem(index)

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Special logic for bucket sampler
            return index

        if self.load_tensors:
            raise NotImplementedError("Loading tensors is not supported.")

        sample = self.samples[index]
        image, video = self._preprocess_video(Path(sample[1]))
        instruction = self.get_instruction(index)

        return {
            "prompt": self.id_token + instruction,
            "image": image,
            "video": video,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
        }

    def _train_test_split(
        self, samples: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        if len(samples) > 4000:
            test_size = 200
        else:
            test_size = max(1, int(len(samples) * 0.05))

        indices = list(range(len(samples)))
        random.shuffle(indices)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        test_samples = [samples[i] for i in test_indices]
        train_samples = [samples[i] for i in train_indices]

        return train_samples, test_samples

    def _load_openx_dataset_from_local_path(self, dataname: str):
        samples = []
        for subdir in tqdm((self.data_root / dataname).iterdir()):
            if subdir.is_dir():
                # skip when no depth data
                depth_dir = subdir.joinpath("depth", "npz", "depth.npz")
                if not depth_dir.exists():
                    continue

                rgb_dir = subdir.joinpath("video", "rgb.mp4")
                rgb_valid = rgb_dir.exists()
                if not rgb_dir.exists():
                    rgb_valid = False
                    # rgb_dir = subdir.joinpath("image", "rgb")
                    # rgb_valid = rgb_dir.exists() and any(rgb_dir.glob("*.png"))
                if rgb_valid:
                    # Load prompt from instruction.txt if available
                    instruction_file = subdir.joinpath("instruction.txt")
                    if instruction_file.is_file():
                        instruction = instruction_file.read_text().strip()
                    else:
                        instruction = "null"
                    # path str
                    samples.append([instruction, str(rgb_dir)])
                    # if len(samples) >= 10000:
                    #     break

        ## split train and test samples. Only retain 200 samples for testing
        test_size = 200
        idxs = list(range(len(samples)))
        random.seed(42)
        random.shuffle(idxs)
        test_idxs = idxs[:test_size]
        train_idxs = idxs[test_size:]
        test_samples = [samples[i] for i in test_idxs]
        train_samples = [samples[i] for i in train_idxs]
        return train_samples, test_samples

    def _load_ssv2_dataset_from_local_path(self) -> Tuple[List[str], List[Path]]:
        labels_file = Path("data/ssv2/labels/train.json")
        video_root = Path("data/ssv2/20bn-something-something-v2")
        with labels_file.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        samples = []
        for entry in labels:
            video_id = entry.get("id")
            label = entry.get("label", "null")
            video_path = video_root / f"{video_id}.webm"
            samples.append([label, str(video_path)])

        train_samples, test_samples = self._train_test_split(samples)
        return train_samples, test_samples

    def _get_rlbench_instructions(self) -> List[str]:
        self.rlbench_instructions = {}
        taskvar_json = Path("data/rlbench/taskvar_instructions.jsonl")
        if not taskvar_json.exists():
            logging.warning(f"Taskvar json {taskvar_json} does not exist.")
            return
        with jsonlines.open(taskvar_json, "r") as reader:
            for obj in reader:
                task = obj["task"]
                self.rlbench_instructions.setdefault(task, obj["variations"]["0"])

    def _load_rlbench_dataset_from_local_path(self) -> List[List[str]]:
        rlbench_path = Path("data/rlbench/train_dataset/microsteps/seed100")

        self._get_rlbench_instructions()

        samples = [
            [task_dir.name, str(rgb_path)]
            for task_dir in rlbench_path.iterdir()
            for episode_dir in task_dir.glob("variation0/episodes/*")
            for rgb_path in episode_dir.glob("video/*rgb.mp4")
        ]
        # find which path don't have video
        for task_dir in rlbench_path.iterdir():
            for episode_dir in task_dir.glob("variation0/episodes/*"):
                rgb_path = episode_dir.glob("video/*rgb.mp4")
                if not rgb_path:
                    print(f"Missing video: {episode_dir}")
        train_samples, test_samples = self._train_test_split(samples)
        return train_samples, test_samples

    def _adjust_num_frames(self, frames, target_num_frames=None):
        if target_num_frames is None:
            target_num_frames = self.max_num_frames
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

    def get_instruction(self, index: int) -> str:
        sample = self.samples[index]
        instruction = sample[0].lower()
        path = sample[1]

        if "rlbench" in str(path):
            task_name = path.split("/")[5]
            instruction = (
                random.choice(self.rlbench_instructions[task_name])
                + f" {DATASET2ROBOT['rlbench']}"
            )
        elif "fractal20220817_data" in str(path):
            instruction += f" {DATASET2ROBOT['fractal20220817_data']}"
        elif "bridge" in str(path):
            instruction += f" {DATASET2ROBOT['bridge']}"
        elif "ssv2" in str(path):
            instruction += f" {DATASET2ROBOT['ssv2']}"
        else:
            raise ValueError(f"Unknown dataset for path: {path}")

        return instruction

    def _read_rgb_data(self, path: Path) -> torch.Tensor:
        if path.is_dir():
            frames = self._read_video_from_dir(path, adjust_num_frames=False)
        elif path.suffix == ".webm" or path.suffix == ".mp4":
            frames = self._read_video_from_webm(path, adjust_num_frames=False)
        else:
            raise ValueError(f"Unsupported video format: {path}")
        return frames

    def _preprocess_video(
        self, path: Path
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Loads a single video from either:
        - A directory of RGB frames, or
        - A single .webm video file.

        Returns:
            image: the first frame as an image if image_to_video=True, else None
            video: a tensor [F, C, H, W] of frames
            None for embeddings if load_tensors=False
        """
        if path.is_dir():
            frames = self._read_video_from_dir(path)
        elif path.suffix == ".webm" or path.suffix == ".mp4":
            frames = self._read_video_from_webm(path)
            if "ssv2" in str(path):
                frames = crop_and_resize_frames(frames, (256, 320))
        else:
            raise ValueError(f"Unsupported video format: {path}")
        # randome resize to other resolutions
        if random.random() < 0.2:
            target_size = random.choice(list(DATASET2RES.values()))
            frames = crop_and_resize_frames(frames, target_size)

        # transform frames to tensor
        frames = [
            self.video_transforms(torch.from_numpy(img).permute(2, 0, 1).float())
            for img in frames
        ]
        video = torch.stack(frames, dim=0)  # [F, C, H, W]
        image = video[:1].clone() if self.image_to_video else None

        return image, video

    def _read_video_from_dir(
        self, path: Path, adjust_num_frames: bool = True
    ) -> List[np.ndarray]:
        assert path.is_dir(), f"Path {path} is not a directory."
        frame_paths = sorted(list(path.glob("*.png")), key=lambda x: int(x.stem))
        if adjust_num_frames:
            frame_paths = self._adjust_num_frames(frame_paths)
        frames = []
        for fp in frame_paths:
            img = np.array(Image.open(fp).convert("RGB"))
            frames.append(img)
        return frames

    def _read_video_from_webm(
        self, path: Path, adjust_num_frames: bool = True
    ) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if adjust_num_frames:
            frames = self._adjust_num_frames(frames)
        return frames


class RoboDepth(RoboDataset):
    def __init__(
        self,
        data_root: str,
        is_train: bool,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] | None = None,
        width_buckets: List[int] | None = None,
        frame_buckets: List[int] | None = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ):
        self.is_train = is_train
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
        )

    def _load_samples(self):
        """Override to load additional datasets"""
        if self.dataset_file is None or not Path(self.dataset_file).exists():
            # bridge_train, bridge_test = self._load_openx_dataset_from_local_path(
            #     "bridge"
            # )
            # fractal_train, fractal_test = self._load_openx_dataset_from_local_path(
            #     "fractal20220817_data"
            # )
            ## use in external
            # self.test_samples = fractal_test + bridge_test
            # self.train_samples = fractal_train + bridge_train

            self.train_samples, self.test_samples = self._load_datasets(["bridge"])

            ## use self.samples internally
            if self.is_train:
                self.samples = self.train_samples[:6000]
            else:
                self.samples = self.test_samples
        else:
            pass

    def _load_datasets(self, datasets: List[Literal["bridge", "fractal"]]):
        train_samples = []
        test_samples = []
        for name in datasets:
            train, test = self._load_openx_dataset_from_local_path(name)
            train_samples += train
            test_samples += test
        return train_samples, test_samples

    def _read_depth_data(self, path: Path) -> torch.Tensor:
        """
        Reads a depth data file in .npz format and returns it as a [T, H, W] torch tensor.
        """
        assert path.is_file(), f"Depth file {path} does not exist."
        depth_array = np.load(path)["arr_0"].astype(np.float32)
        return depth_array

    def get_depth_data(
        self, rgb_dir, rgb_video, target_size
    ) -> Tuple[torch.Tensor, bool]:
        depth_path = Path(
            str(rgb_dir).replace("video", "depth/npz").replace("rgb.mp4", "depth.npz")
        )

        if depth_path.exists():
            depth_video = self._read_depth_data(depth_path)  # [T, H, W]
            depth_video = (depth_video - depth_video.min()) / (
                depth_video.max() - depth_video.min() + 1e-8
            )
            if "rlbench" in str(rgb_dir):
                depth_video = 1 - depth_video
            depth_video *= 255.0
            depth_video = np.stack([depth_video] * 3, axis=-1)  # [T, H, W, 3]
            depth_video = crop_and_resize_frames(depth_video, target_size)
            depth_video = [
                self.video_transforms(
                    torch.from_numpy(img).permute(2, 0, 1).float()
                ).unsqueeze(0)
                for img in depth_video
            ]
            depth_video = torch.cat(depth_video, dim=0)  # [T, 3, H, W]
            if len(rgb_video) != len(depth_video):
                logging.warning(
                    f"{depth_path} RGB {len(rgb_video)} != DEPTH {len(depth_video)}"
                )
                depth_video = self._adjust_num_frames(depth_video, len(rgb_video))
                depth_video = torch.stack(depth_video, dim=0)  # [T, 3, H, W]
            have_depth = True
        else:
            depth_video = torch.zeros_like(rgb_video)
            have_depth = False
        return depth_video, have_depth

    def _preprocess_video(self, path: Path):
        """
        Overrides the parent class method to load both RGB and depth data and return a concatenated video.

        Returns:
            image: [C, 1, H, W]
            rgb_video: [C, F, H, W]
            depth_video: [C, F, H, W]
            have_depth: bool
        """
        target_size = (544, 960)

        # ==== Load RGB frames =====
        rgb_dir = path
        frames = self._read_rgb_data(rgb_dir)
        frames = crop_and_resize_frames(frames, target_size)
        rgb_video = [
            self.video_transforms(
                torch.from_numpy(img).permute(2, 0, 1).float()
            ).unsqueeze(0)
            for img in frames
        ]
        rgb_video = torch.cat(rgb_video, dim=0)  # [T, 3, H, W]

        # ==== Load depth data ====
        depth_video, have_depth = self.get_depth_data(rgb_dir, rgb_video, target_size)
        adjusted_frames, adjusted_depth = (
            self._adjust_num_frames(list(rgb_video)),
            self._adjust_num_frames(list(depth_video)),
        )
        rgb_video = torch.stack(adjusted_frames, dim=0)
        image = rgb_video[:1].clone()
        depth_video = torch.stack(adjusted_depth, dim=0)
        first_depth = depth_video[:1].clone()
        # fchw -> cfhw
        rgb_video, depth_video, image, first_depth = (
            rgb_video.permute(1, 0, 2, 3),
            depth_video.permute(1, 0, 2, 3),
            image.permute(1, 0, 2, 3),
            first_depth.permute(1, 0, 2, 3),
        )

        return image, first_depth, rgb_video, depth_video, have_depth

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Special logic for bucket sampler
            return {"index": index}

        if self.load_tensors:
            raise NotImplementedError("Loading tensors is not supported.")

        sample = self.samples[index]
        if self.is_train:
            ## if is_train, preprocess the data
            image, first_depth, video, depth, have_depth = self._preprocess_video(
                Path(sample[1])
            )
            instruction = self.get_instruction(index)

            ## already between [-1, 1], [c,f,h,w]
            return {
                "prompts": self.id_token + instruction,
                "first_image": image,
                "first_depth": first_depth,
                "frames": video,
                "depth": depth,
                "path": sample[1],
            }
        else:
            ## if testing, dont preprocess data and output raw directly to save (rgb_dir)
            return {"path": sample[1]}


class MiniTrajRoboDepth(RoboDataset):
    def __init__(
        self,
        data_root: str,
        is_train: bool,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] | None = None,
        width_buckets: List[int] | None = None,
        frame_buckets: List[int] | None = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ):
        self.is_train = is_train
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
        )

    def _load_samples(self):
        """Override to load additional datasets"""
        if self.dataset_file is None or not Path(self.dataset_file).exists():
            self.train_samples, self.test_samples = self._load_datasets(["bridge"])

            ## use self.samples internally
            if self.is_train:
                self.samples = self.train_samples[:6000]
            else:
                self.samples = self.test_samples
        else:
            pass

    def _load_datasets(self, datasets: List[Literal["bridge", "fractal"]]):
        train_samples = []
        test_samples = []
        bridge_samples_json_path = (
            "/home/zhouxin/data/DriveGen/UniAnimate-DiT/samples.json"
        )
        with open(bridge_samples_json_path, "r") as f:
            samples = json.load(f)
        folders = [s.split("/")[-3] for s in samples["train_samples"]]
        for name in datasets:
            if name != "bridge":
                continue
            # for folder in os.listdir(str(self.data_root.joinpath(name))):
            for folder in folders:
                folder_path = self.data_root.joinpath(name, folder)
                track_path = folder_path.joinpath("video", "mask_rgb.mp4")
                depth_path = folder_path.joinpath("depth", "npz", "depth.npz")
                # only track and depth exists both
                if track_path.exists() and depth_path.exists():
                    rgb_path = folder_path.joinpath("video", "rgb.mp4")
                    ins_path = folder_path.joinpath("instruction.txt")
                    if ins_path.is_file():
                        instruction = ins_path.read_text().strip()
                    else:
                        instruction = "null"
                    train_samples.append((instruction, str(rgb_path)))
        # 68752
        return train_samples, test_samples

    def get_track_video(self, rgb_path: Path, target_size):
        track_path = rgb_path.with_name("mask_rgb.mp4")
        cap = cv2.VideoCapture(str(track_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames = crop_and_resize_frames(frames, target_size)
        # [F, C, H, W]
        frames = [
            self.video_transforms(
                torch.from_numpy(img).permute(2, 0, 1).float()
            ).unsqueeze(0)
            for img in frames
        ]
        frames = torch.cat(frames, dim=0)
        return frames

    def _read_depth_data(self, path: Path) -> torch.Tensor:
        """
        Reads a depth data file in .npz format and returns it as a [T, H, W] torch tensor.
        """
        assert path.is_file(), f"Depth file {path} does not exist."
        depth_array = np.load(path)["arr_0"].astype(np.float32)
        return depth_array

    def get_depth_data(
        self, rgb_dir, rgb_video, target_size
    ) -> Tuple[torch.Tensor, bool]:
        depth_path = Path(
            str(rgb_dir).replace("video", "depth/npz").replace("rgb.mp4", "depth.npz")
        )

        if depth_path.exists():
            depth_video = self._read_depth_data(depth_path)  # [T, H, W]
            depth_video = (depth_video - depth_video.min()) / (
                depth_video.max() - depth_video.min() + 1e-8
            )
            if "rlbench" in str(rgb_dir):
                depth_video = 1 - depth_video
            depth_video *= 255.0
            depth_video = np.stack([depth_video] * 3, axis=-1)  # [T, H, W, 3]
            depth_video = crop_and_resize_frames(depth_video, target_size)
            depth_video = [
                self.video_transforms(
                    torch.from_numpy(img).permute(2, 0, 1).float()
                ).unsqueeze(0)
                for img in depth_video
            ]
            depth_video = torch.cat(depth_video, dim=0)  # [T, 3, H, W]
            if len(rgb_video) != len(depth_video):
                logging.warning(
                    f"{depth_path} RGB {len(rgb_video)} != DEPTH {len(depth_video)}"
                )
                depth_video = self._adjust_num_frames(depth_video, len(rgb_video))
                depth_video = torch.stack(depth_video, dim=0)  # [T, 3, H, W]
            have_depth = True
        else:
            depth_video = torch.zeros_like(rgb_video)
            have_depth = False
        return depth_video, have_depth

    def _preprocess_video(self, path: Path):
        """
        Overrides the parent class method to load both RGB and depth data and return a concatenated video.

        Returns:
            image: [C, 1, H, W]
            rgb_video: [C, F, H, W]
            depth_video: [C, F, H, W]
            flow_video: [C, F, H, W]
        """
        target_size = (544, 960)

        # ==== Load RGB frames =====
        rgb_dir = path
        frames = self._read_rgb_data(rgb_dir)
        frames = crop_and_resize_frames(frames, target_size)
        rgb_video = [
            self.video_transforms(
                torch.from_numpy(img).permute(2, 0, 1).float()
            ).unsqueeze(0)
            for img in frames
        ]
        rgb_video = torch.cat(rgb_video, dim=0)  # [T, 3, H, W]

        # ==== Load depth data ====
        depth_video, _ = self.get_depth_data(rgb_dir, rgb_video, target_size)

        # ==== Load track data ====
        track_video = self.get_track_video(rgb_dir, target_size)

        adjusted_frames, adjusted_depth, adjusted_track = (
            self._adjust_num_frames(list(rgb_video)),
            self._adjust_num_frames(list(depth_video)),
            self._adjust_num_frames(list(track_video)),
        )
        track_video = torch.stack(adjusted_track, dim=0)
        first_track = track_video[:1].clone()
        rgb_video = torch.stack(adjusted_frames, dim=0)
        image = rgb_video[:1].clone()
        depth_video = torch.stack(adjusted_depth, dim=0)
        first_depth = depth_video[:1].clone()
        # fchw -> cfhw
        rgb_video, depth_video, track_video, image, first_depth, first_track = (
            rgb_video.permute(1, 0, 2, 3),
            depth_video.permute(1, 0, 2, 3),
            track_video.permute(1, 0, 2, 3),
            image.permute(1, 0, 2, 3),
            first_depth.permute(1, 0, 2, 3),
            first_track.permute(1, 0, 2, 3),
        )

        return image, first_depth, first_track, rgb_video, depth_video, track_video

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Special logic for bucket sampler
            return {"index": index}

        if self.load_tensors:
            raise NotImplementedError("Loading tensors is not supported.")

        sample = self.samples[index]
        if self.is_train:
            ## if is_train, preprocess the data
            image, first_depth, first_track, video, depth, track = (
                self._preprocess_video(Path(sample[1]))
            )
            instruction = self.get_instruction(index)

            ## already between [-1, 1], [c,f,h,w]
            return {
                "prompts": self.id_token + instruction,
                "first_image": image,
                "first_depth": first_depth,
                "frames": video,
                "depth": depth,
                "track": track,
                "first_track": first_track,
                "path": sample[1],
            }
        else:
            ## if testing, dont preprocess data and output raw directly to save (rgb_dir)
            return {"path": sample[1]}


class MiniFlowRoboDepth(RoboDataset):
    def __init__(
        self,
        data_root: str,
        is_train: bool,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] | None = None,
        width_buckets: List[int] | None = None,
        frame_buckets: List[int] | None = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ):
        self.is_train = is_train
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
        )

    def _load_samples(self):
        """Override to load additional datasets"""
        if self.dataset_file is None or not Path(self.dataset_file).exists():
            self.train_samples, self.test_samples = self._load_datasets(["bridge"])

            ## use self.samples internally
            if self.is_train:
                self.samples = self.train_samples[:6000]
            else:
                self.samples = self.test_samples
        else:
            pass

    def _load_datasets(self, datasets: List[Literal["bridge", "fractal"]]):
        train_samples = []
        test_samples = []
        for name in datasets:
            # mini_folders = os.listdir(str(self.data_root.joinpath(name)))[:200]
            for folder in os.listdir(str(self.data_root.joinpath(name))):
                folder_path = self.data_root.joinpath(name, folder)
                flow_path = folder_path.joinpath("video", "flow.mp4")
                depth_path = folder_path.joinpath("depth", "npz", "depth.npz")
                # only flow and depth exists both
                if flow_path.exists() and depth_path.exists():
                    rgb_path = folder_path.joinpath("video", "rgb.mp4")
                    ins_path = folder_path.joinpath("instruction.txt")
                    if ins_path.is_file():
                        instruction = ins_path.read_text().strip()
                    else:
                        instruction = "null"
                    train_samples.append((instruction, str(rgb_path)))
        return train_samples, test_samples

    def get_track_video(self, rgb_path: Path, target_size):
        track_path = rgb_path.with_name("track_rgb.mp4")
        cap = cv2.VideoCapture(str(track_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames = crop_and_resize_frames(frames, target_size)
        # [F, H, W]
        masks = [torch.from_numpy(f[..., 0] != 0) for f in frames]
        frames = [
            self.video_transforms(
                torch.from_numpy(img).permute(2, 0, 1).float()
            ).unsqueeze(0)
            for img in frames
        ]
        frames = torch.cat(frames, dim=0)
        masks = torch.stack(masks, dim=0)
        return frames, masks

    def _read_depth_data(self, path: Path) -> torch.Tensor:
        """
        Reads a depth data file in .npz format and returns it as a [T, H, W] torch tensor.
        """
        assert path.is_file(), f"Depth file {path} does not exist."
        depth_array = np.load(path)["arr_0"].astype(np.float32)
        return depth_array

    def get_depth_data(
        self, rgb_dir, rgb_video, target_size
    ) -> Tuple[torch.Tensor, bool]:
        depth_path = Path(
            str(rgb_dir).replace("video", "depth/npz").replace("rgb.mp4", "depth.npz")
        )

        if depth_path.exists():
            depth_video = self._read_depth_data(depth_path)  # [T, H, W]
            depth_video = (depth_video - depth_video.min()) / (
                depth_video.max() - depth_video.min() + 1e-8
            )
            if "rlbench" in str(rgb_dir):
                depth_video = 1 - depth_video
            depth_video *= 255.0
            depth_video = np.stack([depth_video] * 3, axis=-1)  # [T, H, W, 3]
            depth_video = crop_and_resize_frames(depth_video, target_size)
            depth_video = [
                self.video_transforms(
                    torch.from_numpy(img).permute(2, 0, 1).float()
                ).unsqueeze(0)
                for img in depth_video
            ]
            depth_video = torch.cat(depth_video, dim=0)  # [T, 3, H, W]
            if len(rgb_video) != len(depth_video):
                logging.warning(
                    f"{depth_path} RGB {len(rgb_video)} != DEPTH {len(depth_video)}"
                )
                depth_video = self._adjust_num_frames(depth_video, len(rgb_video))
                depth_video = torch.stack(depth_video, dim=0)  # [T, 3, H, W]
            have_depth = True
        else:
            depth_video = torch.zeros_like(rgb_video)
            have_depth = False
        return depth_video, have_depth

    def get_flow_data(self, rgb_dir, target_size):
        flow_path = Path(str(rgb_dir).replace("rgb.mp4", "flow.mp4"))
        flow_frames = self._read_rgb_data(flow_path)
        H, W = flow_frames[0].shape[:2]
        assert (H, W) == target_size, f"Flow size {(H, W)} != target size {target_size}"
        flow_video = [
            self.video_transforms(
                torch.from_numpy(img).permute(2, 0, 1).float()
            ).unsqueeze(0)
            for img in flow_frames
        ]
        flow_video = torch.cat(flow_video, dim=0)  # [T, 3, H, W]
        return flow_video

    def _preprocess_video(self, path: Path):
        """
        Overrides the parent class method to load both RGB and depth data and return a concatenated video.

        Returns:
            image: [C, 1, H, W]
            rgb_video: [C, F, H, W]
            depth_video: [C, F, H, W]
            flow_video: [C, F, H, W]
        """
        target_size = (544, 960)

        # ==== Load RGB frames =====
        rgb_dir = path
        frames = self._read_rgb_data(rgb_dir)
        frames = crop_and_resize_frames(frames, target_size)
        rgb_video = [
            self.video_transforms(
                torch.from_numpy(img).permute(2, 0, 1).float()
            ).unsqueeze(0)
            for img in frames
        ]
        rgb_video = torch.cat(rgb_video, dim=0)  # [T, 3, H, W]

        # ==== Load depth data ====
        depth_video, have_depth = self.get_depth_data(rgb_dir, rgb_video, target_size)

        # ==== Load flow data ====
        flow_video = self.get_flow_data(rgb_dir, target_size)

        adjusted_frames, adjusted_depth, adjusted_flow = (
            self._adjust_num_frames(list(rgb_video)),
            self._adjust_num_frames(list(depth_video)),
            self._adjust_num_frames(list(flow_video)),
        )
        rgb_video = torch.stack(adjusted_frames, dim=0)
        flow_video = torch.stack(adjusted_flow, dim=0)
        image = rgb_video[:1].clone()
        depth_video = torch.stack(adjusted_depth, dim=0)
        first_depth = depth_video[:1].clone()
        # fchw -> cfhw
        rgb_video, depth_video, flow_video, image, first_depth = (
            rgb_video.permute(1, 0, 2, 3),
            depth_video.permute(1, 0, 2, 3),
            flow_video.permute(1, 0, 2, 3),
            image.permute(1, 0, 2, 3),
            first_depth.permute(1, 0, 2, 3),
        )

        return image, first_depth, rgb_video, depth_video, flow_video

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Special logic for bucket sampler
            return {"index": index}

        if self.load_tensors:
            raise NotImplementedError("Loading tensors is not supported.")

        sample = self.samples[index]
        if self.is_train:
            ## if is_train, preprocess the data
            image, first_depth, video, depth, flow = self._preprocess_video(
                Path(sample[1])
            )
            instruction = self.get_instruction(index)

            ## already between [-1, 1], [c,f,h,w]
            return {
                "prompts": self.id_token + instruction,
                "first_image": image,
                "first_depth": first_depth,
                "frames": video,
                "depth": depth,
                "flow": flow,
                "path": sample[1],
            }
        else:
            ## if testing, dont preprocess data and output raw directly to save (rgb_dir)
            return {"path": sample[1]}


if __name__ == "__main__":
    import os
    import shutil
    from torch.utils.data import DataLoader

    def script1():
        ## save first image
        numbers = ["923", "1000", "1500"]
        for number in numbers:
            path = Path(f"/mnt/zhouxin-mnt/bridge/{number}")

            depth_path = path / "depth/npz/depth.npz"
            video_path = path / "video/rgb.mp4"
            ins_path = path / "instruction.txt"

            with open(ins_path, "r") as f:
                cont = f.read()
                cont += f" {DATASET2ROBOT['bridge']}"
                print(f"{number}: {cont}")

            depth_array = np.load(depth_path)["arr_0"].astype(np.float32)
            depth_array = (depth_array - depth_array.min()) / (
                depth_array.max() - depth_array.min() + 1e-8
            )
            first_depth = depth_array[0]  # [H, W]
            first_depth *= 255.0
            first_depth = np.stack([first_depth] * 3, axis=-1)
            first_depth = cv2.cvtColor(first_depth, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"data/input/first_depth_{number}.png", first_depth)

            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            first_frame = frames[0]  # [H, W, C]
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"data/input/first_image_{number}.png", first_frame)

    def script2():
        path = "/mnt/zhouxin-mnt/bridge"
        out_path = "/mnt/zhouxin-mnt/self_forcing/test_images1"
        sample_path = "/home/zhouxin/data/DriveGen/UniAnimate-DiT/samples.json"
        with open(sample_path, "r") as f:
            samples = json.load(f)
        test_samples = samples["test_samples"]
        folders = [s.split("/")[-3] for s in test_samples]

        for folder in folders:
            out_folder = os.path.join(out_path, folder)
            os.makedirs(out_folder, exist_ok=True)
            video_path = os.path.join(path, folder, "video", "rgb.mp4")
            ins_path = os.path.join(path, folder, "instruction.txt")
            with open(ins_path, "r") as f:
                cont = f.read()
            cont += f" {DATASET2ROBOT['bridge']}"
            with open(os.path.join(out_folder, "ins.txt"), "w") as f:
                f.write(cont)

            depth_path = (
                str(video_path)
                .replace("video", "depth/npz")
                .replace("rgb.mp4", "depth.npz")
            )
            depth_array = np.load(depth_path)["arr_0"].astype(np.float32)
            depth_array = (depth_array - depth_array.min()) / (
                depth_array.max() - depth_array.min() + 1e-8
            )
            depth_array *= 255.0
            depth_array = np.stack([depth_array] * 3, axis=-1)
            first_depth = depth_array[0]
            first_depth = cv2.cvtColor(first_depth, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_folder, "depth.png"), first_depth)
            shutil.copyfile(depth_path, os.path.join(out_folder, "target_depth.npz"))

            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            first_frame = frames[0]  # [H, W, C]
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_folder, "image.png"), first_frame)
            shutil.copyfile(video_path, os.path.join(out_folder, "target.mp4"))

    def script3():
        ## confirm dataset
        dataset = RoboDepth(
            data_root="/mnt/zhouxin-mnt", is_train=True, max_num_frames=49
        )
        train_samples = dataset.train_samples
        test_samples = dataset.test_samples
        info = {
            "train_samples": [s[1] for s in train_samples],
            "test_samples": [s[1] for s in test_samples],
        }
        with open("./samples.json", "w") as f:
            json.dump(info, f, indent=4)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        # print(
        #     f"dataset length: {len(dataloader)}\ntrain_samples length: {len(dataset.train_samples)}\ntest_samples length: {len(dataset.test_samples)}"
        # )
        # for batch in tqdm(dataloader):
        #     print(batch["prompts"], batch["frames"].shape)
        #     break

    def script4():
        ## new testing
        dataset = RoboDepth(data_root="/mnt/zhouxin-mnt", is_train=True)
        test_samples = dataset.test_samples
        out_path = "/mnt/zhouxin-mnt/self_forcing/test_images"

        for sample in test_samples:
            instruction = sample[0]
            rgb_path = sample[1]
            rgb_path_list = rgb_path.split("/")
            folder = rgb_path_list[-3]
            out_folder = os.path.join(out_path, folder)
            os.makedirs(out_folder, exist_ok=True)
            with open(os.path.join(out_folder, "ins.txt"), "w") as f:
                f.write(instruction)

            depth_path = (
                str(rgb_path)
                .replace("video", "depth/npz")
                .replace("rgb.mp4", "depth.npz")
            )
            depth_array = np.load(depth_path)["arr_0"].astype(np.float32)
            depth_array = (depth_array - depth_array.min()) / (
                depth_array.max() - depth_array.min() + 1e-8
            )
            depth_array *= 255.0
            depth_array = np.stack([depth_array] * 3, axis=-1)
            first_depth = depth_array[0]
            first_depth = cv2.cvtColor(first_depth, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_folder, "depth.png"), first_depth)
            shutil.copyfile(depth_path, os.path.join(out_folder, "target_depth.npz"))

            cap = cv2.VideoCapture(str(rgb_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            first_frame = frames[0]  # [H, W, C]
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_folder, "image.png"), first_frame)
            shutil.copyfile(rgb_path, os.path.join(out_folder, "target.mp4"))

    def script5():
        path = "/mnt/zhouxin-mnt/self_forcing/test_images/1/depth.png"
        depth_fig = Image.open(path).convert("RGB")
        depth_data = np.array(depth_fig)
        print(depth_data.shape)

    def script6():
        path = "/mnt/zhouxin-mnt/self_forcing/test_images/1/normal.mp4"
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        first_frame = frames[0]  # [H, W, C]
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite("/data/DriveGen/TesserAct/asset/images/1_normal.png", first_frame)

    def script7():
        path = "/mnt/zhouxin-mnt/self_forcing/test_images/1/depth.png"
        depth = Image.open(path).convert("RGB")
        depth_data = np.array(depth)[:, :, 0]
        print(depth_data.shape)
        np.save("/data/DriveGen/TesserAct/asset/images/1_depth.npy", depth_data)

    def script8():
        from examples.depth_wan.flow_utils import flow_to_pil_hsv
        from diffusers.utils.export_utils import export_to_video

        path = "/mnt/zhouxin-mnt/bridge/10/flow/flow.npz"
        shutil.copyfile("/mnt/zhouxin-mnt/bridge/10/video/rgb.mp4", "data/src.mp4")
        flow = np.load(path)["arr_0"]
        frames = []
        for frame in flow:
            print(frame[0][0][0], frame[1][0][0])
            img = flow_to_pil_hsv(frame)
            frames.append(img)
        frames = np.array(frames) / 255.0
        export_to_video(frames, "data/vis.mp4", fps=16)
        # print(flow.shape, flow.min(), flow.max())

    def script9():
        from ptlflow.utils import flow_utils

        path = "/mnt/zhouxin-mnt/bridge/10/flow/flow.npz"
        flow = np.load(path)["arr_0"]
        flow = torch.from_numpy(flow)
        frames = []
        for frame in flow:
            frame = frame.unsqueeze(0).unsqueeze(0)
            flow_rgb = flow_utils.flow_to_rgb(frame)
            flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
            flow_rgb_npy = flow_rgb.detach().cpu().numpy()
            frames.append(flow_rgb_npy)
        frames = np.array(frames)
        # print(frames.max(), frames.min())
        export_to_video(frames, "data/ptl_vis.mp4", fps=16)

    def script10():
        dset = MiniTrajRoboDepth(
            data_root="/mnt/zhouxin-mnt", is_train=True, max_num_frames=49
        )
        sample1 = dset[0]
        print(f"dataset length: {len(dset)}")
        print(sample1["prompts"])
        print(
            sample1["traj_map"].min(),
            sample1["traj_map"].max(),
            sample1["traj_map"].shape,
        )

    def script11():
        path = "/mnt/zhouxin-mnt/bridge/1/video/track_rgb.mp4"
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        print(frames.shape, frames.min(), frames.max())

    def script12():
        path = "/mnt/zhouxin-mnt/bridge/1/video/rgb.mp4"
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        for i in range(len(frames)):
            frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"/mnt/zhouxin-mnt/bridge/1/video/{i}.png", frame)

    def script13():
        path = "data/track_robot_red.mp4"
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        H, W, C = frames[0].shape
        for i in range(H):
            for j in range(W):
                if frames[0][i][j][0]:
                    print(f"row: {i}, col: {j}, pixel: {frames[0][i][j]}")

    def script14():
        import torch.nn.functional as F

        dset = MiniTrajRoboDepth(
            data_root="/mnt/zhouxin-mnt", is_train=True, max_num_frames=49
        )
        sample = dset[0]

        def _get_downsample_mask(traj_mask: torch.Tensor):
            first_mask = traj_mask[:1]
            r1 = F.max_pool2d(first_mask, kernel_size=(8, 8), stride=(8, 8))
            r2 = F.max_pool3d(traj_mask[1:], kernel_size=(4, 8, 8), stride=(4, 8, 8))
            result = torch.cat([r1, r2], dim=1)
            return result

        mask = sample["mask"]
        print(mask.shape)
        d_mask = _get_downsample_mask(mask)
        print(d_mask.shape)

    def script15():
        dset = MiniFlowRoboDepth(
            data_root="/mnt/zhouxin-mnt", is_train=True, max_num_frames=49
        )
        sample = dset[0]
        print(
            f"flow: {sample['flow'].shape}, {sample['flow'].min()}, {sample['flow'].max()}"
        )

    def script16():
        dataset = MiniTrajRoboDepth(
            data_root="/mnt/zhouxin-mnt", is_train=True, max_num_frames=49
        )
        sample1 = dataset[0]
        print(
            f"dataset for real train: {len(dataset)}\ndataset for all train: {len(dataset.train_samples)}"
        )
        print(
            f"sample1 first track: {sample1['first_track'].shape}, min/max: {sample1['first_track'].min()}/{sample1['first_track'].max()}"
        )

    def script17():
        sample_path = "/home/zhouxin/data/DriveGen/UniAnimate-DiT/samples.json"
        test_path = "/mnt/zhouxin-mnt/self_forcing/test_images"
        test_folders = set([folder for folder in os.listdir(test_path)])
        test_length = len(test_folders)
        with open(sample_path, "r") as f:
            res = json.load(f)
        real_train = res["train_samples"][:6000]
        real_train = set([s.split("/")[-3] for s in real_train])
        inter_length = len(test_folders.intersection(real_train))
        print(f"In/Out: {inter_length}/{test_length}, prob: {inter_length / test_length:.2f}")

    script17()
