import os
from copy import deepcopy
from pathlib import Path
import random
import json

random.seed(923)


def get_bridge_samples():
    path = "/mnt/zhouxin-mnt/bridge"
    folders = os.listdir(path)
    print(f"bridge: {len(folders)}")

    random.shuffle(folders)

    selected_folders = random.sample(folders, k=len(folders) // 2)
    unselected_folders = list(set(folders) - set(selected_folders))
    print(len(selected_folders))
    print(len(unselected_folders))

    selected_folders = [os.path.join(path, f) for f in selected_folders]
    unselected_folders = [os.path.join(path, f) for f in unselected_folders]

    d = {"train_samples": selected_folders, "test_samples": unselected_folders}

    with open("bridge_samples.json", "w") as f:
        json.dump(d, f, indent=4)


def get_fractal_samples():
    path = "/mnt/workspace/tusifan/Kairos_world_model/fractal20220817_data/processed"
    folders = os.listdir(path)
    print(f"fractal: {len(folders)}")

    random.shuffle(folders)

    selected_folders = random.sample(folders, k=len(folders) // 2)
    unselected_folders = list(set(folders) - set(selected_folders))
    print(len(selected_folders))
    print(len(unselected_folders))

    selected_folders = [os.path.join(path, f) for f in selected_folders]
    unselected_folders = [os.path.join(path, f) for f in unselected_folders]

    d = {"train_samples": selected_folders, "test_samples": unselected_folders}

    with open("fractal_samples.json", "w") as f:
        json.dump(d, f, indent=4)


def merge_samples(sample_jsons):
    final = {"train_samples": [], "test_samples": []}

    for sample_file in sample_jsons:
        with open(sample_file, "r") as f:
            cont = json.load(f)
        final["train_samples"].extend(cont["train_samples"])
        # final["test_samples"].extend(cont["test_samples"])

    with open("final_samples.json", "w") as f:
        json.dump(final, f, indent=4)

def test_cv2_decord():
    import time
    import cv2
    import decord
    videos_list = [
        "/mnt/workspace/tusifan/Kairos_world_model/fractal20220817_data/processed/17090/video/rgb.mp4",
        "/mnt/workspace/tusifan/Kairos_world_model/fractal20220817_data/processed/12848/video/rgb.mp4",
        "/mnt/workspace/tusifan/Kairos_world_model/fractal20220817_data/processed/43761/video/rgb.mp4",
        "/mnt/workspace/tusifan/Kairos_world_model/fractal20220817_data/processed/52495/video/rgb.mp4",
        "/mnt/workspace/tusifan/Kairos_world_model/fractal20220817_data/processed/61173/video/rgb.mp4"
    ]
    
    start_time = time.time()
    for video in videos_list:
        frames = []
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
    end_time = time.time()
    print(f"cv2: {end_time - start_time}s")

    start_time = time.time()
    for video in videos_list:
        vid = decord.VideoReader(video)
        v = vid.get_batch(list(range(len(vid)))).asnumpy()
    end_time = time.time()
    print(f"decord: {end_time - start_time}s")


if __name__ == "__main__":
    # get_bridge_samples()
    # get_fractal_samples()
    a = [
        "/mnt/workspace/tusifan/Kairos_world_model/DriveGen/data_process/Video-Depth-Anything/bridge_samples.json",
        "/mnt/workspace/tusifan/Kairos_world_model/DriveGen/data_process/Video-Depth-Anything/fractal_samples.json",
    ]
    merge_samples(a)