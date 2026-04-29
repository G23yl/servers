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

def get_test_samples():
    fractal = "/mnt/workspace/tusifan/Kairos_world_model/DriveGen/data_process/Video-Depth-Anything/fractal_samples.json"
    bridge = "/mnt/workspace/tusifan/Kairos_world_model/DriveGen/data_process/Video-Depth-Anything/bridge_samples.json"
    selected = {"test_samples": []}
    with open(fractal, "r") as f:
        test = json.load(f)["test_samples"]
        f_test = random.sample(test, k=200)
        selected["test_samples"].extend(f_test)
    with open(bridge, "r") as f:
        test = json.load(f)["test_samples"]
        b_test = random.sample(test, k=200)
        selected["test_samples"].extend(b_test)
    print(len(selected['test_samples']))
    with open("test_samples.json", "w") as f:
        json.dump(selected, f, indent=4)

def create_test_folders():
    from decord import VideoReader
    import imageio
    import shutil
    from tqdm import tqdm
    DATASET2ROBOT = {
        "fractal20220817_data": "google robot",
        "bridge": "Trossen WidowX 250 robot arm",
    }
    with open("test_samples.json", "r") as f:
        test_samples = json.load(f)["test_samples"]
    test_dir = "/mnt/workspace/tusifan/Kairos_world_model/ks_test_folder"
    for i in tqdm(range(len(test_samples))):
        test_folder = os.path.join(test_dir, f"{i:03}")
        os.makedirs(test_folder, exist_ok=True)
        sample = test_samples[i]
        ins_path = os.path.join(sample, "instruction.txt")
        test_ins_path = os.path.join(test_folder, "ins.txt")
        video_path = os.path.join(sample, "video", "rgb.mp4")
        with open(ins_path, "r") as f:
            ins = f.read().strip(". ").lower()
        if "bridge" in sample:
            ins += f" {DATASET2ROBOT['bridge']}"
        elif "fractal20220817_data" in sample:
            ins += f" {DATASET2ROBOT['fractal20220817_data']}"
        with open(test_ins_path, "w") as f:
            f.write(f"{ins}")
        vid = VideoReader(video_path)
        video = vid.get_batch(list(range(len(vid)))).asnumpy()
        first_image = video[0]
        imageio.imwrite(os.path.join(test_folder, "image.png"), first_image)
        shutil.copyfile(video_path, os.path.join(test_folder, "target_rgb.mp4"))


if __name__ == "__main__":
    # get_bridge_samples()
    # get_fractal_samples()
    # a = [
    #     "/mnt/workspace/tusifan/Kairos_world_model/DriveGen/data_process/Video-Depth-Anything/bridge_samples.json",
    #     "/mnt/workspace/tusifan/Kairos_world_model/DriveGen/data_process/Video-Depth-Anything/fractal_samples.json",
    # ]
    # merge_samples(a)
    # get_test_samples()
    # create_test_folders()
    from decord import VideoReader
    import imageio
    path = "/mnt/workspace/tusifan/Kairos_world_model/fractal20220817_data/processed/8348"
    with open(os.path.join(path, "instruction.txt"), "r") as f:
        ins = f.read()
    print(ins)
    vid = VideoReader(os.path.join(path, "video", "rgb.mp4"))
    video = vid.get_batch(list(range(len(vid)))).asnumpy()
    first_image = video[0]
    imageio.imwrite("/mnt/workspace/tusifan/Kairos_world_model/DriveGen/UniAnimate-DiT-utils/data/input/first_image_fractal_8348.png", first_image)