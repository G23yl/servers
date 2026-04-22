import torch
import argparse
import os
import cv2
import numpy as np
import json
from tqdm import tqdm


def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert gt.shape == pred.shape == valid_mask.shape, (
        f"{gt.shape}, {pred.shape}, {valid_mask.shape}"
    )

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


def abs_relative_difference(output, target, valid_mask=None):
    # [F, H, W]
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    # print(abs_relative_diff)
    return abs_relative_diff.mean().item()


def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2 < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n
    return threshold_mat.mean().item()


def delta1_acc(pred, gt, valid_mask=None):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask=None):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--pred", type=str, help="Predicted video base name")
    parser.add_argument("--target", type=str, help="Target video base name")
    args = parser.parse_args()

    path = args.test_dir
    pred_video_name = f"{args.pred}.mp4"
    target_video_name = f"{args.target}.npz"
    absrel_tot = []
    delta1_tot = []
    delta2_tot = []
    min_depth = 0.05
    max_depth = 0.95

    folders = os.listdir(path)

    for folder in tqdm(folders):
        pred_depth_path = os.path.join(path, folder, pred_video_name)
        target_depth_path = os.path.join(path, folder, target_video_name)
        if not os.path.exists(pred_depth_path) or not os.path.exists(target_depth_path):
            continue
        cap = cv2.VideoCapture(pred_depth_path)
        pred_frames = []  # [F, H, W, C]
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_frames.append(frame)
        cap.release()
        pred_frames = np.stack(pred_frames, axis=0)  # 0..255
        pred_frames = torch.from_numpy(pred_frames).permute(0, 3, 1, 2)[
            :, 0
        ]  # [F, H, W]
        pred_frames = min_depth + pred_frames * (max_depth - min_depth) / 255.0
        pred_frames = pred_frames.clamp(min_depth, max_depth)

        # [-1, 1]
        target_depths = np.load(target_depth_path)["arr_0"].astype(np.float32)
        # target_depths = (target_depths + 1.0) / 2.0  # 0..1
        # target_depths *= 255.0
        target_depths = min_depth + target_depths * (max_depth - min_depth) / 2.0
        target_depths = torch.from_numpy(target_depths)  # [F, H, W]
        target_depths = target_depths.clamp(min_depth, max_depth)

        valid_mask = torch.logical_and(
            target_depths >= min_depth, target_depths <= max_depth
        ).bool()
        # valid_mask = None

        pred_frames, scale, shift = align_depth_least_square(
            gt_arr=target_depths.numpy(),
            pred_arr=pred_frames.numpy(),
            valid_mask_arr=valid_mask.numpy(),
        )

        pred_frames = torch.from_numpy(pred_frames).clamp(min_depth, max_depth)
        target_depths = target_depths.clamp(min_depth, max_depth)

        abs_rel = abs_relative_difference(pred_frames, target_depths, valid_mask)
        d1 = delta1_acc(pred_frames, target_depths, valid_mask)
        d2 = delta2_acc(pred_frames, target_depths, valid_mask)
        absrel_tot.append(abs_rel)
        delta1_tot.append(d1)
        delta2_tot.append(d2)
        print(f"{folder}: {abs_rel}")

    res = {
        "abs_rel": np.array(absrel_tot).mean(),
        "delta1_acc": np.array(delta1_tot).mean(),
        "delta2_acc": np.array(delta2_tot).mean(),
    }

    print(res)

    with open("./abs_rel.json", "w") as f:
        json.dump(res, f, indent=4)
