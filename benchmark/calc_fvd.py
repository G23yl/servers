import numpy as np
import os
import json
import torch
# from fvd.styleganv.fvd import frechet_distance

method = 'styleganv'

if method == 'styleganv':
    from fvd.styleganv.fvd import frechet_distance
elif method == 'videogpt':
    from fvd.videogpt.fvd import frechet_distance

target_path = "preprocess_data/target"
pred_path = "preprocess_data/pred"

target_feats = []
pred_feats = []

for file_path in os.listdir(target_path):
    file_path = os.path.join(target_path, file_path)

    feat = np.load(file_path)
    target_feats.append(feat)

for file_path in os.listdir(pred_path):
    file_path = os.path.join(pred_path, file_path)

    feat = np.load(file_path)
    pred_feats.append(feat)

target_feats = np.array(target_feats)
pred_feats = np.array(pred_feats)

# target_feats = torch.from_numpy(target_feats).float()
# pred_feats = torch.from_numpy(pred_feats).float()

# print(target_feats.shape)
# print(pred_feats.shape)

fvd = frechet_distance(pred_feats, target_feats)
res = {
    "fvd": {
        "value": [fvd]
    }
}

print(res)

with open("./fvd.json", "w") as f:
    json.dump(res, f, indent=4)