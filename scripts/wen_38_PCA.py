"""
Batch-process ResNet50 frame embeddings: HRF, resample to TRs, scale, save per layer.

Set NATURALISTIC_ENCODING_RESNET_VIDEO_DIR (inputs) and NATURALISTIC_ENCODING_RESNET_DIR
(outputs); see README.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

from naturalistic_encoding import hrf_tools
from naturalistic_encoding.config import RESNET_FEATURES_DIR, RESNET_VIDEO_DIR


layers = ["relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"]


def load_resnet50(layer):
    X = []
    emb = np.load(RESNET_VIDEO_DIR / f"{layer}.npz")
    for k in list(emb.keys()):
        X.append(emb[k].flatten())
    X = np.asanyarray(X)
    return X


def load_resnet50_split(layer):
    splits = []
    for split in ["1", "2", "3", "4", "5"]:
        split_dir = RESNET_VIDEO_DIR.parent / f"DM_frames_all{split}_resnet50"
        X = []
        emb = np.load(split_dir / f"{layer}.npz")
        for k in list(emb.keys()):
            X.append(emb[k].flatten())
        X = np.asanyarray(X)
        splits.append(X)
    splits = np.concatenate(splits)
    return splits


for layer in layers:
    X = load_resnet50_split(layer)
    print(layer, X.shape)
    hz = X.shape[0] / 600
    X = hrf_tools.apply_optimal_hrf_10hz_NEW(X, hz)
    X = resample(X, 750, axis=0)
    print("resample", X.shape)

    scaler = StandardScaler()
    X = scaler.fit_transform(X=X, y=None)
    print("scale", X.shape)

    out = RESNET_FEATURES_DIR / f"{layer}_hrf_resamp_scale.npz"
    RESNET_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X)
