import os

import h5py
import nibabel as nb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection

from naturalistic_encoding.config import ATLAS_ROOT, HBN_PTEMPLATE, data_path


def segment_and_average(motion_features, num_segments=750):
    num_frames, num_features = motion_features.shape
    segment_length = num_frames // num_segments
    trimmed_length = segment_length * num_segments
    motion_features_trimmed = motion_features[:trimmed_length, :]
    reshaped_features = motion_features_trimmed.reshape(
        num_segments, segment_length, num_features
    )
    averaged_features = np.mean(reshaped_features, axis=1)
    return averaged_features


def fdr_correction(p_values):
    sorted_p_indices = np.argsort(p_values)
    sorted_p_values = np.sort(p_values)

    m = len(p_values)
    ranks = np.arange(1, m + 1)
    corrected_p_values = sorted_p_values * m / ranks

    corrected_p_values = np.minimum.accumulate(corrected_p_values[::-1])[::-1]

    unsorted_corrected_p_values = np.empty_like(corrected_p_values)
    unsorted_corrected_p_values[sorted_p_indices] = corrected_p_values

    return unsorted_corrected_p_values


def standardscale(X_raw):
    from sklearn.preprocessing import StandardScaler

    X = []
    for xx in X_raw:
        scaler = StandardScaler()
        X.append(scaler.fit_transform(X=xx, y=None))
    return X


def apply_srp(X, eps):
    from sklearn.random_projection import johnson_lindenstrauss_min_dim

    X_srp = []
    for xx in X:
        n_samples = xx.shape[0]
        n_components = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)

        if n_components < xx.shape[1]:
            srp = SparseRandomProjection(n_components=n_components, random_state=42)
            X_srp.append(srp.fit_transform(xx))
        else:
            X_srp.append(xx)
    return X_srp


def apply_pca(X, n_components):
    from sklearn.decomposition import PCA

    X_pca = []
    for xx in X:
        pca = PCA(n_components=n_components)
        X_pca.append(pca.fit_transform(xx))
    return X_pca


def apply_zscore(X):
    from scipy.stats import zscore

    X_z = []
    for x in X:
        X_z.append(zscore(x))
    return X_z


def compute_pca_components(arrays, variance_threshold=0.95):
    from sklearn.decomposition import PCA

    component_counts = []
    for array in arrays:
        pca = PCA()
        pca.fit(array)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(explained_variance >= variance_threshold) + 1
        component_counts.append(num_components)
    return component_counts


def load_audio_features(stim, all_layers):
    save_features_dir = data_path(f"{stim}_clips_cochresnet50")
    X = []
    file = h5py.File(
        save_features_dir / "cochresnet50_activations.h5",
        "r",
    )
    for layer in all_layers:
        data = file[layer]
        X.append(np.array(data))
    file.close()
    return X


def load_features_processed(filename, all_layers):
    save_features_dir = data_path("features")
    X = []
    file = h5py.File(save_features_dir / filename, "r")
    for layer in all_layers:
        X.append(np.array(file[layer]))
    file.close()
    return X


def load_audio_features_manual_hrf(stim, features):
    from scipy.signal import resample

    from naturalistic_encoding import hrf_tools

    X = []
    for f in features:
        feature = np.load(data_path("features", f"{stim}_{f}.npy"))
        scaler = StandardScaler()
        feature = scaler.fit_transform(X=feature, y=None)
        hz = feature.shape[0] / 600
        feature = hrf_tools.apply_optimal_hrf_10hz(feature, hz)
        feature = resample(feature, 750, axis=0)
        X.append(feature)
    return X


def load_both_features_hrf(stim):
    features_manual = ["as_embed", "as_scores"]
    features_cochresnet = [
        "input_after_preproc",
        "conv1_relu1",
        "maxpool1",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "avgpool",
    ]
    from naturalistic_encoding import hrf_tools

    features = features_cochresnet
    X_raw = load_audio_features(stim, features)
    X = standardscale(X_raw)
    X = apply_pca(X, 1)
    for xx in X:
        hz = xx.shape[0] / 600
        hrf_tools.apply_optimal_hrf_10hz(xx, hz)
    x1shape = X[0].shape[0]
    X2 = load_audio_features_manual_hrf(stim, features_manual)
    for xx in X2:
        X.append(xx)
    x2shape = X[-1].shape[0]
    X = [array[: min([x2shape, x1shape]), :] for array in X]
    return X


def load_audio_features_SRP_delay(stim, delay, all_layers, n_components):
    transformer = SparseRandomProjection(n_components=n_components, random_state=42)
    print(f"loading features {n_components} SRP components")
    save_features_dir = data_path(f"{stim}_clips_cochresnet50")
    X = []
    file = h5py.File(save_features_dir / "cochresnet50_activations.h5", "r")
    for layer in all_layers:
        data = file[layer]
        X.append(transformer.fit_transform(np.array(data)[: (-1 * delay), :]))
    file.close()
    return X


def load_audio_features_PCA(stim, all_layers, n_components):
    transformer = PCA(n_components=n_components)
    print(f"loading features {n_components} PCA components")
    save_features_dir = data_path(f"{stim}_clips_cochresnet50")
    X = []
    file = h5py.File(save_features_dir / "cochresnet50_activations.h5", "r")
    for layer in all_layers:
        data = file[layer]
        data = np.nan_to_num(data, nan=0.0)
        scaler = StandardScaler()
        X.append(
            scaler.fit_transform(X=transformer.fit_transform(np.array(data)), y=None)
        )
    file.close()
    return X


def load_audio_features_SRP(stim, all_layers, eps):
    print("loading features for srp")
    save_features_dir = data_path(f"{stim}_clips_cochresnet50")
    X = []
    file = h5py.File(save_features_dir / "cochresnet50_activations.h5", "r")
    for layer in all_layers:
        data = file[layer]
        data = np.nan_to_num(data, nan=0.0)
        scaler = StandardScaler()
        X.append(scaler.fit_transform(np.array(data), y=None))
    file.close()
    X = apply_srp(X, eps)
    return X


def load_audio_features_PCAc2(stim, all_layers):
    transformer = PCA(n_components=2)
    save_features_dir = data_path(f"{stim}_clips_cochresnet50")

    X = []
    file = h5py.File(save_features_dir / "cochresnet50_activations.h5", "r")
    for layer in all_layers:
        data = file[layer]
        X.append(transformer.fit_transform(np.array(data))[:, 1:])
    file.close()
    return X


def load_video_features_srp(stim, all_layers):
    transformer = SparseRandomProjection(n_components=50, random_state=42)
    save_path = data_path(f"{stim}_frames_resnet50")
    X = []
    for layer in all_layers:
        X_layer = []
        emb = np.load(save_path / f"{layer}.npz")
        for k in list(emb.keys()):
            X_layer.append(emb[k].flatten())
        X.append(transformer.fit_transform(np.array(X_layer)))
    return X


def load_fmri_data(im_file, delay, indices):
    img = nb.load(im_file)
    img_y = img.get_fdata()
    Y = img_y[delay:, indices]
    return Y


def get_subject_list():
    if not HBN_PTEMPLATE:
        raise ValueError(
            "Set NATURALISTIC_ENCODING_HBN_PTEMPLATE to a path template containing "
            "{sub}, pointing to Glasser parcellated mean timeseries .ptseries.nii "
            "(see docs/data_layout.md)."
        )
    pilot_subjects = pd.read_csv(data_path("pilots_ru_dm.csv"))
    id_col = (
        "participant_id"
        if "participant_id" in pilot_subjects.columns
        else "Identifiers_y"
    )
    subjects = []
    for sub in pilot_subjects[id_col].astype(str):
        im_file = HBN_PTEMPLATE.format(sub=sub)
        if not os.path.isfile(im_file):
            print(f"missing {sub}")
            continue
        try:
            nb.load(im_file)
            subjects.append(sub)
        except Exception:
            print(f"missing {sub}")
    print(f"loaded {len(subjects)} subjects")
    return subjects


def get_parcel_indices(atlas, parcels):
    patternR = "|".join(["Right_" + parcel for parcel in parcels])
    patternL = "|".join(["Left_" + parcel for parcel in parcels])

    matches = atlas["label"].str.contains(patternR) | atlas["label"].str.contains(
        patternL
    )

    atlas_indices = atlas[matches]["index"].tolist()
    indices = matches[matches].index.tolist()
    parcel_names = atlas[matches]["label"].tolist()
    return atlas_indices, indices, parcel_names


def load_audio_features_RAW(stim, all_layers):
    save_features_dir = data_path(f"{stim}_clips_cochresnet50")

    X = []
    file = h5py.File(save_features_dir / "cochresnet50_activations.h5", "r")
    for layer in all_layers:
        data = file[layer]
        print(data.shape, layer)
        X.append(np.array(data))

    file.close()
    return X


def load_glasser():
    atlas_dlabel = ATLAS_ROOT / "atlas-Glasser_space-fsLR_den-91k_dseg.dlabel.nii"
    atlas_tsv = ATLAS_ROOT / "atlas-Glasser_dseg.tsv"
    if not atlas_dlabel.is_file():
        raise FileNotFoundError(
            f"Expected Glasser dlabel at {atlas_dlabel}. "
            "Set NATURALISTIC_ENCODING_ATLAS or place atlas files under atlases/."
        )
    img = nb.load(str(atlas_dlabel))
    atlas_data = img.get_fdata()
    atlas_data = atlas_data[0, :]
    atlas = pd.read_csv(atlas_tsv, sep="\t")
    return atlas, atlas_data