# Low-level video features + moten motion energy (requires pip install .[av]).

import sys

import cv2
import h5py
import moten
import numpy as np
from scipy.signal import resample

from naturalistic_encoding.config import STIMULI_ROOT, data_path

if len(sys.argv) > 1:
    stim = sys.argv[1]
    print(f"Received string: {stim}")
else:
    print("Usage: python extract_llv.py <stim_id e.g. s01e02a>")
    sys.exit(2)

TR = 1.49
print(stim)


def resolve_video(preferred_name: str) -> str:
    candidates = [
        STIMULI_ROOT / "friends.stimuli" / "s1" / preferred_name,
        STIMULI_ROOT / "s1" / preferred_name,
        STIMULI_ROOT / preferred_name,
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return str(candidates[0])


video_path = resolve_video(f"friends_{stim}.mkv")
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

num_TRs = int(np.round(total_frames / fps / TR))
print(f"num_TRs={str(num_TRs)}")

brightness_list = []
contrast_list = []
color_features_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_frame)
    brightness_list.append(brightness)

    contrast = np.std(gray_frame)
    contrast_list.append(contrast)

    mean_color = cv2.mean(frame)[:3]
    color_features_list.append(mean_color)

cap.release()

brightness_list = resample(np.asanyarray(brightness_list), num_TRs, axis=0)
contrast_list = resample(np.asanyarray(contrast_list), num_TRs, axis=0)
color_features_list = resample(np.asanyarray(color_features_list), num_TRs, axis=0)

feat_dir = data_path("features")
feat_dir.mkdir(parents=True, exist_ok=True)
np.save(feat_dir / f"friends_{stim}_color.npy", color_features_list)
np.save(feat_dir / f"friends_{stim}_brightness.npy", brightness_list)
np.save(feat_dir / f"friends_{stim}_contrast.npy", contrast_list)

print("Color features array shape:", color_features_list.shape)

gamma = 2.2


def gamma_correction(channel):
    return ((channel / 255.0) ** gamma) * 255


def calculate_brightness(frame):
    B, G, R = cv2.split(frame)
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return brightness.mean()


def calculate_perceptual_brightness_with_gamma(frame):
    B, G, R = cv2.split(frame)
    R = gamma_correction(R)
    G = gamma_correction(G)
    B = gamma_correction(B)
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return brightness.mean()


video = cv2.VideoCapture(video_path)
brightness_values = []
brightness_values4 = []

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    brightness_values.append(calculate_brightness(frame))
    brightness_values4.append(calculate_perceptual_brightness_with_gamma(frame))

video.release()

brightness_values = resample(np.asanyarray(brightness_values), num_TRs, axis=0)
brightness_values4 = resample(np.asanyarray(brightness_values4), num_TRs, axis=0)

np.save(feat_dir / f"friends_{stim}_brightness3.npy", np.array(brightness_values))
np.save(feat_dir / f"friends_{stim}_brightness4.npy", np.array(brightness_values4))

video_alt = resolve_video(f"{stim}.mkv")
video = cv2.VideoCapture(video_alt)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total number of frames: {total_frames}")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Width: {width}, Height: {height}")
fps = video.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")
video.release()

video_file = video_alt
small_vhsize = (72, 48)
luminance_images = moten.io.video2luminance(video_file, size=small_vhsize)
nimages, vdim, hdim = luminance_images.shape
pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=fps)
moten_features = pyramid.project_stimulus(luminance_images)

hdf5_path = feat_dir / f"{stim}_pymoten.h5"
with h5py.File(hdf5_path, "w") as hdf5_file:
    hdf5_file.create_dataset("pymoten", data=moten_features)

print(f"Motion features saved to {hdf5_path}")
