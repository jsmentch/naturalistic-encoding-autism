"""Filesystem layout via environment variables (see README)."""

from __future__ import annotations

import os
from pathlib import Path

# src/naturalistic_encoding/config.py -> project root is parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env_key: str, default: Path) -> Path:
    v = os.environ.get(env_key)
    return Path(v).expanduser() if v else default


DATA_ROOT = _path("NATURALISTIC_ENCODING_DATA", _PROJECT_ROOT / "data")
STIMULI_ROOT = _path("NATURALISTIC_ENCODING_STIMULI", _PROJECT_ROOT / "stimuli")
ATLAS_ROOT = _path("NATURALISTIC_ENCODING_ATLAS", _PROJECT_ROOT / "atlases")
RESNET_FEATURES_DIR = _path(
    "NATURALISTIC_ENCODING_RESNET_DIR",
    DATA_ROOT / "DM_frames_all_resnet50",
)
RESNET_VIDEO_DIR = _path(
    "NATURALISTIC_ENCODING_RESNET_VIDEO_DIR",
    DATA_ROOT / "DM_videos_resnet50",
)
EXTRA_DATA_ROOT = _path(
    "NATURALISTIC_ENCODING_EXTRA_ROOT",
    DATA_ROOT / "extra",
)


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


# HBN / XCP time series: use format with {sub}, e.g.
# .../sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii
HBN_PTEMPLATE = os.environ.get(
    "NATURALISTIC_ENCODING_HBN_PTEMPLATE",
    "",
)
