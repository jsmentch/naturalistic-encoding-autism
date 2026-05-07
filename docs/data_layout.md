# Data layout

All paths are configurable via environment variables (see project `README.md`). Defaults assume a `data/` directory at the repository root when you run scripts from `scripts/` or notebooks from `notebooks/`. For publication links, tests, and pipeline mapping, see [reproducibility.md](reproducibility.md).

## Directory layout (default)

```text
data/
  features/              # `.npy` / `.h5` feature caches
  {stim}_clips_cochresnet50/
    cochresnet50_activations.h5
  {stim}_frames_resnet50/
    {layer}.npz
  DM_frames_all{1..5}_resnet50/   # optional splits for scripts/wen_38_PCA.py
  DM_frames_all_resnet50/         # NATURALISTIC_ENCODING_RESNET_DIR — processed layer bundles
  pilots_ru_dm.csv               # optional list for get_subject_list(); use `participant_id` (anonymous) or legacy `Identifiers_y` locally only — never commit restricted IDs
atlases/
  atlas-Glasser_space-fsLR_den-91k_dseg.dlabel.nii
  atlas-Glasser_dseg.tsv
stimuli/                         # NATURALISTIC_ENCODING_STIMULI — raw Friends A/V for extraction scripts
extra/
  yamnet_class_names.npy         # optional; NATURALISTIC_ENCODING_EXTRA_ROOT
```

## fMRI inputs

Set `NATURALISTIC_ENCODING_HBN_PTEMPLATE` to a Python format string with `{sub}`, pointing to a CIFTI / GIFTI timeseries file per subject (see your preprocessing pipeline, e.g. XCP-D Glasser parcellated outputs).

Example pattern (adapt to your site):

`.../sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii`

## Script outputs

`scripts/pilot.py` writes result `.npz` files under `../{output_directory_name}/` relative to the script working directory (typically the repository root when invoking from `scripts/`).
