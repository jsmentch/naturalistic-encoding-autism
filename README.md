# Naturalistic encoding and autism spectrum code

This repository accompanies research on **encoding models** fit to fMRI data collected during **naturalistic movie and audio** stimuli, including work on autism spectrum and typical development. The codebase centers on **cross-validated ridge regression**, **quadratic-programming feature stacking** (following Wolpert-style stacking for continuous outputs), **HRF convolution** of high-rate sensory features, and utilities for loading multimodal predictors (audio CNNs, video embeddings, low-level vision, arousal, etc.).

## Publication and preregistration

| Resource | Link |
|----------|------|
| **bioRxiv preprint** | [*Pregistered movie-fMRI analyses reveal altered visual feature encoding in autism in pSTS*](https://www.biorxiv.org/content/10.64898/2026.03.23.713749v1.abstract) · doi [10.64898/2026.03.23.713749](https://doi.org/10.64898/2026.03.23.713749) |
| **Peer review** | Accepted for review at **eLife** (will be updated with the version of record once published). |
| **Preregistered analysis plans (OSF)** | [osf.io/h92gr](https://osf.io/h92gr) · [osf.io/47kj6](https://osf.io/47kj6) |

For reproducibility notes, tests, and a **pipeline ↔ manuscript** map, see [docs/reproducibility.md](docs/reproducibility.md).

**Note:** Raw neuroimaging data and stimuli are not redistributed here. Configure paths via environment variables and see [docs/data_layout.md](docs/data_layout.md).

## Continuous integration

[![CI](https://github.com/jsmentch/naturalistic-encoding-autism/actions/workflows/ci.yml/badge.svg)](https://github.com/jsmentch/naturalistic-encoding-autism/actions)

Tests run on **Ubuntu** for Python **3.10** and **3.12** via [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (`pytest`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Audio / video feature extraction scripts additionally need:

```bash
pip install -e ".[av]"
```

Run tests:

```bash
pip install -e ".[dev]"
pytest
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `NATURALISTIC_ENCODING_DATA` | Root for `data/` (default: `<repo>/data`) |
| `NATURALISTIC_ENCODING_STIMULI` | Raw Friends (or other) media for extraction scripts |
| `NATURALISTIC_ENCODING_ATLAS` | Glasser atlas files (`.dlabel.nii`, `.tsv`) |
| `NATURALISTIC_ENCODING_RESNET_DIR` | Processed ResNet layer bundles (`*_hrf_resamp_scale.npz`, etc.) |
| `NATURALISTIC_ENCODING_RESNET_VIDEO_DIR` | Input `.npz` per ResNet layer for `scripts/wen_38_PCA.py` |
| `NATURALISTIC_ENCODING_EXTRA_ROOT` | Aux files e.g. `yamnet_class_names.npy` |
| `NATURALISTIC_ENCODING_HBN_PTEMPLATE` | fMRI template with `{sub}` for `get_subject_list()` |
| `NATURALISTIC_ENCODING_YOLO_DIR` | Base folder for YOLO label exports (video notebook) |
| `NATURALISTIC_ENCODING_COCHMODEL_DIR` | CochResNet / cochdnn model directory |
| `NATURALISTIC_ENCODING_FMRIPREP_DIR` | Optional public fMRIprep tree root (CNeuroMoD-style paths) |

## Usage

Encoding driver (after installing the package editable and arranging inputs under `data/`):

```bash
python scripts/pilot.py --help
# Example (use your anonymous participant_id, not restricted GUIDs):
python scripts/pilot.py -s sub-01 -p auditory -f cochresnet50pca1 -d 7 -l
```

Helpers live in the importable package `naturalistic_encoding`:

```python
from naturalistic_encoding.stacking_fmri import stacking_CV_fmri
from naturalistic_encoding import hrf_tools, nat_asd_utils
```

## Notebooks

Curated pipeline notebooks (numbered) are under [notebooks/](notebooks/): QC → stimulus preprocessing → video/audio features → pilot encoding. They are **cleared of outputs** and use **placeholders / environment variables** instead of institution paths or participant GUIDs. Use a local CSV with an anonymous `participant_id` column (see [docs/data_layout.md](docs/data_layout.md)). Install the package in the same environment as Jupyter so `naturalistic_encoding` imports resolve. To re-run the cleaner: `python scripts/clean_notebooks.py`.

## Citation

Please cite the **bioRxiv** preprint (and the **eLife** article once available). GitHub can surface metadata from [`CITATION.cff`](CITATION.cff).

**bioRxiv**

> Mentch, J., Chen, Y., Vanderwal, T., & Ghosh, S. S. (2026). Pregistered movie-fMRI analyses reveal altered visual feature encoding in autism in pSTS. *bioRxiv*. [https://doi.org/10.64898/2026.03.23.713749](https://doi.org/10.64898/2026.03.23.713749)

**eLife** — will add volume, page, and doi here after the version of record is published.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT License](LICENSE).
