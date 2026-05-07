# Reproducibility

## Publication status

- **bioRxiv:** [Pregistered movie-fMRI analyses reveal altered visual feature encoding in autism in pSTS](https://www.biorxiv.org/content/10.64898/2026.03.23.713749v1.abstract) (doi [10.64898/2026.03.23.713749](https://doi.org/10.64898/2026.03.23.713749)).
- **Peer review:** Manuscript accepted for review at **eLife** (update the root `README.md` when the version of record is available).
- **Preregistration:** Analysis plans on the Open Science Framework — [osf.io/h92gr](https://osf.io/h92gr) and [osf.io/47kj6](https://osf.io/47kj6).

## Environment

- **Python:** 3.9+ (CI tests **3.10** and **3.12** on Ubuntu).
- **Install:** `pip install -e .` for the core library; `pip install -e ".[av]"` if you run audio/video extraction helpers; `pip install -e ".[dev]"` before running tests.

Restricted inputs (HBN identifiers, raw imaging, stimuli under data-use agreements) are **not** bundled; configure paths via environment variables ([data_layout.md](data_layout.md)).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Continuous integration runs the same suite via [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

## Pipeline overview (code ↔ paper)

| Stage | Location | Role |
|--------|----------|------|
| QC / cohort (illustrative only in repo) | [notebooks/01_QC.ipynb](../notebooks/01_QC.ipynb) | Synthetic example; real QC stays local |
| Stimulus framing | [notebooks/02-preprocess_stimuli.ipynb](../notebooks/02-preprocess_stimuli.ipynb) | Extract TR-aligned frames / clips |
| Visual features | [notebooks/03-video_feature_extraction.ipynb](../notebooks/03-video_feature_extraction.ipynb) | DNN / low-level video features |
| Auditory features | [notebooks/04-audio_feature_extraction.ipynb](../notebooks/04-audio_feature_extraction.ipynb) | CochResNet and related audio embeddings |
| Encoding pilots | [notebooks/05-pilot_encoding.ipynb](../notebooks/05-pilot_encoding.ipynb) | Stacked encoding prototypes |
| Full jobs | [scripts/pilot.py](../scripts/pilot.py), [scripts/pilot_bootstrap.py](../scripts/pilot_bootstrap.py) | Parcel-wise encoding with CLI flags |

Core algorithms (`ridge_tools`, `stacking_fmri`, `hrf_tools`) live under `src/naturalistic_encoding/` and are imported by notebooks and scripts.
