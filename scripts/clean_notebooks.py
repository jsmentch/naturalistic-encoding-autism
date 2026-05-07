#!/usr/bin/env python3
"""De-identify notebooks: clear outputs, drop empty cells, remove commented-out code, redact paths."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
QC_PATH = NB_DIR / "01_QC.ipynb"

COMMENTED_CODE = re.compile(
    r"^\s*#\s*(import |from |if |for |with |return |print\(|plt\.|np\.|pd\.|def |class |elif |else:|try:|except|raise |pass|[\w.]+\s*=|open\(|h5py\.|cv2\.|\w+\()"
)


def is_commented_out_code(line: str) -> bool:
    s = line.lstrip()
    if not s.startswith("#"):
        return False
    rest = s[1:].strip()
    if not rest or rest.startswith("##"):
        return False
    if rest.startswith("Read ") or rest.startswith("Release ") or rest.startswith("Loop "):
        return False
    if "dimensionality reduction" in rest or "Don't forget" in rest:
        return True
    return bool(COMMENTED_CODE.match(rest)) or (
        "=" in rest[:100] and not rest.startswith("Example")
    )


def strip_commented_code(src: str) -> str:
    lines = src.splitlines(keepends=True)
    return "".join(L for L in lines if not is_commented_out_code(L))


def norm_source(text: str) -> list[str]:
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]


def clear_code_cell(c: dict) -> None:
    c["outputs"] = []
    c["execution_count"] = None


def redact_sources(path: Path, src: str) -> str:
    name = path.name
    src = src.replace("#from dotenv import load_dotenv\n", "")
    src = src.replace("#from utils import setup_logging\n", "")
    src = src.replace("#import logging\n", "")
    src = src.replace("#logging.", "# logging.")

    if name == "02-preprocess_stimuli.ipynb":
        src = src.replace(
            "friends_path='/nese/mit/group/sig/projects/cneuromod/cneuromod/friends/stimuli/s1'",
            "friends_path = str(Path(os.environ.get('NATURALISTIC_ENCODING_STIMULI', '../stimuli')) / 'friends' / 's1')",
        )
        src = src.replace(
            "f'/om2/scratch/tmp/jsmentch/nat_img/sourcedata/data/cneuromod/stimuli/friends/s1/s1{stim}.wav'",
            "str(Path(os.environ.get('NATURALISTIC_ENCODING_STIMULI', '../stimuli')) / 'friends' / 's1' / f's1{stim}.wav')",
        )
        if "from pathlib import Path" not in src and ("Path(os.environ" in src or "Path(" in src):
            src = "from pathlib import Path\nimport os\n" + src
        elif "Path(os.environ" in src and "import os" not in src:
            src = "import os\n" + src

    if name == "04-audio_feature_extraction.ipynb":
        src = src.replace(
            "model_dir = '/om2/user/jsmentch/cochdnn/model_directories/resnet50_word_speaker_audioset'",
            "model_dir = os.environ.get('NATURALISTIC_ENCODING_COCHMODEL_DIR', '../models/cochresnet50')",
        )
        src = src.replace(
            "model_dir = os.environ.get('NATURALISTIC_ENCODING_COCHMODEL_DIR', '../models/cochresnet50')",
            "model_dir = os.environ.get('NATURALISTIC_ENCODING_COCHMODEL_DIR', '../models/cochresnet50')",
        )
        if "model_dir = os.environ" in src and "import os" not in src.split("\n")[:5]:
            src = "import os\n" + src

    if name == "03-video_feature_extraction.ipynb":
        subs = [
            (
                "txt_dir = '/om2/scratch/tmp/jsmentch/yolov11_output/yolo_DM/labels'",
                "txt_dir = str(Path(os.environ.get('NATURALISTIC_ENCODING_YOLO_DIR', '../data/yolo_labels')) / 'yolo_DM' / 'labels')",
            ),
            (
                "fmriprep_folder='/nese/mit/group/sig/projects/cneuromod/postproc_fmriprep/friends'",
                "fmriprep_folder = str(Path(os.environ.get('NATURALISTIC_ENCODING_FMRIPREP_DIR', '../data/cneuromod_fmriprep')) / 'friends')",
            ),
            (
                "pliers_all=pd.read_csv('/om2/scratch/tmp/jsmentch/nat_img/sourcedata/data/HBN/features/DM_pliers_all.csv')",
                "pliers_all = pd.read_csv(Path(os.environ.get('NATURALISTIC_ENCODING_DATA', '../data')) / 'features' / 'DM_pliers_all.csv')",
            ),
        ]
        for a, b in subs:
            src = src.replace(a, b)
        src = re.sub(
            r"txt_dir = f'/om2/user/jsmentch/yolov11_output/yolo_\{stim\}/labels'",
            "txt_dir = str(Path(os.environ.get('NATURALISTIC_ENCODING_YOLO_DIR', '../data/yolo_labels')) / f'yolo_{stim}' / 'labels')",
            src,
        )
        src = re.sub(
            r"txt_dir = f'/om2/user/jsmentch/yolov11_output/yolo_results_\{stim\}/labels'",
            "txt_dir = str(Path(os.environ.get('NATURALISTIC_ENCODING_YOLO_DIR', '../data/yolo_labels')) / f'yolo_results_{stim}' / 'labels')",
            src,
        )
        src = re.sub(
            r"video_path = f'/om2/scratch/tmp/jsmentch/nat_img/sourcedata/data/cneuromod/friends\.stimuli/s1/\{stim\}\.mkv'",
            "video_path = str(Path(os.environ.get('NATURALISTIC_ENCODING_STIMULI', '../stimuli')) / 'friends.stimuli' / 's1' / f'{stim}.mkv')",
            src,
        )
        src = re.sub(
            r"video_path = f'/om2/scratch/tmp/jsmentch/nat_img/sourcedata/data/cneuromod/friends\.stimuli/s1/friends_\{stim\}\.mkv'",
            "video_path = str(Path(os.environ.get('NATURALISTIC_ENCODING_STIMULI', '../stimuli')) / 'friends.stimuli' / 's1' / f'friends_{stim}.mkv')",
            src,
        )
        src = re.sub(
            r"video_path = f'/om2/scratch/tmp/jsmentch/nat_asd/data/\{stim\}\.mp4'",
            "video_path = str(Path(os.environ.get('NATURALISTIC_ENCODING_DATA', '../data')) / f'{stim}.mp4')",
            src,
        )
        src = re.sub(
            r"video_file = f'/om2/scratch/tmp/jsmentch/nat_img/sourcedata/data/cneuromod/friends\.stimuli/s1/\{stim\}\.mkv'",
            "video_file = str(Path(os.environ.get('NATURALISTIC_ENCODING_STIMULI', '../stimuli')) / 'friends.stimuli' / 's1' / f'{stim}.mkv')",
            src,
        )
        if ("Path(os.environ" in src or " Path(" in src) and "from pathlib import Path" not in src:
            src = "from pathlib import Path\nimport os\n" + src

    if name == "05-pilot_encoding.ipynb":
        src = src.replace(
            "pilot_subjects=pd.read_csv('../data/pilots_ru_dm.csv') # load pilot subjects\n"
            "sub=pilot_subjects['Identifiers_y'].iloc[1] #get first subject\n"
            "\n"
            "im_file = os.environ['NATURALISTIC_ENCODING_HBN_PTEMPLATE'].format(sub=sub)  # set env; no hard-coded paths",
            "import os\n"
            "pilot_subjects = pd.read_csv('../data/pilots_ru_dm.csv')\n"
            "assert 'participant_id' in pilot_subjects.columns, 'use de-identified participant_id column'\n"
            "sub = str(pilot_subjects['participant_id'].iloc[0])\n"
            "im_file = os.environ['NATURALISTIC_ENCODING_HBN_PTEMPLATE'].format(sub=sub)",
        )
        src = src.replace(
            "for sub in list(pilot_subjects['Identifiers_y']):",
            "for sub in pilot_subjects['participant_id'].astype(str):",
        )
        src = re.sub(
            r"im_file = f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/xcp_d_[^']+/sub-\{sub\}/[^']+'",
            "im_file = os.environ['NATURALISTIC_ENCODING_HBN_PTEMPLATE'].format(sub=sub)",
            src,
        )
        if "os.environ['NATURALISTIC_ENCODING_HBN_PTEMPLATE']" in src and "import os" not in src:
            src = "import os\n" + src

    src = re.sub(
        r"/net/vast-storage\.ib\.cluster/[^\s'\"`]+",
        "os.environ.get('NATURALISTIC_ENCODING_MODEL_CHECKPOINT', 'checkpoint.pt')",
        src,
    )

    return src


def process_file(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    setup_key = "NATURALISTIC_PATH_SETUP"
    first_code_done = False
    for cell in nb["cells"]:
        ct = cell["cell_type"]
        raw = "".join(cell.get("source", []))
        if ct == "code":
            clear_code_cell(cell)
            raw = strip_commented_code(raw)
            raw = redact_sources(path, raw)
            if not raw.strip():
                continue
            if path.name in (
                "02-preprocess_stimuli.ipynb",
                "03-video_feature_extraction.ipynb",
            ) and not first_code_done:
                cell0 = raw.lstrip()
                if setup_key not in cell0:
                    pre = (
                        "# " + setup_key + "\n"
                        "from pathlib import Path\n"
                        "import os\n"
                        "STIMULI = Path(os.environ.get('NATURALISTIC_ENCODING_STIMULI', '../stimuli'))\n"
                        "DATA = Path(os.environ.get('NATURALISTIC_ENCODING_DATA', '../data'))\n\n"
                    )
                    raw = pre + raw
                first_code_done = True
            cell["source"] = norm_source(raw)
        else:
            if not raw.strip():
                continue
            cell["source"] = norm_source(raw)
        out.append(cell)
    nb["cells"] = out
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def write_qc_stub() -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Quality control and cohort selection\n",
                    "\n",
                    "QC on restricted cohorts is done locally (motion scrubbing, site flags, matching). **No** participant ages, sex, or study GUIDs appear in this repository.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Synthetic cohort table\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "\n",
                    "rng = np.random.default_rng(42)\n",
                    "n = 24\n",
                    "participants = pd.DataFrame(\n",
                    "    {\n",
                    "        \"participant_id\": [f\"anon_{i:03d}\" for i in range(n)],\n",
                    "        \"qc_include\": rng.random(n) > 0.15,\n",
                    "    }\n",
                    ")\n",
                    "cohort = participants.loc[participants[\"qc_include\"]].copy()\n",
                    "cohort.head()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "For real preprocessing, keep tables local and use anonymous `participant_id` only. See [docs/data_layout.md](../docs/data_layout.md).\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    QC_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    write_qc_stub()
    for fn in [
        "02-preprocess_stimuli.ipynb",
        "03-video_feature_extraction.ipynb",
        "04-audio_feature_extraction.ipynb",
        "05-pilot_encoding.ipynb",
    ]:
        p = NB_DIR / fn
        if p.exists():
            process_file(p)


if __name__ == "__main__":
    main()
