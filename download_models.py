"""Download required OpenVINO models into the project.

Usage:
    python download_models.py

This script downloads the following Open Model Zoo models (FP16):
- face-detection-adas-0001
- landmarks-regression-retail-0009
- head-pose-estimation-adas-0001
- gaze-estimation-adas-0002

It stores them under the configured models directory (default: models/).
"""

import json
import os
from pathlib import Path
from urllib import request
from urllib.error import URLError, HTTPError

DEFAULT_MODELS = [
    "face-detection-adas-0001",
    "landmarks-regression-retail-0009",
    "head-pose-estimation-adas-0001",
    "gaze-estimation-adas-0002",
]

BASE_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2024.0/models_bin/1"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            prefix = dest.read_bytes()[:64].lstrip().lower()
            if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
                print(f"    replacing invalid placeholder file: {dest.name}")
            else:
                return
        except OSError:
            pass
    try:
        with request.urlopen(url) as response, open(dest, "wb") as out_file:
            out_file.write(response.read())
    except HTTPError as e:
        raise RuntimeError(f"Failed to download {url} (HTTP {e.code})")
    except URLError as e:
        raise RuntimeError(f"Failed to download {url} ({e.reason})")


def download_model(model_name: str, download_dir: Path, precision: str = "FP16") -> None:
    model_dir = download_dir / model_name / precision
    xml_url = f"{BASE_URL}/{model_name}/{precision}/{model_name}.xml"
    bin_url = f"{BASE_URL}/{model_name}/{precision}/{model_name}.bin"

    print(f"  - downloading {model_name} ({precision})")
    _download_file(xml_url, model_dir / f"{model_name}.xml")
    _download_file(bin_url, model_dir / f"{model_name}.bin")


def main():
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config" / "default_config.json"

    download_dir = project_root / "models"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            download_dir = project_root / cfg.get("models", {}).get("download_dir", "models")
    except Exception:
        pass

    download_dir.mkdir(exist_ok=True, parents=True)
    precision = "FP16"

    print(f"Downloading models to: {download_dir}")

    for model_name in DEFAULT_MODELS:
        download_model(model_name, download_dir, precision=precision)

    print("Done.")


if __name__ == "__main__":
    main()
