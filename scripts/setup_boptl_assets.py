from __future__ import annotations

import bz2
import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT / "external" / "bringing-old-photos-back-to-life"

LANDMARK_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
FACE_CHECKPOINTS_URL = (
    "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/"
    "download/v1.0/face_checkpoints.zip"
)
GLOBAL_CHECKPOINTS_URL = (
    "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/"
    "download/v1.0/global_checkpoints.zip"
)


def download_file(url: str, target_path: Path) -> None:
    if target_path.exists():
        print(f"Exists: {target_path}")
        return

    print(f"Downloading: {url}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target_path)


def copy_sync_batchnorm() -> None:
    source = (
        REPO_ROOT
        / "Face_Enhancement"
        / "models"
        / "networks"
        / "Synchronized-BatchNorm-PyTorch"
        / "sync_batchnorm"
    )
    if not source.exists():
        raise FileNotFoundError(
            "Could not find the cloned Synchronized-BatchNorm-PyTorch repo under "
            f"{source.parent}"
        )

    destinations = [
        REPO_ROOT / "Face_Enhancement" / "models" / "networks" / "sync_batchnorm",
        REPO_ROOT / "Global" / "detection_models" / "sync_batchnorm",
    ]

    for destination in destinations:
        if destination.exists():
            print(f"Exists: {destination}")
            continue
        shutil.copytree(source, destination)
        print(f"Copied: {destination}")


def extract_bz2(source: Path, target: Path) -> None:
    if target.exists():
        print(f"Exists: {target}")
        return

    with bz2.open(source, "rb") as compressed, target.open("wb") as extracted:
        shutil.copyfileobj(compressed, extracted)
    print(f"Extracted: {target}")


def extract_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source, "r") as archive:
        archive.extractall(destination)
    print(f"Extracted: {source} -> {destination}")


def main() -> None:
    if not REPO_ROOT.exists():
        raise FileNotFoundError(
            "The external BOPTL repo is missing. Expected it at "
            f"{REPO_ROOT}"
        )

    copy_sync_batchnorm()

    landmark_bz2 = REPO_ROOT / "Face_Detection" / "shape_predictor_68_face_landmarks.dat.bz2"
    landmark_dat = REPO_ROOT / "Face_Detection" / "shape_predictor_68_face_landmarks.dat"
    face_zip = REPO_ROOT / "Face_Enhancement" / "face_checkpoints.zip"
    global_zip = REPO_ROOT / "Global" / "global_checkpoints.zip"

    download_file(LANDMARK_URL, landmark_bz2)
    download_file(FACE_CHECKPOINTS_URL, face_zip)
    download_file(GLOBAL_CHECKPOINTS_URL, global_zip)

    extract_bz2(landmark_bz2, landmark_dat)
    extract_zip(face_zip, REPO_ROOT / "Face_Enhancement")
    extract_zip(global_zip, REPO_ROOT / "Global")

    print("BOPTL assets are ready.")


if __name__ == "__main__":
    main()
