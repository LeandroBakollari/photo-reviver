from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from photo_reviver.io_utils import ensure_color, save_image, save_json
from photo_reviver.types import ImageAnalysis, ImageValidation


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_histogram(gray_image: np.ndarray) -> np.ndarray:
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    return histogram.flatten()


def draw_histogram(histogram: np.ndarray, width: int = 512, height: int = 280) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    normalized = cv2.normalize(histogram, None, 0, height - 30, cv2.NORM_MINMAX).flatten()
    points = []
    for index, value in enumerate(normalized):
        x = int(index * (width - 1) / 255)
        y = height - 10 - int(value)
        points.append((x, y))

    for start, end in zip(points[:-1], points[1:]):
        cv2.line(canvas, start, end, (40, 40, 40), 2)

    return canvas


def detect_low_contrast(
    gray_image: np.ndarray,
    std_threshold: float,
    dynamic_range_threshold: int,
) -> tuple[bool, float, int]:
    brightness_std = float(np.std(gray_image))
    dynamic_range = int(gray_image.max() - gray_image.min())
    low_contrast = (
        brightness_std < std_threshold or dynamic_range < dynamic_range_threshold
    )
    return low_contrast, brightness_std, dynamic_range


def estimate_scratch_severity(
    gray_image: np.ndarray,
    medium_threshold: float,
    high_threshold: float,
) -> tuple[float, str, np.ndarray, str]:
    # Keep this stage simple and readable: a blackhat highlights thin dark damage.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
    _, scratch_mask = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    scratch_mask = cv2.medianBlur(scratch_mask, 3)

    scratch_ratio = float(np.count_nonzero(scratch_mask) / scratch_mask.size)
    if scratch_ratio >= high_threshold:
        severity = "high"
    elif scratch_ratio >= medium_threshold:
        severity = "medium"
    else:
        severity = "low"

    return scratch_ratio, severity, scratch_mask, "blackhat threshold heuristic"


def summarize_scratch_mask(
    scratch_mask: np.ndarray,
    medium_threshold: float,
    high_threshold: float,
) -> tuple[float, str]:
    scratch_ratio = float(np.count_nonzero(scratch_mask) / scratch_mask.size)
    if scratch_ratio >= high_threshold:
        severity = "high"
    elif scratch_ratio >= medium_threshold:
        severity = "medium"
    else:
        severity = "low"
    return scratch_ratio, severity


def estimate_scratch_severity_with_boptl(
    image: np.ndarray,
    restoration_config: dict,
    stage_dir: Path,
    medium_threshold: float,
    high_threshold: float,
) -> tuple[float, str, np.ndarray, str]:
    repo_root = Path(restoration_config["repo_root"]).resolve()
    detection_script = repo_root / "Global" / "detection.py"
    python_executable = str(restoration_config.get("python_executable", "python"))
    gpu = str(restoration_config.get("gpu", "-1"))

    detection_input_dir = stage_dir / "boptl_detection_input"
    detection_output_dir = stage_dir / "boptl_detection_output"
    detection_input_dir.mkdir(parents=True, exist_ok=True)
    detection_output_dir.mkdir(parents=True, exist_ok=True)

    input_path = save_image(detection_input_dir / "analysis_input.png", image)
    command = [
        python_executable,
        str(detection_script),
        "--test_path",
        str(detection_input_dir),
        "--output_dir",
        str(detection_output_dir),
        "--input_size",
        "full_size",
        "--GPU",
        gpu,
    ]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root / "Global",
    )
    log_text = completed.stdout
    if completed.stderr:
        log_text = f"{log_text}\n\n[stderr]\n{completed.stderr}".strip()
    (stage_dir / "scratch_detection.log").write_text(log_text, encoding="utf-8")

    mask_path = detection_output_dir / "mask" / f"{input_path.stem}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Microsoft scratch detector did not create mask: {mask_path}")

    scratch_mask = load_image(mask_path)
    if scratch_mask.ndim == 3:
        scratch_mask = cv2.cvtColor(scratch_mask, cv2.COLOR_BGR2GRAY)
    _, scratch_mask = cv2.threshold(scratch_mask, 127, 255, cv2.THRESH_BINARY)

    target_height, target_width = image.shape[:2]
    if scratch_mask.shape[:2] != (target_height, target_width):
        scratch_mask = cv2.resize(
            scratch_mask,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    scratch_ratio, severity = summarize_scratch_mask(
        scratch_mask,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )
    return scratch_ratio, severity, scratch_mask, "Microsoft U-Net scratch detector"


def create_scratch_overlay(image: np.ndarray, scratch_mask: np.ndarray) -> np.ndarray:
    overlay = ensure_color(image).copy()
    highlight = overlay.copy()
    highlight[scratch_mask > 0] = (40, 40, 220)
    return cv2.addWeighted(overlay, 0.72, highlight, 0.28, 0.0)


def detect_faces(gray_image: np.ndarray) -> tuple[bool, int, str]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return False, 0, "OpenCV cascade not found"

    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return False, 0, "OpenCV cascade could not be loaded"

    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(36, 36),
    )
    count = int(len(faces))
    return count > 0, count, "OpenCV Haar cascade"


def decide_high_resolution_path(
    validation: ImageValidation,
    face_count: int,
    longest_side_threshold: int,
) -> bool:
    longest_side = max(validation.width, validation.height)
    shortest_side = min(validation.width, validation.height)

    if longest_side < longest_side_threshold:
        return True

    if face_count > 0 and shortest_side < 900:
        return True

    return False


def analyze_image(
    image: np.ndarray,
    validation: ImageValidation,
    analysis_config: dict,
    stage_dir: Path,
    restoration_config: dict | None = None,
) -> ImageAnalysis:
    gray_image = to_grayscale(image)
    grayscale_path = save_image(stage_dir / "grayscale.png", gray_image)

    histogram = compute_histogram(gray_image)
    histogram_path = save_image(stage_dir / "histogram.png", draw_histogram(histogram))

    low_contrast, brightness_std, dynamic_range = detect_low_contrast(
        gray_image,
        std_threshold=float(analysis_config["low_contrast_std_threshold"]),
        dynamic_range_threshold=int(
            analysis_config["low_contrast_dynamic_range_threshold"]
        ),
    )

    medium_threshold = float(analysis_config["scratch_ratio_thresholds"]["medium"])
    high_threshold = float(analysis_config["scratch_ratio_thresholds"]["high"])
    scratch_notes: list[str] = []
    try:
        if (
            restoration_config
            and restoration_config.get("backend") == "boptl"
            and Path(restoration_config["repo_root"]).exists()
        ):
            scratch_ratio, scratch_severity, scratch_mask, scratch_detection_method = (
                estimate_scratch_severity_with_boptl(
                    image=image,
                    restoration_config=restoration_config,
                    stage_dir=stage_dir,
                    medium_threshold=medium_threshold,
                    high_threshold=high_threshold,
                )
            )
        else:
            scratch_ratio, scratch_severity, scratch_mask, scratch_detection_method = (
                estimate_scratch_severity(
                    gray_image,
                    medium_threshold=medium_threshold,
                    high_threshold=high_threshold,
                )
            )
    except Exception as error:
        scratch_ratio, scratch_severity, scratch_mask, scratch_detection_method = estimate_scratch_severity(
            gray_image,
            medium_threshold=medium_threshold,
            high_threshold=high_threshold,
        )
        scratch_notes.append(
            "Microsoft scratch detector was unavailable, so the app fell back to the OpenCV heuristic."
        )
        scratch_notes.append(f"Detector fallback reason: {error}")

    scratch_mask_path = save_image(stage_dir / "scratch_mask.png", scratch_mask)
    scratch_overlay_path = save_image(
        stage_dir / "scratch_overlay.png",
        create_scratch_overlay(image, scratch_mask),
    )

    face_detected, face_count, face_method = detect_faces(gray_image)
    needs_hr = decide_high_resolution_path(
        validation,
        face_count=face_count,
        longest_side_threshold=int(
            analysis_config["small_image_longest_side_threshold"]
        ),
    )

    notes: list[str] = []
    if low_contrast:
        notes.append("Low contrast detected from brightness spread and dynamic range.")
    if scratch_severity != "low":
        notes.append(f"Scratch severity estimated as {scratch_severity}.")
    if face_detected:
        notes.append(f"Detected {face_count} face(s) with a classical OpenCV detector.")
    if needs_hr:
        notes.append("Image is small enough that a high-resolution path may help later.")
    notes.extend(scratch_notes)

    analysis = ImageAnalysis(
        grayscale_path=grayscale_path.resolve(),
        histogram_path=histogram_path.resolve(),
        scratch_mask_path=scratch_mask_path.resolve(),
        scratch_overlay_path=scratch_overlay_path.resolve(),
        brightness_mean=float(np.mean(gray_image)),
        brightness_std=brightness_std,
        dynamic_range=dynamic_range,
        low_contrast=low_contrast,
        scratch_ratio=scratch_ratio,
        scratch_severity=scratch_severity,
        scratch_detection_method=scratch_detection_method,
        face_detected=face_detected,
        face_count=face_count,
        face_detection_method=face_method,
        needs_high_resolution_path=needs_hr,
        notes=notes,
    )

    save_json(stage_dir / "analysis.json", analysis)
    return analysis
