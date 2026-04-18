from __future__ import annotations

from photo_reviver.types import ImageAnalysis, RestorationDecision


def choose_restoration_mode(analysis: ImageAnalysis) -> RestorationDecision:
    reasons: list[str] = []

    if analysis.scratch_severity == "high" and analysis.needs_high_resolution_path:
        mode = "scratch+hr"
        reasons.append("Strong scratch pattern and low-resolution input were both detected.")
    elif analysis.scratch_severity in {"medium", "high"}:
        mode = "scratch"
        reasons.append("Visible scratch pattern was detected.")
    else:
        mode = "normal"
        reasons.append("Scratch pattern looks light, so the normal path is enough.")

    if analysis.low_contrast:
        reasons.append("Image contrast is low.")
    if analysis.needs_high_resolution_path and mode == "normal":
        reasons.append("An HR path may still be useful later if you add an upscaler.")
    if analysis.face_detected:
        reasons.append("A face was detected, so preserving facial detail matters.")

    return RestorationDecision(mode=mode, reasons=reasons)
