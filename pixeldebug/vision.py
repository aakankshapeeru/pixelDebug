"""Deterministic visual evidence; category labels are heuristics, not AI claims."""

from __future__ import annotations
import base64
import io
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.metrics import structural_similarity


def decode_image(data: bytes, max_pixels: int = 12_000_000) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Use PNG, JPEG, or WebP screenshots")
            if image.width * image.height > max_pixels or min(image.size) < 8:
                raise ValueError(
                    "Image dimensions exceed limits or are smaller than 8 pixels"
                )
            image.load()
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image.convert("RGBA"))
            return np.array(background.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or unsafe image") from exc


def png_data(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def compare(
    reference: np.ndarray,
    defective: np.ndarray,
    sensitivity: float = 0.5,
    min_area: int = 24,
    align: bool = False,
) -> dict:
    if reference.shape != defective.shape:
        raise ValueError("Screenshot dimensions must match; capture the same viewport")
    if not 0 <= sensitivity <= 1:
        raise ValueError("Sensitivity must be between 0 and 1")
    height, width = reference.shape[:2]
    gray_a = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(defective, cv2.COLOR_RGB2GRAY)
    valid = np.ones((height, width), dtype=np.uint8)
    alignment = {"requested": align, "applied": False, "translation": [0.0, 0.0]}
    if align and gray_a.std() > 1 and gray_b.std() > 1:
        shift, response = cv2.phaseCorrelate(
            gray_a.astype(np.float32), gray_b.astype(np.float32)
        )
        if response > 0.5 and max(abs(shift[0]), abs(shift[1])) <= 8:
            matrix = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
            defective = cv2.warpAffine(
                defective, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
            )
            valid = cv2.warpAffine(
                valid, matrix, (width, height), flags=cv2.INTER_NEAREST
            )
            alignment.update(applied=True, translation=list(shift))
    score, similarity = structural_similarity(
        reference, defective, channel_axis=2, data_range=255, full=True
    )
    perceptual = 1 - similarity.mean(axis=2)
    delta = np.max(
        np.abs(reference.astype(np.int16) - defective.astype(np.int16)), axis=2
    )
    mask = (
        (delta > (40 - 32 * sensitivity))
        & (perceptual > (0.18 - 0.15 * sensitivity))
        & (valid > 0)
    ).astype(np.uint8)
    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    overlay = defective.copy()
    overlay[mask > 0] = (
        0.45 * overlay[mask > 0] + 0.55 * np.array([255, 66, 103])
    ).astype(np.uint8)
    defects = []
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if area < min_area:
            continue
        region = labels == index
        changed = int(np.count_nonzero(mask & region))
        if not changed:
            continue
        a = reference[y : y + h, x : x + w]
        b = defective[y : y + h, x : x + w]
        edge_a = cv2.Canny(a, 80, 160) > 0
        edge_b = cv2.Canny(b, 80, 160) > 0
        union = np.count_nonzero(edge_a | edge_b)
        agreement = np.count_nonzero(edge_a & edge_b) / max(union, 1)
        category = "color_change" if agreement > 0.65 else "layout_or_content_change"
        explanation = (
            "Similar edge structure with changed pixels; inspect color or background styles."
            if category == "color_change"
            else "Edge structure changed; inspect positioning, text, missing content, or overflow."
        )
        defects.append(
            {
                "id": len(defects) + 1,
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "category": category,
                "source": "cv_heuristic",
                "changed_pixels": changed,
                "explanation": explanation,
            }
        )
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (255, 66, 103), 2)
    return {
        "width": width,
        "height": height,
        "ssim": float(score),
        "changed_fraction": float(mask.sum() / max(valid.sum(), 1)),
        "alignment": alignment,
        "defects": defects,
        "overlay": png_data(overlay),
        "diagnosis_source": "deterministic_cv",
        "warnings": [
            "Categories are heuristic suggestions, not calibrated predictions.",
            "Opt-in alignment can hide genuine global layout shifts.",
        ],
    }
