"""Strict schema for experimental generated diagnoses; never trust raw LM JSON."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class GeneratedBox(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class GeneratedDefect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal[
        "color_change",
        "missing_element",
        "misalignment",
        "overflow",
        "typography",
        "spacing",
    ]
    bbox: GeneratedBox


class GeneratedDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    defects: list[GeneratedDefect] = Field(max_length=100)


def validate_diagnosis(text: str, width: int, height: int) -> dict:
    result = GeneratedDiagnosis.model_validate_json(text)
    for defect in result.defects:
        box = defect.bbox
        if box.x + box.width > width or box.y + box.height > height:
            raise ValueError("Generated box is outside image bounds")
    return result.model_dump()
