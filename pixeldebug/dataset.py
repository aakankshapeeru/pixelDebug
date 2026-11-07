"""Render controlled defects; split by layout family, never individual screenshot."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

CATEGORIES = [
    "color_change",
    "missing_element",
    "misalignment",
    "overflow",
    "typography",
    "spacing",
]


def split_for(family: int) -> str:
    # All variants of a column-layout family stay in the same split.
    return ["train", "validation", "test"][family % 3]


def template(family: int, category: str | None = None) -> str:
    columns = 2 + family % 3
    gap = 12 + (family // 3 % 4) * 6
    color = ["#6554e8", "#087e8b", "#bb4e72", "#b56320"][family % 4]
    change = {
        "color_change": "#target {background:#e65050}",
        "missing_element": "#target {visibility:hidden}",
        "misalignment": "#target {transform:translate(18px,12px)}",
        "overflow": "#target {width:550px}",
        "typography": "#target {font-size:27px;font-weight:900}",
        "spacing": "#target {padding:36px}",
    }.get(category, "")
    cards = "".join(
        f'<article id="{"target" if i == 0 else f"card{i}"}"><b>Component {i + 1}</b><p>Inspect interface changes.</p></article>'
        for i in range(columns * 2)
    )
    return f"""<!doctype html><html><head><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f2f4f8;color:#192238;font:16px Arial}}
    header{{background:#192238;color:white;padding:{24 + family % 5 * 3}px 32px}}
    main{{padding:32px}}section{{display:grid;grid-template-columns:repeat({columns},1fr);gap:{gap}px}}
    article{{background:white;border-top:5px solid {color};border-radius:8px;padding:20px;min-height:120px}}
    #target{{background:{color};color:white}}{change}
    </style></head><body><header><h1>Workspace {family}</h1><p>Visual regression fixture</p></header><main><section>{cards}</section></main></body></html>"""


def union_box(a: dict, b: dict) -> dict:
    x, y = min(a["x"], b["x"]), min(a["y"], b["y"])
    return dict(
        x=int(x),
        y=int(y),
        width=int(max(a["x"] + a["width"], b["x"] + b["width"]) - x),
        height=int(max(a["y"] + a["height"], b["y"] + b["height"]) - y),
    )


def generate(output: Path, families: int = 20):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 960, "height": 720}, device_scale_factor=1
        )
        for family in range(families):
            folder = output / f"family-{family:03d}"
            folder.mkdir(exist_ok=True)
            page.set_content(template(family))
            page.screenshot(path=str(folder / "reference.png"))
            before = page.locator("#target").bounding_box()
            for category in CATEGORIES:
                page.set_content(template(family, category))
                after = page.locator("#target").bounding_box()
                page.screenshot(path=str(folder / f"{category}.png"))
                record = {
                    "id": f"{family}-{category}",
                    "family": f"grid-{2 + family % 3}-columns",
                    "variant": family,
                    "split": split_for(family),
                    "reference": f"{folder.name}/reference.png",
                    "defective": f"{folder.name}/{category}.png",
                    "category": category,
                    "bbox": union_box(before, after),
                    "annotation_scope": "target element union; reflow may affect neighboring elements",
                    "diagnosis": {
                        "defects": [
                            {"category": category, "bbox": union_box(before, after)}
                        ]
                    },
                }
                records.append(record)
        browser.close()
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--families", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "examples": len(generate(args.output, args.families)),
                "output": str(args.output),
            }
        )
    )
