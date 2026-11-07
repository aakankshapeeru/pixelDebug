"""Real dashboard → proxy → API upload and export smoke test."""

from pathlib import Path
import json
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context(
        viewport={"width": 1440, "height": 1080},
        device_scale_factor=1,
        record_video_dir=str(root / "data/demo"),
    )
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto("http://127.0.0.1:5173")
    page.get_by_text("API connected", exact=True).wait_for()
    page.screenshot(path=str(root / "docs/dashboard-empty.png"), full_page=True)
    inputs = page.locator("input[type=file]")
    inputs.nth(0).set_input_files(root / "data/synthetic/family-000/reference.png")
    inputs.nth(1).set_input_files(root / "data/synthetic/family-000/color_change.png")
    page.get_by_role("button", name="Analyze screenshots").click()
    page.get_by_text("DETECTED REGIONS", exact=True).wait_for()
    page.get_by_role("button", name="Select defect 1", exact=True).click()
    page.get_by_text("Selected region #1", exact=False).wait_for()
    with page.expect_download() as download_info:
        page.get_by_role("button", name="Export JSON").click()
    download = download_info.value
    download.save_as(root / "docs/example-report.json")
    report = json.loads((root / "docs/example-report.json").read_text())
    assert report["defects"] and report["diagnosis_source"] == "deterministic_cv"
    page.evaluate("window.scrollTo(0,0)")
    page.screenshot(path=str(root / "docs/dashboard-results.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.screenshot(path=str(root / "docs/dashboard-mobile.png"), full_page=True)
    assert not errors, errors
    print(
        json.dumps(
            {
                "browser_errors": errors,
                "defects": len(report["defects"]),
                "export": "passed",
                "mobile_overflow": False,
            }
        )
    )
    context.close()
    page.video.save_as(str(root / "docs/demo.webm"))
    browser.close()
