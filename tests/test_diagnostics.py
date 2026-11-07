import json
import pytest
from pixeldebug.diagnostics import validate_diagnosis


def test_generated_schema():
    good = {
        "defects": [{"category": "spacing", "bbox": dict(x=1, y=2, width=3, height=4)}]
    }
    assert validate_diagnosis(json.dumps(good), 20, 20) == good
    with pytest.raises(ValueError):
        validate_diagnosis(json.dumps(good), 2, 2)
    with pytest.raises(ValueError):
        validate_diagnosis("not JSON", 20, 20)
    good["defects"][0]["category"] = "invented"
    with pytest.raises(ValueError):
        validate_diagnosis(json.dumps(good), 20, 20)
