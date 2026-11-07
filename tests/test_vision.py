import io
import numpy as np
import pytest
from PIL import Image
from pixeldebug.vision import compare, decode_image


def test_identical():
    image = np.full((100, 120, 3), 255, np.uint8)
    result = compare(image, image)
    assert result['defects'] == []
    assert result['ssim'] == 1
    assert result['changed_fraction'] == 0


def test_localization():
    a = np.full((100, 120, 3), 255, np.uint8)
    b = a.copy()
    b[30:60, 40:80] = [30, 80, 200]
    result = compare(a, b)
    assert len(result['defects']) == 1
    assert result['defects'][0]['bbox'] == dict(x=40, y=30, width=40, height=30)
    assert result['changed_fraction'] == pytest.approx(.1)


def test_invalid_and_dimensions():
    with pytest.raises(ValueError):
        decode_image(b'not an image')
    with pytest.raises(ValueError):
        compare(np.zeros((10, 10, 3), np.uint8), np.zeros((20, 20, 3), np.uint8))
    buffer = io.BytesIO()
    Image.new('RGB', (20, 20)).save(buffer, 'PNG')
    with pytest.raises(ValueError):
        decode_image(buffer.getvalue(), max_pixels=100)


def test_transparency_composited_on_white():
    buffer = io.BytesIO()
    Image.new('RGBA', (10, 10), (0, 0, 0, 0)).save(buffer, 'PNG')
    assert np.all(decode_image(buffer.getvalue()) == 255)
