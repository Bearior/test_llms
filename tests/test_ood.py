"""Out-of-distribution input must route to HUMAN_REVIEW (the 'unknown' path).

This exercises the abstention route with a genuinely off-distribution image
(random noise, no matching manifest record), not merely an invalid family
string.
"""
import numpy as np
from PIL import Image

from thaidoc import confidence
from thaidoc.config import DATA_DIR
from thaidoc.labels import UNKNOWN_TYPE
from thaidoc.synth import image_path


def test_ood_noise_image_routes_to_human_review(pipeline_and_rows):
    pipe, _ = pipeline_and_rows
    # Synthesize an OOD image that is in no manifest row.
    img_dir = DATA_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    noise = (rng.random((360, 520, 3)) * 255).astype("uint8")
    fname = "__ood_noise__.jpg"
    Image.fromarray(noise).save(img_dir / fname, "JPEG")

    # record=None -> MockReader has no text -> Stage-2 abstains.
    c = pipe.classify(image_path(fname), record=None)
    assert c.routing == confidence.ROUTE_HUMAN_REVIEW
    assert c.physical_type == UNKNOWN_TYPE


def test_image_path_rejects_traversal():
    import pytest
    with pytest.raises(ValueError):
        image_path("..\\..\\Windows\\win.ini")
