import pytest
import requests
from PIL import Image
from PIL.Image import Image as PilImage


@pytest.fixture
def image_url() -> str:
    # the image from https://github.com/LAION-AI/aesthetic-predictor/blob/main/asthetics_predictor.ipynb
    url = "https://thumbs.dreamstime.com/b/lovely-cat-as-domestic-animal-view-pictures-182393057.jpg"
    return url


@pytest.fixture
def sample_image(image_url: str) -> PilImage:
    return Image.open(requests.get(image_url, stream=True).raw)
