import pytest
import requests
import torch
from PIL import Image
from PIL.Image import Image as PilImage
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1
from aesthetics_predictor.v1 import URLS


@pytest.fixture
def sample_image() -> PilImage:
    # return Image.open(
    #     requests.get(
    #         "https://thumbs.dreamstime.com/b/lovely-cat-as-domestic-animal-view-pictures-182393057.jpg",
    #         stream=True,
    #     ).raw
    # )
    return Image.open("lovely-cat-as-domestic-animal-view-pictures-182393057.jpg")


@pytest.mark.parametrize(
    argnames="model_name, expected_img_embeds, expected_prediction",
    argvalues=(
        (
            "openai/clip-vit-base-patch16",
            1.5450,
            4.3533,
        ),
        (
            "openai/clip-vit-base-patch32",
            -0.4287,
            4.4723,
        ),
        (
            "openai/clip-vit-large-patch14",
            0.2653,
            5.0491,
        ),
    ),
)
def test_aesthetics_predictor_v1(
    model_name: str,
    expected_img_embeds: float,
    expected_prediction: float,
    sample_image: PilImage,
) -> None:
    processor = CLIPProcessor.from_pretrained(model_name)

    model = AestheticsPredictorV1.from_pretrained(model_name)
    state_dict = torch.hub.load_state_dict_from_url(URLS[model_name])
    model.predictor.load_state_dict(state_dict)
    model.eval()

    inputs = processor(images=sample_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    actual_img_embeds = outputs.hidden_states.sum().item()
    actual_prediction = outputs.logits.item()

    assert actual_img_embeds == pytest.approx(expected_img_embeds, 0.1)
    assert actual_prediction == pytest.approx(expected_prediction, 0.1)