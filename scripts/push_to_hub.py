import logging

import torch
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1
from aesthetics_predictor.utils import get_model_name
from aesthetics_predictor.v1 import URLS as V1_URLS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def main():
    for openai_model_name, url in V1_URLS.items():
        processor = CLIPProcessor.from_pretrained(openai_model_name)
        model = AestheticsPredictorV1.from_pretrained(openai_model_name)
        state_dict = torch.hub.load_state_dict_from_url(url)
        model.predictor.load_state_dict(state_dict)

        model_name = get_model_name(openai_model_name, version=1)
        logger.info(f"Push model to the hub: {model_name}")
        model.push_to_hub(model_name, private=True)

        logger.info(f"Push processor to the hub: {model_name}")
        processor.push_to_hub(model_name, private=True)


if __name__ == "__main__":
    main()
