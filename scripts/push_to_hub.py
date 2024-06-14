import logging

from transformers import CLIPProcessor

from aesthetics_predictor.configuration_predictor import AestheticsPredictorConfig
from aesthetics_predictor.modeling_v1 import URLS as V1_URLS
from aesthetics_predictor.modeling_v1 import convert_from_openai_clip
from aesthetics_predictor.modeling_v2 import URLS_LINEAR as URLS_V2_LINEAR
from aesthetics_predictor.modeling_v2 import URLS_RELU as URLS_V2_RELU
from aesthetics_predictor.modeling_v2 import (
    convert_v2_linear_from_openai_clip,
    convert_v2_relu_from_openai_clip,
)
from aesthetics_predictor.utils import get_model_name_for_v1, get_model_name_for_v2

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def push_aesthetics_predictor_v1() -> None:
    for openai_model_name in V1_URLS.keys():
        config = AestheticsPredictorConfig.from_pretrained(openai_model_name)
        assert isinstance(config, AestheticsPredictorConfig)

        model = convert_from_openai_clip(openai_model_name, config=config)

        processor = CLIPProcessor.from_pretrained(openai_model_name)

        model_name = get_model_name_for_v1(openai_model_name)
        config.register_for_auto_class()
        model.register_for_auto_class()

        logger.info(f"Push model to the hub: {model_name}")
        model.push_to_hub(model_name, private=True)

        logger.info(f"Push processor to the hub: {model_name}")
        processor.push_to_hub(model_name, private=True)


def push_aesthetics_predictor_v2(
    openai_model_name: str = "openai/clip-vit-large-patch14",
) -> None:
    def push_aesthetics_predictor_v2_linear(
        config: AestheticsPredictorConfig,
        processor: CLIPProcessor,
    ):
        for predictor_head_name in URLS_V2_LINEAR.keys():
            model = convert_v2_linear_from_openai_clip(
                predictor_head_name=predictor_head_name,
                openai_model_name=openai_model_name,
                config=config,
            )

            model_name = get_model_name_for_v2(predictor_head_name)
            logger.info(f"Push model to the hub: {model_name}")
            model.register_for_auto_class()
            model.push_to_hub(model_name, private=True)

            logger.info(f"Push processor to the hub: {model_name}")
            processor.push_to_hub(model_name, private=True)

    def push_aesthetics_predictor_v2_relu(
        config: AestheticsPredictorConfig, processor: CLIPProcessor
    ):
        for predictor_head_name in URLS_V2_RELU.keys():
            model = convert_v2_relu_from_openai_clip(
                predictor_head_name=predictor_head_name,
                openai_model_name=openai_model_name,
                config=config,
            )

            model_name = get_model_name_for_v2(predictor_head_name)
            logger.info(f"Push model to the hub: {model_name}")
            model.register_for_auto_class()
            model.push_to_hub(model_name, private=True)

            logger.info(f"Push processor to the hub: {model_name}")
            processor.push_to_hub(model_name, private=True)

    config = AestheticsPredictorConfig.from_pretrained(openai_model_name)
    assert isinstance(config, AestheticsPredictorConfig)

    processor = CLIPProcessor.from_pretrained(openai_model_name)

    push_aesthetics_predictor_v2_linear(config=config, processor=processor)
    push_aesthetics_predictor_v2_relu(config=config, processor=processor)


def main():
    push_aesthetics_predictor_v1()
    push_aesthetics_predictor_v2()


if __name__ == "__main__":
    main()
