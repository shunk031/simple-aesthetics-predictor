import logging
from typing import Dict, Union

import gradio as gr
import torch
from PIL.Image import Image as PilImage
from transformers import CLIPProcessor

from aesthetics_predictor import (
    AestheticsPredictorV1,
    AestheticsPredictorV2Linear,
    AestheticsPredictorV2ReLU,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def load_predictor(
    model_name: str,
) -> Union[
    AestheticsPredictorV1, AestheticsPredictorV2Linear, AestheticsPredictorV2ReLU
]:
    logger.info(f"Try to load the following model: {model_name}")
    if "v1" in model_name:
        return AestheticsPredictorV1.from_pretrained(model_name)
    elif "v2" in model_name:
        if "linearMSE" in model_name:
            return AestheticsPredictorV2Linear.from_pretrained(model_name)
        elif "reluMSE" in model_name:
            return AestheticsPredictorV2ReLU.from_pretrained(model_name)
        else:
            raise ValueError(f"Invalid v2 model name: {model_name}")
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def predict(model_name: str, image: PilImage) -> str:
    predictor = load_predictor(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    inputs = processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        outputs = predictor(**inputs)

    score = outputs.logits.item()
    logger.info(f"Aesthetics score: {score}")
    return f"{score:.3f}"


def main():
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Dropdown(
                choices=[
                    "shunk031/aesthetics-predictor-v1-vit-base-patch16",
                    "shunk031/aesthetics-predictor-v1-vit-base-patch32",
                    "shunk031/aesthetics-predictor-v1-vit-large-patch14",
                    "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
                    "shunk031/aesthetics-predictor-v2-ava-logos-l14-linearMSE",
                    "shunk031/aesthetics-predictor-v2-ava-logos-l14-reluMSE",
                ]
            ),
            gr.Image(type="pil"),
        ],
        outputs=gr.Label(label="Aesthetics score (1 to 10)"),
        allow_flagging="never",
    )
    demo.launch()


if __name__ == "__main__":
    main()
