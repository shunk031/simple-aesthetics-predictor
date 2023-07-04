# 🤗 Simple Aesthetics Predictor

[![CI](https://github.com/shunk031/simple-aesthetics-predictor/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/simple-aesthetics-predictor/actions/workflows/ci.yaml)

[CLIP](https://arxiv.org/abs/2103.00020)-based aesthetics predictor inspired by the interface of [🤗 huggingface transformers](https://huggingface.co/docs/transformers/index). This library provides a simple wrapper that can load the predictor using the `from_pretrained` method.

## Install

```shell
pip install git+https://github.com/shunk031/simple-aesthetics-predictor.git
```

## How to Use

```python
import requests
import torch
from PIL import Image
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1

#
# Load the aesthetics predictor
#
model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"

model = AestheticsPredictorV1.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

#
# Download sample image
#
url = "https://github.com/shunk031/simple-aesthetics-predictor/blob/master/assets/a-photo-of-an-astronaut-riding-a-horse.png?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

#
# Preprocess the image
#
inputs = processor(images=image, return_tensor="pt")

#
# Inference for the image
#
with torch.no_grad():
    outputs = model(**inputs)
prediction = outputs.logits

print(f"Aesthetics score: {prediction}")
```

## The Predictors found in 🤗 Huggingface Hub

- [🤗 aesthetics-predictor-v1](https://huggingface.co/models?search=aesthetics-predictor-v1)
- [🤗 aesthetics-predictor-v2](https://huggingface.co/models?search=aesthetics-predictor-v2)

## Acknowledgements

- LAION-AI/aesthetic-predictor: A linear estimator on top of clip to predict the aesthetic quality of pictures https://github.com/LAION-AI/aesthetic-predictor 
- christophschuhmann/improved-aesthetic-predictor: CLIP+MLP Aesthetic Score Predictor https://github.com/christophschuhmann/improved-aesthetic-predictor 
