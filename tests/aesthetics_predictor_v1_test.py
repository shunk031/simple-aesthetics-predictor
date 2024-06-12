from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve

import open_clip
import pytest
import torch
import torch.nn as nn
from PIL.Image import Image as PilImage
from transformers import CLIPProcessor

from aesthetics_predictor.v1 import convert_from_openai_clip


class TestCompareOriginalAndOurs(object):
    def get_aesthetic_model(
        self,
        clip_model: str,
    ) -> nn.Module:
        url_model = f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{clip_model}_linear.pth?raw=true"

        with NamedTemporaryFile(suffix=".pth") as tmp_file:
            urlretrieve(url=url_model, filename=tmp_file.name)

            if clip_model == "vit_l_14":
                m = nn.Linear(768, 1)
            elif clip_model == "vit_b_32" or "vit_b_16":
                m = nn.Linear(512, 1)
            else:
                raise ValueError(f"clip_model={clip_model} not supported")

            s = torch.load(tmp_file.name)
            m.load_state_dict(s)
            m.eval()
        return m

    def load_hf(self, hf_clip_model_name: str):
        """Load the model based on the Hugging Face (HF)."""
        processor = CLIPProcessor.from_pretrained(hf_clip_model_name)
        model = convert_from_openai_clip(hf_clip_model_name)
        assert model.training is False
        return processor, model

    def load_op(self, ap_clip_model_name: str, op_clip_model_name: str):
        """Load the model based on the OpenAI (OP) CLIP library (open_clip)"""
        amodel = self.get_aesthetic_model(clip_model=ap_clip_model_name)
        assert amodel.training is False

        model, _, preprocessor = open_clip.create_model_and_transforms(
            model_name=op_clip_model_name, pretrained="openai"
        )
        return preprocessor, model, amodel

    def preprocess_1(self, processor, sample_image):
        return processor(images=sample_image, return_tensors="pt")

    def preprocess_2(self, processor, sample_image):
        return processor(sample_image).unsqueeze(0)  # type: ignore

    def predict_1(self, inputs, model):
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs

    def predict_2(self, image, model, amodel):
        with torch.no_grad():
            image_features = model.encode_image(image)  # type: ignore
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = amodel(image_features)
        return {"logits": prediction, "hidden_states": image_features}

    @pytest.mark.parametrize(
        argnames="ap_clip_model_name, op_clip_model_name, hf_clip_model_name",
        argvalues=(
            ("vit_l_14", "ViT-L-14", "openai/clip-vit-large-patch14"),
            ("vit_b_16", "ViT-B-16", "openai/clip-vit-base-patch16"),
            ("vit_b_32", "ViT-B-32", "openai/clip-vit-base-patch32"),
        ),
    )
    def test_compare_original_and_ours(
        self,
        sample_image: PilImage,
        ap_clip_model_name: str,
        op_clip_model_name: str,
        hf_clip_model_name: str,
    ):
        processor1, model1 = self.load_hf(
            hf_clip_model_name=hf_clip_model_name,
        )
        processor2, model2, amodel2 = self.load_op(
            ap_clip_model_name=ap_clip_model_name,
            op_clip_model_name=op_clip_model_name,
        )
        assert torch.equal(
            sum(p.sum() for p in amodel2.parameters()),  # type: ignore
            sum(p.sum() for p in model1.predictor.parameters()),  # type: ignore
        )

        inputs1 = self.preprocess_1(processor1, sample_image)
        inputs2 = self.preprocess_2(processor2, sample_image)
        assert torch.equal(inputs1["pixel_values"], inputs2)  # type: ignore

        outputs1 = self.predict_1(inputs1, model1)
        outputs2 = self.predict_2(inputs2, model2, amodel2)

        assert torch.allclose(
            outputs1["hidden_states"].sum(),
            outputs2["hidden_states"].sum(),
        )
        assert torch.allclose(
            outputs1["logits"],
            outputs2["logits"],
        )
