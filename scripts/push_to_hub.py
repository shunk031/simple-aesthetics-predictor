import torch
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1
from aesthetics_predictor.v1 import URLS as V1_URLS


def main():
    for model_name, url in V1_URLS.items():
        processor = CLIPProcessor.from_pretrained(model_name)
        model = AestheticsPredictorV1.from_pretrained(model_name)
        state_dict = torch.hub.load_state_dict_from_url(url)
        model.predictor.load_state_dict(state_dict)
        model.eval()

        model.push_to_hub("shunk031/aesthetics-predictor-v1")

        breakpoint()


if __name__ == "__main__":
    main()
