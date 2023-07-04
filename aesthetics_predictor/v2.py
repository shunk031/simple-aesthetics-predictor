from typing import Dict, Final, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, logging
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers.models.clip.configuration_clip import CLIPVisionConfig

logging.set_verbosity_error()

URLS: Final[Dict[str, str]] = {
    "sac+logos+ava1-l14-linearMSE": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
    "ava+logos-l14-linearMSE": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/ava%2Blogos-l14-linearMSE.pth",
    "ava+logos-l14-reluMSE": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/ava%2Blogos-l14-reluMSE.pth",
}


class AestheticsPredictorV2(CLIPVisionModelWithProjection):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.layers = nn.Sequential(
            nn.Linear(config.projection_dim, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = super().forward(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = outputs[0]  # image_embeds
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)

        prediction = self.layer(image_embeds)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct()

        if not return_dict:
            return (loss, prediction, image_embeds)

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=prediction,
            hidden_states=image_embeds,
        )
