from transformers.models.clip.configuration_clip import CLIPVisionConfig


class AestheticsPredictorConfig(CLIPVisionConfig):
    model_type = "aesthetics_predictor"

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        projection_dim: int = 512,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 32,
        hidden_act: str = "quick_gelu",
        layer_norm_eps: float = 0.00001,
        attention_dropout: float = 0,
        initializer_range: float = 0.02,
        initializer_factor: float = 1,
        **kwargs,
    ):
        super().__init__(
            hidden_size,
            intermediate_size,
            projection_dim,
            num_hidden_layers,
            num_attention_heads,
            num_channels,
            image_size,
            patch_size,
            hidden_act,
            layer_norm_eps,
            attention_dropout,
            initializer_range,
            initializer_factor,
            **kwargs,
        )
