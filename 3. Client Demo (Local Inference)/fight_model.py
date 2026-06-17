import torch
import torch.nn as nn
from transformers import TimesformerModel, TimesformerConfig


class FightTimeSformer(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        config = TimesformerConfig.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )
        config.num_frames = 8
        config.image_size = 224

        self.backbone = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            config=config
        )

        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        """
        x shape MUST be: (B, T, C, H, W)
        """
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        return self.classifier(cls_token)


def load_fight_model(model_path, device):
    model = FightTimeSformer()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
