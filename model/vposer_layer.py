from torch import nn

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

from paths import VPOSER_DIR


class VPoserLayer(nn.Module):
    def __init__(self, expr_dir=VPOSER_DIR, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.vp, _ = load_model(
            expr_dir,
            model_code=VPoser,
            remove_words_in_model_weights="vp_model.",
            disable_grad=True,
        )

    def forward(self, x):
        body_pose = self.vp.decode(x)["pose_body"] # type: ignore
        return body_pose