import torch

from scaling_mlps.models.networks import get_model

from lucent.optvis import render
from lucent.optvis.objectives import Objective
from lucent.modelzoo import inceptionv1
from lucent.modelzoo.util import get_model_layers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = inceptionv1(pretrained=True)
model = get_model(
    architecture="B_6-Wi_512", resolution=64, num_classes=11230, checkpoint="in21k"
)

model.to(device).eval()

# layer_names = get_model_layers(model)
# print(layer_names)
# input()

objective = Objective(..., "bla", "desc")
# render.render_vis(model, "mixed4a:476")
render.render_vis(model, "blocks_3_block_0:2", verbose=True)
