import torch

from scaling_mlps.models.networks import get_model

from lucent.optvis import render
from lucent.optvis.objectives import Objective
from lucent.modelzoo import inceptionv1
from lucent.modelzoo.util import get_model_layers

MODE = "g"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if MODE == "l":
    model = inceptionv1(pretrained=True)
else:
    model = get_model(
        architecture="B_6-Wi_512", resolution=64, num_classes=11230, checkpoint="in21k"
    )

model.to(device).eval()

# layer_names = get_model_layers(model)
# print(layer_names)
# input()

objective = Objective(..., "bla", "desc")

if MODE == "l":
    render.render_vis(model, "conv2d0_pre_relu_conv:0", verbose=True)
else:
    render.render_vis_flat(
        model, "blocks_0_block_1:1", verbose=True, thresholds=(1024,)
    )
