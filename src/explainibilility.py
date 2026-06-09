# ==================================================
# SECTION 15.7
# HEATMAPS
# ==================================================
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

import torch.nn as nn

def replace_inplace_relu(module):

    for name, child in module.named_children():

        if isinstance(child, nn.ReLU):

            setattr(
                module,
                name,
                nn.ReLU(inplace=False)
            )

        else:

            replace_inplace_relu(child)

replace_inplace_relu(model)

print(
    "All inplace ReLU layers replaced."
)
# ==================================================
# SECTION 15.4
# GRADCAM
# ==================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.register_hooks()

    def register_hooks(self):

        def forward_hook(
            module,
            inp,
            out
        ):
            self.activations = out

        def backward_hook(
            module,
            grad_in,
            grad_out
        ):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(
            forward_hook
        )

        self.target_layer.register_full_backward_hook(
            backward_hook
        )

    def generate(
        self,
        image_tensor,
        class_idx=None
    ):

        self.model.eval()

        features, logits = self.model(
            image_tensor
        )

        if class_idx is None:

            class_idx = logits.argmax(
                dim=1
            )

        score = logits[
            :,
            class_idx
        ]

        self.model.zero_grad()

        score.backward(
            retain_graph=True
        )

        gradients = self.gradients

        activations = self.activations

        weights = gradients.mean(
            dim=(2,3),
            keepdim=True
        )

        cam = (
            weights *
            activations
        ).sum(
            dim=1
        )

        cam = torch.relu(
            cam
        )

        cam = cam.squeeze()

        cam = cam.detach().cpu().numpy()

        cam = (
            cam -
            cam.min()
        )

        cam = cam / (
            cam.max() +
            1e-8
        )

        return cam
    
model.eval()

for idx, row in sample_df.iterrows():

    image_path = row["filepath"]

    image = Image.open(
        image_path
    ).convert("RGB")

    image_resized = image.resize(
        (224,224)
    )

    img_np = np.array(
        image_resized
    )

    tensor = test_transform(
        image_resized
    ).unsqueeze(0)

    tensor = tensor.to(
        DEVICE
    )

    cam = gradcam.generate(
        tensor
    )

    heatmap = cv2.resize(
        cam,
        (224,224)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(

        heatmap,

        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(

        img_np,

        0.6,

        heatmap,

        0.4,

        0
    )

    plt.figure(
        figsize=(8,4)
    )

    plt.subplot(1,2,1)

    plt.imshow(
        img_np
    )

    plt.axis("off")

    plt.title(
        "Original"
    )

    plt.subplot(1,2,2)

    plt.imshow(
        overlay
    )

    plt.axis("off")

    plt.title(
        "GradCAM"
    )

    plt.show()