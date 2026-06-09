

import torch
import torch.nn as nn

class MultiScaleBlock(nn.Module):
    """
    Multi-Scale Feature Extraction Block

    Input:
        [B, 3, H, W]

    Output:
        [B, out_channels*3, H, W]
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=16
    ):
        super().__init__()

        # ------------------------------
        # 3x3 Branch
        # ------------------------------

        self.branch3 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            CAF()
        )

        # ------------------------------
        # 5x5 Branch
        # ------------------------------

        self.branch5 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                padding=2,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            CAF()
        )

        # ------------------------------
        # 7x7 Branch
        # ------------------------------

        self.branch7 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                padding=3,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            CAF()
        )

        # ------------------------------
        # Fusion Layer
        # ------------------------------

        self.fusion_bn = nn.BatchNorm2d(
            out_channels * 3
        )

        self.fusion_act = CAF()

    def forward(
        self,
        x
    ):

        x3 = self.branch3(x)

        x5 = self.branch5(x)

        x7 = self.branch7(x)

        fused = torch.cat(
            [x3, x5, x7],
            dim=1
        )

        fused = self.fusion_bn(
            fused
        )

        fused = self.fusion_act(
            fused
        )

        return fused



class AttentionBlock(nn.Module):
    """
    Lightweight Channel Attention
    """

    def __init__(
        self,
        channels,
        reduction=16
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.fc1 = nn.Linear(
            channels,
            channels // reduction
        )

        self.act = CAF()

        self.fc2 = nn.Linear(
            channels // reduction,
            channels
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x
    ):

        batch_size, channels, _, _ = x.size()

        attention = self.pool(
            x
        ).view(
            batch_size,
            channels
        )

        attention = self.fc1(
            attention
        )

        attention = self.act(
            attention
        )

        attention = self.fc2(
            attention
        )

        attention = self.sigmoid(
            attention
        )

        attention = attention.view(
            batch_size,
            channels,
            1,
            1
        )

        return x * attention  


class ChannelAdapter(nn.Module):
    """
    96 -> 3 channel projection
    """

    def __init__(self):

        super().__init__()

        self.adapter = nn.Sequential(

            nn.Conv2d(
                48,
                3,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(3),

            CAF()
        )

    def forward(
        self,
        x
    ):

        return self.adapter(x)  

# ==================================================
# SECTION 7.9
# COMPLETE FRUITNET
# ==================================================

import torch
import torch.nn as nn
from torchvision import models

class FruitNet(nn.Module):

    def __init__(
        self,
        num_classes=15,
        dropout=0.4
    ):
        super().__init__()

        # ----------------------------------
        # MultiScale
        # ----------------------------------

        self.multiscale = MultiScaleBlock(
            in_channels=3,
            out_channels=16
        )

        # ----------------------------------
        # Attention
        # ----------------------------------

        self.attention = AttentionBlock(
            channels=96
        )

        # ----------------------------------
        # Adapter
        # ----------------------------------

        self.adapter = ChannelAdapter()

        # ----------------------------------
        # VGG16 Backbone
        # ----------------------------------

        backbone = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT
        )

        self.backbone = backbone.features

        # ----------------------------------
        # CAF
        # ----------------------------------

        self.caf = CAF()

        # ----------------------------------
        # Pooling
        # ----------------------------------

        self.gap = nn.AdaptiveAvgPool2d(
            (1,1)
        )

        # ----------------------------------
        # Embedding
        # ----------------------------------

        self.embedding = nn.Sequential(

            nn.Linear(
                512,
                512
            ),

            nn.BatchNorm1d(
                512
            ),

            CAF(),

            nn.Dropout(
                dropout
            )
        )

        # ----------------------------------
        # Classifier
        # ----------------------------------

        self.classifier = nn.Linear(
            512,
            num_classes
        )

    def forward(
        self,
        x
    ):

        x = self.multiscale(x)

        x = self.attention(x)

        x = self.adapter(x)

        x = self.backbone(x)

        x = self.caf(x)

        x = self.gap(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        features = self.embedding(x)

        logits = self.classifier(
            features
        )

        return features, logits        