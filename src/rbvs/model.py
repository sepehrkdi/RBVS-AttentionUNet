"""Attention U-Net (and a plain U-Net baseline) in PyTorch.

Faithful port of CV_Project.ipynb cell 29 (attention) and cell 25 (plain U-Net).
The Keras original used K.int_shape() ratios and a Lambda channel-tile that broke
under Keras 3 and blocked dynamic input shapes. Here the spatial factors are
provably constant, so the network is fully convolutional and accepts any (H, W)
divisible by 16 with no dynamic-shape gymnastics:

  In every attention gate, inter_channels == skip channels == the level width
  n_filters, and the skip is exactly 2x the spatial size of the gating signal.
  Therefore:
    theta_x : Conv 2x2 stride2  -> skip downsampled to gating resolution
    phi_g   : Conv 1x1          -> gating projected (same resolution)
    up_phi  : ConvT 3x3 stride1 -> keeps gating resolution (the ratio is 1)
    psi/sig : 1-channel attention coefficient at gating resolution
    up_psi  : nearest x2 upsample back to skip resolution == the alpha map
  and the channel-broadcast multiply (skip * alpha) replaces the Lambda tile.

Split head: vessel_logits (Conv 1x1, no activation) -> sigmoid = vessel_prob.
Numerically identical to the original single sigmoid conv, but exposes a logit
tensor for Grad-CAM. Attention gate 1 is the deepest (on skip f4, gating at
input/16); gate 4 is the shallowest (on f1, gating at input/2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 same-conv + ReLU layers (notebook double_conv_block)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class DownBlock(nn.Module):
    """double_conv -> skip f, then MaxPool + Dropout -> p (notebook downsample_block).

    Dropout is elementwise p=0.2 (nn.Dropout), matching Keras layers.Dropout — NOT
    channel-wise Dropout2d.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        f = self.double_conv(x)
        p = self.drop(self.pool(f))
        return f, p


class GatingSignal(nn.Module):
    """1x1 conv + BN + ReLU projecting a decoder feature to a gating signal
    (notebook gating_signal)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class AttentionGate(nn.Module):
    """Additive attention gate filtering a skip connection (notebook attention_block).

    n_filters == skip channels == gating channels for every call in this network.
    The (B, 1, H, W) attention map is stored in self.alpha on each forward for
    explainability.
    """

    def __init__(self, n_filters):
        super().__init__()
        self.theta = nn.Conv2d(n_filters, n_filters, kernel_size=2, stride=2)
        self.phi = nn.Conv2d(n_filters, n_filters, kernel_size=1)
        self.up_phi = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=3,
                                         stride=1, padding=1)
        self.psi = nn.Conv2d(n_filters, 1, kernel_size=1)
        self.out_conv = nn.Conv2d(n_filters, n_filters, kernel_size=1)
        self.bn = nn.BatchNorm2d(n_filters)
        self.alpha = None  # last (B,1,H,W) attention map, set on forward

    def forward(self, skip, gating):
        theta_x = self.theta(skip)                     # (B,C,H/2,W/2)
        phi_g = self.phi(gating)                        # (B,C,H/2,W/2)
        up_phi = self.up_phi(phi_g)                     # (B,C,H/2,W/2), ratio 1
        act = F.relu(up_phi + theta_x)
        psi = self.psi(act)                             # (B,1,H/2,W/2)
        sig = torch.sigmoid(psi)
        alpha = F.interpolate(sig, scale_factor=2, mode="nearest")  # (B,1,H,W)
        self.alpha = alpha
        y = skip * alpha                                # broadcast over channels
        return self.bn(self.out_conv(y))


class UpBlock(nn.Module):
    """Attention-gated decoder block (notebook upsample_block).

    in_ch = channels of the deeper decoder feature; n_filters = level width and
    also the skip's channel count.
    """

    def __init__(self, in_ch, n_filters):
        super().__init__()
        self.gating = GatingSignal(in_ch, n_filters)
        self.up = nn.ConvTranspose2d(in_ch, n_filters, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        self.attn = AttentionGate(n_filters)
        self.double_conv = DoubleConv(2 * n_filters, n_filters)

    def forward(self, x, skip):
        g = self.gating(x)          # gating at skip/2 resolution
        x = self.up(x)              # transpose-upsample to skip resolution
        attn_feats = self.attn(skip, g)
        x = torch.cat([x, attn_feats], dim=1)
        return self.double_conv(x)

    @property
    def alpha(self):
        return self.attn.alpha


class AttentionUNet(nn.Module):
    """Attention U-Net; input 1 channel, output vessel probability in [0,1]."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = UpBlock(1024, 512)   # attn_gate_1 on f4 (deepest)
        self.up2 = UpBlock(512, 256)    # attn_gate_2 on f3
        self.up3 = UpBlock(256, 128)    # attn_gate_3 on f2
        self.up4 = UpBlock(128, 64)     # attn_gate_4 on f1 (shallowest)
        self.vessel_logits = nn.Conv2d(64, 1, kernel_size=1)  # split head: logits

    def forward(self, x, return_features=False):
        f1, p1 = self.down1(x)
        f2, p2 = self.down2(p1)
        f3, p3 = self.down3(p2)
        f4, p4 = self.down4(p3)
        b = self.bottleneck(p4)
        u1 = self.up1(b, f4)
        u2 = self.up2(u1, f3)
        u3 = self.up3(u2, f2)
        u4 = self.up4(u3, f1)                # decoder_last_conv feature
        logits = self.vessel_logits(u4)
        prob = torch.sigmoid(logits)
        if return_features:
            return {
                "logits": logits,
                "prob": prob,
                "feat": u4,                  # decoder_last_conv, for Grad-CAM
                "attn": [self.up1.alpha, self.up2.alpha,
                         self.up3.alpha, self.up4.alpha],
            }
        return prob


class _PlainUp(nn.Module):
    """Plain (no-attention) decoder block for the baseline U-Net (cell 24)."""

    def __init__(self, in_ch, n_filters):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, n_filters, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        self.double_conv = DoubleConv(2 * n_filters, n_filters)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)


class BaselineUNet(nn.Module):
    """Plain U-Net (notebook cell 25), no attention. Provided untrained as a
    baseline; not part of the reported results."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = _PlainUp(1024, 512)
        self.up2 = _PlainUp(512, 256)
        self.up3 = _PlainUp(256, 128)
        self.up4 = _PlainUp(128, 64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        f1, p1 = self.down1(x)
        f2, p2 = self.down2(p1)
        f3, p3 = self.down3(p2)
        f4, p4 = self.down4(p3)
        b = self.bottleneck(p4)
        x = self.up1(b, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        return torch.sigmoid(self.head(x))


def build_attention_unet(input_shape=(128, 128, 1)):
    """Attention U-Net. input_shape is kept for API parity with the notebook;
    only the channel count (last dim) is used — the net is fully convolutional."""
    return AttentionUNet(in_channels=input_shape[-1])


def build_baseline_unet(input_shape=(128, 128, 1)):
    """Untrained plain U-Net baseline."""
    return BaselineUNet(in_channels=input_shape[-1])


def build_attention_extractor(model):
    """Return fn(x) -> [alpha1, alpha2, alpha3, alpha4] attention maps."""
    def extract(x):
        return model(x, return_features=True)["attn"]
    return extract


def build_gradcam_model(model):
    """Return fn(x) -> (vessel_logits, decoder_last_conv feature)."""
    def forward(x):
        out = model(x, return_features=True)
        return out["logits"], out["feat"]
    return forward
