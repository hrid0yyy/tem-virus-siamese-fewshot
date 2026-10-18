"""Backbones and heads.

* EmbeddingNet   - shared-weight (Siamese) encoder: ImageNet backbone -> GAP -> MLP -> L2-norm.
* PaperClassifier - exact replica of the authors' transfer model (Keras train.py):
                    backbone(include_top=False) -> Flatten -> Dense1024 -> Dropout0.3
                    -> Dense1024 -> Dropout0.3 -> Dense(n_classes).
                    For DenseNet201/14 classes this gives 144,987,022 trainable parameters and
                    for VGG16 49,334,094, matching Table 2 of the paper.
* GAPClassifier  - simpler GAP + linear / cosine classifier baseline.

Inputs are (B, 1, H, W) z-scored patches; the grayscale channel is repeated to 3 channels
(as the authors did with np.concatenate((X, X, X), -1)).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONES = ("densenet201", "resnet50", "vgg16", "resnet18")


class _DenseNetFeatures(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        # Keras DenseNet201(include_top=False) ends with BN + ReLU; torchvision's .features ends
        # with norm5, so the ReLU is applied here.
        return F.relu(self.features(x))


def build_backbone(name: str, pretrained: bool = True, memory_efficient: bool = False):
    """Return (module mapping (B,3,H,W) -> feature map (B,C,h,w), C)."""
    import torchvision.models as tvm

    name = name.lower()
    if name == "densenet201":
        w = tvm.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.densenet201(weights=w, memory_efficient=memory_efficient)
        return _DenseNetFeatures(m.features), m.classifier.in_features  # 1920
    if name in ("resnet50", "resnet18"):
        if name == "resnet50":
            w = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            m = tvm.resnet50(weights=w)
        else:
            w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            m = tvm.resnet18(weights=w)
        body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        return body, m.fc.in_features  # 2048 / 512
    if name == "vgg16":
        w = tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.vgg16(weights=w)
        return m.features, 512  # includes the final max-pool, like Keras include_top=False
    raise ValueError(f"Unknown backbone '{name}'. Choose from {BACKBONES}")


def to_three_channels(x):
    if x.dim() != 4:
        raise ValueError(f"expected (B,C,H,W), got {tuple(x.shape)}")
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] == 3:
        return x
    raise ValueError(f"expected 1 or 3 channels, got {x.shape[1]}")


class CosineClassifier(nn.Module):
    """logits = s * cos(z, w_c)  (Baseline++ style, Chen et al. 2019)."""

    def __init__(self, in_dim, n_classes, scale=16.0, learn_scale=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, in_dim) * 0.01)
        s = torch.tensor(float(scale))
        if learn_scale:
            self.scale = nn.Parameter(s)
        else:
            self.register_buffer("scale", s)

    def forward(self, z):
        return self.scale * F.normalize(z, dim=1) @ F.normalize(self.weight, dim=1).t()


class EmbeddingNet(nn.Module):
    """Siamese encoder. forward(x) -> (z, f): z = L2-normalised embedding, f = GAP backbone features."""

    def __init__(self, backbone="densenet201", pretrained=True, emb_dim=128, hidden_dim=512,
                 memory_efficient=False, n_aux_classes=0, aux_scale=16.0, pair_head=False):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(backbone, pretrained, memory_efficient)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.emb_dim = emb_dim
        self.aux = CosineClassifier(emb_dim, n_aux_classes, aux_scale) if n_aux_classes > 0 else None
        # Koch et al. (2015) style verification head: p(same) = sigmoid(w . |z1 - z2| + b)
        self.pair_head = nn.Linear(emb_dim, 1) if pair_head else None

    def features(self, x):
        fmap = self.backbone(to_three_channels(x))
        return torch.flatten(self.pool(fmap), 1)

    def embed_from_features(self, f):
        return F.normalize(self.head(f), dim=1)

    def forward(self, x):
        f = self.features(x)
        return self.embed_from_features(f), f

    def head_parameters(self):
        mods = [self.head] + [m for m in (self.aux, self.pair_head) if m is not None]
        return [p for m in mods for p in m.parameters()]


class PaperClassifier(nn.Module):
    """Replica of the authors' Keras transfer model (Flatten + 1024 + 1024 head)."""

    def __init__(self, backbone="densenet201", n_classes=14, pretrained=True, input_size=256,
                 dropout=0.3, memory_efficient=False):
        super().__init__()
        self.backbone, c = build_backbone(backbone, pretrained, memory_efficient)
        with torch.no_grad():
            was_training = self.backbone.training
            self.backbone.eval()
            h = self.backbone(torch.zeros(1, 3, input_size, input_size))
            self.backbone.train(was_training)
        self.feat_dim = c
        self.fmap_shape = tuple(h.shape[1:])
        flat = int(h[0].numel())
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 1024), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(1024, n_classes),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.classifier(self.backbone(to_three_channels(x)))

    def features(self, x):
        """GAP backbone features (used for nearest-prototype evaluation of CE baselines)."""
        return torch.flatten(self.pool(self.backbone(to_three_channels(x))), 1)

    def head_parameters(self):
        return list(self.classifier.parameters())


class GAPClassifier(nn.Module):
    """Backbone -> GAP -> linear or cosine classifier."""

    def __init__(self, backbone="densenet201", n_classes=14, pretrained=True, head="linear",
                 memory_efficient=False, cos_scale=16.0, dropout=0.0):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(backbone, pretrained, memory_efficient)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if head == "linear":
            self.classifier = nn.Linear(self.feat_dim, n_classes)
        elif head == "cosine":
            self.classifier = CosineClassifier(self.feat_dim, n_classes, cos_scale)
        else:
            raise ValueError(f"unknown head {head}")

    def features(self, x):
        return torch.flatten(self.pool(self.backbone(to_three_channels(x))), 1)

    def forward(self, x):
        return self.classifier(self.drop(self.features(x)))

    def head_parameters(self):
        return list(self.classifier.parameters())


def count_parameters(module, trainable_only=True):
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def build_model_from_config(cfg: dict, pretrained=None):
    """Rebuild a model from a saved run config (pretrained weights not needed when loading a checkpoint)."""
    pre = cfg.get("pretrained", True) if pretrained is None else pretrained
    kind = cfg["model_type"]
    n_classes = len(cfg["train_classes"])
    if kind == "metric":
        return EmbeddingNet(cfg["backbone"], pre, cfg["emb_dim"], cfg["hidden_dim"],
                            cfg.get("memory_efficient", False),
                            n_aux_classes=n_classes if cfg.get("aux_ce_weight", 0) > 0 else 0,
                            aux_scale=cfg.get("aux_scale", 16.0),
                            pair_head=(cfg["loss"] == "pair"))
    if kind == "classifier":
        if cfg["ce_head"] == "paper":
            return PaperClassifier(cfg["backbone"], n_classes, pre, dropout=cfg.get("dropout", 0.3),
                                   memory_efficient=cfg.get("memory_efficient", False))
        return GAPClassifier(cfg["backbone"], n_classes, pre, head=cfg["ce_head"],
                             memory_efficient=cfg.get("memory_efficient", False),
                             cos_scale=cfg.get("aux_scale", 16.0), dropout=cfg.get("dropout", 0.0))
    raise ValueError(f"unknown model_type {kind}")
