# ==========================================
# model.py
# Reverse-engineered from best_model.pth
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MetaTokenProj(nn.Module):
    """
    Metadata projection:
    3 -> 128 (BN) -> 256 (BN) -> 768 (LayerNorm)
    Matches keys: meta_token_proj.proj.{0,1,3,4,6}, meta_token_proj.norm
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, 128),           # proj.0
            nn.BatchNorm1d(128),         # proj.1
            nn.ReLU(),                   # proj.2  (no weights, skipped in state_dict)
            nn.Linear(128, 256),         # proj.3
            nn.BatchNorm1d(256),         # proj.4
            nn.ReLU(),                   # proj.5  (no weights)
            nn.Linear(256, 768),         # proj.6
        )
        self.norm = nn.LayerNorm(768)    # meta_token_proj.norm

    def forward(self, meta):
        return self.norm(self.proj(meta))


class EfficientNet_ViT_Metadata(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # ─── 1. ViT-Base/16 (pretrained) ───────────────────────────────
        # vit.cls_token          [1, 1, 768]
        # vit.pos_embed          [1, 197, 768]  (196 patches + 1 cls)
        # vit.patch_embed.proj   [768, 3, 16, 16]
        # vit.blocks.0-11        12 standard ViT blocks
        # vit.norm               LayerNorm(768)
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,   # weights come from best_model.pth
            num_classes=0,      # remove head
        )

        # ─── 2. Metadata projection ─────────────────────────────────────
        # meta_token_proj.proj.{0,1,3,4,6} + meta_token_proj.norm
        self.meta_token_proj = MetaTokenProj()

        # ─── 3. Extra cross-attention / fusion blocks ────────────────────
        # extra_blocks.layers.{0,1}  — 2-layer TransformerEncoder, d=768
        extra_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=3072,
            dropout=0.1,
            batch_first=True,
        )
        self.extra_blocks = nn.TransformerEncoder(extra_layer, num_layers=2)
        self.extra_norm = nn.LayerNorm(768)   # extra_norm

        # ─── 4. Classifier ───────────────────────────────────────────────
        # Input: concat(img_feat[768], meta_feat[768]) = 1536
        # classifier.0  Linear(1536, 1024)
        # classifier.1  BatchNorm1d(1024)
        # classifier.2  ReLU
        # classifier.3  Dropout  (no weights)
        # classifier.4  Linear(1024, 512)
        # classifier.5  BatchNorm1d(512)
        # classifier.6  ReLU
        # classifier.7  Dropout  (no weights)
        # classifier.8  Linear(512, 7)
        self.classifier = nn.Sequential(
            nn.Linear(1536, 1024),       # classifier.0
            nn.BatchNorm1d(1024),        # classifier.1
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),        # classifier.4
            nn.BatchNorm1d(512),         # classifier.5
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes), # classifier.8
        )

        # ─── 5. Auxiliary classifier (on ViT cls token only) ────────────
        # aux_classifier  Linear(768, 7)
        self.aux_classifier = nn.Linear(768, num_classes)

    def forward(self, x, meta):

        # ── ViT forward (get cls token) ──────────────────────────────────
        # patch_embed + pos_embed + cls_token handled inside timm ViT
        tokens = self.vit.patch_embed(x)                        # [B, 196, 768]
        cls = self.vit.cls_token.expand(tokens.size(0), -1, -1) # [B, 1, 768]
        tokens = torch.cat([cls, tokens], dim=1)                 # [B, 197, 768]
        tokens = tokens + self.vit.pos_embed
        tokens = self.vit.pos_drop(tokens) if hasattr(self.vit, 'pos_drop') else tokens

        for block in self.vit.blocks:
            tokens = block(tokens)

        tokens = self.vit.norm(tokens)
        img_feat = tokens[:, 0, :]   # cls token → [B, 768]

        # ── Metadata token ───────────────────────────────────────────────
        meta_feat = self.meta_token_proj(meta)   # [B, 768]

        # ── Fuse: stack as sequence, run extra transformer blocks ────────
        fused = torch.stack([img_feat, meta_feat], dim=1)  # [B, 2, 768]
        fused = self.extra_blocks(fused)                    # [B, 2, 768]
        fused = self.extra_norm(fused)

        img_out  = fused[:, 0, :]   # [B, 768]
        meta_out = fused[:, 1, :]   # [B, 768]

        # ── Main classifier ──────────────────────────────────────────────
        combined = torch.cat([img_out, meta_out], dim=1)  # [B, 1536]
        out = self.classifier(combined)                    # [B, 7]

        # ── Aux classifier (used during training only) ───────────────────
        aux_out = self.aux_classifier(img_out)             # [B, 7]

        if self.training:
            return out, aux_out
        return out