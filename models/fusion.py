"""
fusion.py
Three multi-modal fusion strategies combining RGB and Pose branches:
1. LateFusionModel:
   - Runs RGB and Pose branches independently
   - Averages their softmax outputs at inference time
   - Compatible with raw frames (5D) and precomputed embeddings (3D)

2. FeatureConcatFusionModel:
   - Extracts feature vectors from both branches, concatenates, and classifies.
   - Compatible with raw frames (5D) and precomputed embeddings (3D)

3. CrossAttentionFusionModel:
   - Bidirectional cross-attention between RGB and Pose sequences.
   - Compatible with raw frames (5D) and precomputed embeddings (3D)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from models.rgb_branch import RGBBaselineModel
from models.pose_branch import PoseBranchModel


# ── Late Fusion ────────────────────────────────────────────────────────────────
class LateFusionModel(nn.Module):
    """
    Averages softmax probabilities from RGB and Pose branches.
    Supports raw frames (B, T, C, H, W) and precomputed embeddings (B, T, 2048).
    """

    def __init__(self, rgb_model: RGBBaselineModel, pose_model: PoseBranchModel):
        super().__init__()
        self.rgb_model  = rgb_model
        self.pose_model = pose_model
        self.softmax    = nn.Softmax(dim=1)

    def forward(self, frames: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames : (batch, T, C, H, W) or (batch, T, 2048)
            pose   : (batch, T, 258)
        Returns:
            logits : (batch, num_classes)
        """
        if len(frames.shape) == 5:
            # Raw frames
            rgb_logits = self.rgb_model(frames)
        else:
            # Precomputed embeddings: bypass CNN feature extractor
            B, T, D = frames.shape
            x = self.rgb_model.input_proj(frames.view(B * T, D)).view(B, T, -1)
            x = x + self.rgb_model.pos_embedding[:, :T, :]
            x = self.rgb_model.transformer(x)
            x = x.mean(dim=1)
            rgb_logits = self.rgb_model.classifier(x)

        rgb_probs  = self.softmax(rgb_logits)
        pose_probs = self.softmax(self.pose_model(pose))
        return (rgb_probs + pose_probs) / 2


# ── Feature Concatenation Fusion ───────────────────────────────────────────────
class FeatureConcatFusionModel(nn.Module):
    """
    Extracts penultimate features from both branches, concatenates,
    and learns a joint classifier MLP. Supports both raw frames and embeddings.
    """

    def __init__(
        self,
        num_classes: int = 8,
        num_frames: int = 16,
        rgb_d_model: int = 512,
        pose_d_model: int = 256,
        dropout: float = 0.1,
        rgb_weights_path: str = None,
        pose_weights_path: str = None,
        freeze_encoders: bool = True,
    ):
        super().__init__()

        # ── RGB encoder (ResNet50 + Transformer, no classifier head) ──────────
        _rgb = RGBBaselineModel(
            num_classes=num_classes,
            num_frames=num_frames,
            d_model=rgb_d_model,
        )
        if rgb_weights_path and os.path.exists(rgb_weights_path):
            _rgb.load_state_dict(torch.load(rgb_weights_path, map_location="cpu"))
            print(f"[INFO] Loaded pretrained RGB weights for Concat Fusion: {rgb_weights_path}")
        self.rgb_cnn         = _rgb.cnn
        self.rgb_input_proj  = _rgb.input_proj
        self.rgb_pos_emb     = _rgb.pos_embedding
        self.rgb_transformer = _rgb.transformer

        # ── Pose encoder (Transformer, no classifier head) ────────────────────
        _pose = PoseBranchModel(
            num_classes=num_classes,
            num_frames=num_frames,
            d_model=pose_d_model,
        )
        if pose_weights_path and os.path.exists(pose_weights_path):
            _pose.load_state_dict(torch.load(pose_weights_path, map_location="cpu"))
            print(f"[INFO] Loaded pretrained Pose weights for Concat Fusion: {pose_weights_path}")
        self.pose_input_proj  = _pose.input_proj
        self.pose_pos_emb     = _pose.pos_embedding
        self.pose_transformer = _pose.transformer

        # ── Fusion MLP ─────────────────────────────────────────────────────────
        fused_dim = rgb_d_model + pose_d_model   # 512 + 256 = 768
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        if freeze_encoders:
            for p in self.rgb_input_proj.parameters(): p.requires_grad = False
            self.rgb_pos_emb.requires_grad = False
            for p in self.rgb_transformer.parameters(): p.requires_grad = False
            for p in self.rgb_cnn.parameters(): p.requires_grad = False

            for p in self.pose_input_proj.parameters(): p.requires_grad = False
            self.pose_pos_emb.requires_grad = False
            for p in self.pose_transformer.parameters(): p.requires_grad = False
            print("[INFO] Freezing feature encoders in Concat Fusion Model.")

    def _encode_rgb(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim feature vector from RGB frames/embeddings."""
        if len(frames.shape) == 5:
            B, T, C, H, W = frames.shape
            x = frames.view(B * T, C, H, W)
            x = self.rgb_cnn(x).view(B * T, -1)        # (B*T, 2048)
        else:
            B, T, D = frames.shape
            x = frames.view(B * T, D)

        x = self.rgb_input_proj(x).view(B, T, -1)  # (B, T, 512)
        x = x + self.rgb_pos_emb[:, :T, :]
        x = self.rgb_transformer(x)                 # (B, T, 512)
        return x.mean(dim=1)                        # (B, 512)

    def _encode_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Extract 256-dim feature vector from pose sequence."""
        x = self.pose_input_proj(pose)              # (B, T, 256)
        x = x + self.pose_pos_emb[:, :x.size(1), :]
        x = self.pose_transformer(x)                # (B, T, 256)
        return x.mean(dim=1)                        # (B, 256)

    def forward(self, frames: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        rgb_feat  = self._encode_rgb(frames)        # (B, 512)
        pose_feat = self._encode_pose(pose)         # (B, 256)
        fused     = torch.cat([rgb_feat, pose_feat], dim=1)  # (B, 768)
        return self.fusion_mlp(fused)               # (B, 8)


# ── Cross-Attention Fusion ─────────────────────────────────────────────────────
class CrossAttentionFusionModel(nn.Module):
    """
    Learns cross-attended representations where RGB and Pose branches attend
    to each other frame-by-frame. Supports raw frames and precomputed embeddings.
    """

    def __init__(
        self,
        num_classes: int = 8,
        num_frames: int = 16,
        rgb_d_model: int = 512,
        pose_d_model: int = 256,
        cross_attn_d_model: int = 512,
        cross_attn_nhead: int = 8,
        dropout: float = 0.1,
        rgb_weights_path: str = None,
        pose_weights_path: str = None,
        freeze_encoders: bool = True,
    ):
        super().__init__()

        # RGB Branch
        _rgb = RGBBaselineModel(
            num_classes=num_classes,
            num_frames=num_frames,
            d_model=rgb_d_model,
        )
        if rgb_weights_path and os.path.exists(rgb_weights_path):
            _rgb.load_state_dict(torch.load(rgb_weights_path, map_location="cpu"))
            print(f"[INFO] Loaded pretrained RGB weights for Cross-Attention Fusion: {rgb_weights_path}")
        self.rgb_cnn         = _rgb.cnn
        self.rgb_input_proj  = _rgb.input_proj
        self.rgb_pos_emb     = _rgb.pos_embedding
        self.rgb_transformer = _rgb.transformer

        # Pose Branch
        _pose = PoseBranchModel(
            num_classes=num_classes,
            num_frames=num_frames,
            d_model=pose_d_model,
        )
        if pose_weights_path and os.path.exists(pose_weights_path):
            _pose.load_state_dict(torch.load(pose_weights_path, map_location="cpu"))
            print(f"[INFO] Loaded pretrained Pose weights for Cross-Attention Fusion: {pose_weights_path}")
        self.pose_input_proj  = _pose.input_proj
        self.pose_pos_emb     = _pose.pos_embedding
        self.pose_transformer = _pose.transformer

        # Projections to align dimensions to cross_attn_d_model
        self.pose_to_cross = nn.Linear(pose_d_model, cross_attn_d_model)
        self.rgb_to_cross  = nn.Linear(rgb_d_model, cross_attn_d_model)

        # Cross-Attention modules
        self.cross_attn_rgb = nn.MultiheadAttention(
            embed_dim=cross_attn_d_model, num_heads=cross_attn_nhead,
            dropout=dropout, batch_first=True
        )
        self.cross_attn_pose = nn.MultiheadAttention(
            embed_dim=cross_attn_d_model, num_heads=cross_attn_nhead,
            dropout=dropout, batch_first=True
        )

        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(cross_attn_d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        if freeze_encoders:
            for p in self.rgb_input_proj.parameters(): p.requires_grad = False
            self.rgb_pos_emb.requires_grad = False
            for p in self.rgb_transformer.parameters(): p.requires_grad = False
            for p in self.rgb_cnn.parameters(): p.requires_grad = False

            for p in self.pose_input_proj.parameters(): p.requires_grad = False
            self.pose_pos_emb.requires_grad = False
            for p in self.pose_transformer.parameters(): p.requires_grad = False
            print("[INFO] Freezing feature encoders in Cross-Attention Fusion Model.")

    def _encode_rgb(self, frames: torch.Tensor) -> torch.Tensor:
        if len(frames.shape) == 5:
            B, T, C, H, W = frames.shape
            x = frames.view(B * T, C, H, W)
            x = self.rgb_cnn(x).view(B * T, -1)
        else:
            B, T, D = frames.shape
            x = frames.view(B * T, D)

        x = self.rgb_input_proj(x).view(B, T, -1)
        x = x + self.rgb_pos_emb[:, :T, :]
        return self.rgb_transformer(x)  # (B, T, 512)

    def _encode_pose(self, pose: torch.Tensor) -> torch.Tensor:
        x = self.pose_input_proj(pose)
        x = x + self.pose_pos_emb[:, :x.size(1), :]
        return self.pose_transformer(x)  # (B, T, 256)

    def forward(self, frames: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        rgb_seq  = self._encode_rgb(frames)             # (B, T, 512)
        pose_seq = self._encode_pose(pose)             # (B, T, 256)

        rgb_seq_aligned  = self.rgb_to_cross(rgb_seq)   # (B, T, 512)
        pose_seq_aligned = self.pose_to_cross(pose_seq) # (B, T, 512)

        # Bidirectional cross-attention
        rgb_attended, _ = self.cross_attn_rgb(
            query=rgb_seq_aligned, key=pose_seq_aligned, value=pose_seq_aligned
        )  # (B, T, 512)

        pose_attended, _ = self.cross_attn_pose(
            query=pose_seq_aligned, key=rgb_seq_aligned, value=rgb_seq_aligned
        )  # (B, T, 512)

        # Temporal average pooling
        feat_rgb  = rgb_attended.mean(dim=1)           # (B, 512)
        feat_pose = pose_attended.mean(dim=1)          # (B, 512)

        # Concatenate and classify
        fused = torch.cat([feat_rgb, feat_pose], dim=1) # (B, 1024)
        return self.classifier(fused)                  # (B, 8)


# ── Sanity test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T = 2, 16
    dummy_frames = torch.randn(B, T, 3, 224, 224)
    dummy_pose   = torch.randn(B, T, 258)
    dummy_embs   = torch.randn(B, T, 2048)

    # 1. Test Late Fusion
    print("[TEST] LateFusionModel (raw frames) ...")
    rgb_model  = RGBBaselineModel(num_classes=8, num_frames=T)
    pose_model = PoseBranchModel(input_dim=258, num_classes=8, num_frames=T)
    late_model = LateFusionModel(rgb_model, pose_model)
    late_model.eval()
    with torch.no_grad():
        out = late_model(dummy_frames, dummy_pose)
    print(f"  Output shape : {out.shape}")
    assert out.shape == torch.Size([B, 8])
    print("  PASSED\n")

    print("[TEST] LateFusionModel (precomputed embeddings) ...")
    with torch.no_grad():
        out = late_model(dummy_embs, dummy_pose)
    print(f"  Output shape : {out.shape}")
    assert out.shape == torch.Size([B, 8])
    print("  PASSED\n")

    # 2. Test Feature Concat Fusion
    print("[TEST] FeatureConcatFusionModel (raw frames) ...")
    concat_model = FeatureConcatFusionModel(num_classes=8, num_frames=T)
    concat_model.eval()
    with torch.no_grad():
        out = concat_model(dummy_frames, dummy_pose)
    print(f"  Output shape : {out.shape}")
    assert out.shape == torch.Size([B, 8])
    print("  PASSED\n")

    print("[TEST] FeatureConcatFusionModel (precomputed embeddings) ...")
    with torch.no_grad():
        out = concat_model(dummy_embs, dummy_pose)
    print(f"  Output shape : {out.shape}")
    assert out.shape == torch.Size([B, 8])
    print("  PASSED\n")

    # 3. Test Cross-Attention Fusion
    print("[TEST] CrossAttentionFusionModel (raw frames) ...")
    cross_model = CrossAttentionFusionModel(num_classes=8, num_frames=T)
    cross_model.eval()
    with torch.no_grad():
        out = cross_model(dummy_frames, dummy_pose)
    print(f"  Output shape : {out.shape}")
    assert out.shape == torch.Size([B, 8])
    print("  PASSED\n")

    print("[TEST] CrossAttentionFusionModel (precomputed embeddings) ...")
    with torch.no_grad():
        out = cross_model(dummy_embs, dummy_pose)
    print(f"  Output shape : {out.shape}")
    assert out.shape == torch.Size([B, 8])
    print("  PASSED")
