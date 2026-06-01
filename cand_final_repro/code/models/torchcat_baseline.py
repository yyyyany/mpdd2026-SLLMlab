from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_temporal_encoder import HybridTemporalEncoder, TemporalAttentionPool


class ModalityEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.5,
        pre_dim: int | None = None,
        pool_type: str = "mean",
    ) -> None:
        super().__init__()
        if pool_type not in {"mean", "attn"}:
            raise ValueError(f"Unknown pool_type: {pool_type}")
        self.pool_type = pool_type
        if pre_dim is not None and input_dim > pre_dim:
            self.pre_proj = nn.Linear(input_dim, pre_dim)
            lstm_in = pre_dim
        else:
            self.pre_proj = None
            lstm_in = input_dim
        self.proj = nn.Linear(lstm_in, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn_pool = TemporalAttentionPool(hidden_dim) if pool_type == "attn" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_proj is not None:
            x = F.relu(self.pre_proj(x))
        x = F.relu(self.proj(x))
        x = self.dropout(x)
        x, _ = self.lstm(x)
        if self.attn_pool is not None:
            seq_mask = torch.any(torch.abs(x) > 0, dim=-1)
            return self.norm(self.attn_pool(x, seq_mask))
        return self.norm(x.mean(dim=1))


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int, n_modalities: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim * n_modalities, n_modalities)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(features, dim=1)
        weights = torch.softmax(self.gate(torch.cat(features, dim=-1)), dim=-1)
        return (weights.unsqueeze(-1) * stacked).sum(dim=1)


class CrossModalFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(features, dim=1)
        query = stacked.mean(dim=1, keepdim=True)
        out, _ = self.attn(query, stacked, stacked)
        return self.norm(out.squeeze(1))


class PersonalityEncoder(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMGenerator(nn.Module):
    def __init__(self, condition_dim: int, target_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, target_dim * 2),
        )

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(condition)
        gamma, beta = params.chunk(2, dim=-1)
        return torch.tanh(gamma), beta


class TorchcatBaseline(nn.Module):
    SUBTRACKS = {
        "A-V+P": ["audio", "video", "personality"],
        "A-V-G+P": ["audio", "video", "gait", "personality"],
        "G+P": ["gait", "personality"],
    }
    ENCODER_TYPES = {"bilstm_mean", "hybrid_attn"}
    HEAD_TYPES = {"softmax", "coral"}
    FUSION_TYPES = {"concat", "gated", "cross_attn"}
    POOL_TYPES = {"mean", "attn"}

    def __init__(
        self,
        subtrack: str = "A-V-G+P",
        num_classes: int = 3,
        is_regression: bool = False,
        use_regression_head: bool = False,
        audio_dim: int = 64,
        video_dim: int = 1000,
        gait_dim: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        encoder_type: str = "bilstm_mean",
        head_type: str = "softmax",
        use_film: bool = False,
        pool_type: str = "mean",
        modality_dropout: float = 0.0,
        fusion_type: str = "concat",
    ) -> None:
        super().__init__()
        if subtrack not in self.SUBTRACKS:
            raise ValueError(f"Unknown subtrack: {subtrack}")
        if encoder_type not in self.ENCODER_TYPES:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        if head_type not in self.HEAD_TYPES:
            raise ValueError(f"Unknown head_type: {head_type}")
        if fusion_type not in self.FUSION_TYPES:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        if pool_type not in self.POOL_TYPES:
            raise ValueError(f"Unknown pool_type: {pool_type}")

        self.subtrack = subtrack
        self.modalities = self.SUBTRACKS[subtrack]
        self.encoder_type = encoder_type
        self.is_regression = is_regression
        self.use_regression_head = use_regression_head
        self.head_type = head_type
        self.use_film = use_film
        self.pool_type = pool_type
        self.modality_dropout = modality_dropout
        self.fusion_type = fusion_type
        n_modalities = len(self.modalities)

        if "audio" in self.modalities:
            pre_audio = 128 if audio_dim > 128 else None
            self.audio_enc = (
                HybridTemporalEncoder(audio_dim, hidden_dim, dropout, pre_dim=pre_audio)
                if encoder_type == "hybrid_attn"
                else ModalityEncoder(audio_dim, hidden_dim, dropout, pre_dim=pre_audio, pool_type=pool_type)
            )
        if "video" in self.modalities:
            pre_video = 128 if video_dim > 128 else None
            self.video_enc = (
                HybridTemporalEncoder(video_dim, hidden_dim, dropout, pre_dim=pre_video)
                if encoder_type == "hybrid_attn"
                else ModalityEncoder(video_dim, hidden_dim, dropout, pre_dim=pre_video, pool_type=pool_type)
            )
        if "gait" in self.modalities:
            self.gait_enc = ModalityEncoder(gait_dim, hidden_dim, dropout, pool_type=pool_type)
        if "personality" in self.modalities:
            self.pers_enc = PersonalityEncoder(1024, hidden_dim, dropout)

        fused_dim = hidden_dim if fusion_type != "concat" else hidden_dim * n_modalities
        self.fused_dim = fused_dim
        if fusion_type == "gated":
            self.gated_fusion = GatedFusion(hidden_dim, n_modalities)
        elif fusion_type == "cross_attn":
            self.cross_fusion = CrossModalFusion(hidden_dim, dropout)
        if self.use_film and "personality" in self.modalities:
            self.film = FiLMGenerator(hidden_dim, fused_dim, dropout)
        if is_regression:
            cls_out_dim = 1
        elif head_type == "coral":
            cls_out_dim = 1
        else:
            cls_out_dim = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cls_out_dim, bias=head_type != "coral"),
        )
        if head_type == "coral":
            self.coral_bias = nn.Parameter(torch.zeros(num_classes - 1))
        if use_regression_head:
            self.regressor = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    @staticmethod
    def _masked_average_sequences(x: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pair_mask is None:
            return x.mean(dim=1)
        weights = pair_mask.unsqueeze(-1).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    @staticmethod
    def _masked_average_features(x: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pair_mask is None:
            return x.mean(dim=1)
        weights = pair_mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def _encode_pairwise_sequences(
        self,
        x: torch.Tensor,
        encoder: nn.Module,
        pair_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, pair_count, seq_len, feat_dim = x.shape
        encoded = encoder(x.reshape(batch_size * pair_count, seq_len, feat_dim))
        encoded = encoded.reshape(batch_size, pair_count, -1)
        return self._masked_average_features(encoded, pair_mask)

    def forward(
        self,
        audio: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        gait: torch.Tensor | None = None,
        personality: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = []

        if "audio" in self.modalities:
            if self.encoder_type == "hybrid_attn":
                features.append(self._encode_pairwise_sequences(audio, self.audio_enc, pair_mask))
            else:
                features.append(self.audio_enc(self._masked_average_sequences(audio, pair_mask)))

        if "video" in self.modalities:
            if self.encoder_type == "hybrid_attn":
                features.append(self._encode_pairwise_sequences(video, self.video_enc, pair_mask))
            else:
                features.append(self.video_enc(self._masked_average_sequences(video, pair_mask)))

        if "gait" in self.modalities:
            features.append(self.gait_enc(gait))

        personality_feature = None
        if "personality" in self.modalities:
            personality_feature = self.pers_enc(personality)
            features.append(personality_feature)

        if self.training and self.modality_dropout > 0:
            keep = torch.rand(len(features), features[0].size(0), 1, device=features[0].device) >= self.modality_dropout
            features = [feat * keep[i] for i, feat in enumerate(features)]

        if self.fusion_type == "concat":
            fused = torch.cat(features, dim=-1)
        elif self.fusion_type == "gated":
            fused = self.gated_fusion(features)
        else:
            fused = self.cross_fusion(features)
        if self.use_film and personality_feature is not None:
            gamma, beta = self.film(personality_feature)
            fused = fused * (1.0 + gamma) + beta
        logits = self.classifier(fused)
        if self.head_type == "coral":
            logits = logits + self.coral_bias
        if self.use_regression_head:
            return logits, self.regressor(fused).squeeze(-1)
        return logits
