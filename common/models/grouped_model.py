from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mtcn_backbone import MTCNBackbone, BackboneConfig


class GradientReversal(torch.autograd.Function):
    """GRL: forward=identity, backward=gradient * (-lambda)."""
    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


class ParticipantAggregator(nn.Module):

    def __init__(self, d_in: int, d_out: int, method: str = "mlp", dropout: float = 0.2):
        super().__init__()
        self.method = method
        self.d_in = d_in
        self.d_out = d_out

        if method == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_out, d_out),
            )
        elif method == "attention":
            self.query = nn.Linear(d_in, 1)
            self.proj = nn.Linear(d_in, d_out)
        elif method == "mean":
            self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def forward(self, session_reprs: torch.Tensor, session_valid: torch.Tensor) -> torch.Tensor:
        mask = session_valid.float().unsqueeze(-1)  
        masked_reprs = session_reprs * mask

        if self.method == "mean":
            n_valid = mask.sum(dim=1).clamp(min=1)  
            pooled = masked_reprs.sum(dim=1) / n_valid  
            return self.proj(pooled)

        elif self.method == "mlp":
            n_valid = mask.sum(dim=1).clamp(min=1)
            pooled = masked_reprs.sum(dim=1) / n_valid
            return self.mlp(pooled)

        elif self.method == "attention":
            scores = self.query(session_reprs).squeeze(-1)  
            scores = scores.masked_fill(~session_valid, float("-inf"))
            weights = F.softmax(scores, dim=-1) 
            weights = weights.masked_fill(~session_valid, 0.0)
            pooled = (weights.unsqueeze(-1) * session_reprs).sum(dim=1)  
            return self.proj(pooled)


class TransformerAggregator(nn.Module):
    """Transformer encoder with residual mean-pool connection for stability.

    Mean-pool provides a safe baseline; transformer learns session interactions
    as a residual delta. Even if transformer collapses, baseline works.
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.2, max_sessions: int = 4):
        super().__init__()
        self.d_model = d_model
        self.pos_embedding = nn.Parameter(torch.randn(1, max_sessions + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, session_reprs: torch.Tensor,
                session_valid: torch.Tensor) -> torch.Tensor:
        B, N, D = session_reprs.shape

        # Mean-pool baseline (stable, works even if transformer fails)
        mask = session_valid.float().unsqueeze(-1)
        n_valid = mask.sum(dim=1).clamp(min=1)
        mean_pooled = (session_reprs * mask).sum(dim=1) / n_valid

        # Transformer with CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, session_reprs], dim=1)
        x = x + self.pos_embedding[:, :N + 1]

        pad_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=x.device),
            ~session_valid,
        ], dim=1)

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        transformer_out = x[:, 0]

        return mean_pooled + transformer_out


class SessionTypeClassifier(nn.Module):
    def __init__(self, d_in: int, n_classes: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class GroupedModel(nn.Module):

    def __init__(
        self,
        backbone: MTCNBackbone,
        d_shared: int,
        aggregator_method: str = "mlp",
        dropout: float = 0.2,
        d_llm: int = 0,
        llm_offset: int = 0,
        n_schools: int = 0,
        d_school_emb: int = 16,
        adv_lambda: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_llm = d_llm
        self.llm_offset = llm_offset
        self.n_schools = n_schools
        self.adv_lambda = adv_lambda
        if aggregator_method == "transformer":
            self.aggregator = TransformerAggregator(
                d_model=d_shared, n_heads=4, n_layers=2, dropout=dropout,
            )
        else:
            self.aggregator = ParticipantAggregator(
                d_in=d_shared, d_out=d_shared,
                method=aggregator_method, dropout=dropout,
            )
        self.session_type_head = SessionTypeClassifier(d_in=d_shared + (64 if d_llm > 0 else 0))
        if d_llm > 0:
            self.llm_proj = nn.Sequential(
                nn.Linear(d_llm, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 64),
            )
        else:
            self.llm_proj = None
        # Adversarial school classifier — penalizes backbone for school-identifiable features
        if n_schools > 0:
            self.school_classifier = nn.Linear(d_shared, n_schools)
        else:
            self.school_classifier = None
        # School embedding (additive bias — not currently used, replaced by adversarial)
        self.school_emb = None
        self.school_proj = None

    def forward(
        self,
        flat_batch: dict,
        n_participants: int,
        session_valid: torch.Tensor,
        llm_features: torch.Tensor | None = None,
        school_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        session_reprs = self.backbone(flat_batch)

        B = n_participants
        session_grid = session_reprs.view(B, 4, -1)

        participant_repr = self.aggregator(session_grid, session_valid)

        # Fuse LLM features at participant level
        if self.llm_proj is not None and llm_features is not None:
            sliced = llm_features[:, self.llm_offset:self.llm_offset + self.d_llm]
            llm_emb = self.llm_proj(sliced.to(participant_repr.dtype))
            participant_repr = torch.cat([participant_repr, llm_emb], dim=-1)
            # Pad session reprs to same dim so single task_head works for both
            session_reprs = F.pad(session_reprs, (0, 64))

        # Adversarial school classifier: penalize backbone for school-identifiable features
        school_logits = None
        if self.school_classifier is not None and school_idx is not None:
            reversed_feats = grad_reverse(participant_repr, self.adv_lambda)
            school_logits = self.school_classifier(reversed_feats)

        session_type_logits = self.session_type_head(session_reprs)

        return {
            "session_reprs": session_reprs,
            "participant_repr": participant_repr,
            "session_type_logits": session_type_logits,
            "school_logits": school_logits,
        }


class CORALHead(nn.Module):

    def __init__(self, d_in: int, n_items: int = 21, n_thresholds: int = 3):
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds

        self.score_fc = nn.Linear(d_in, n_items)

        self.raw_thresholds = nn.Parameter(torch.zeros(n_items, n_thresholds))
        nn.init.constant_(self.raw_thresholds, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score_fc(x)

        spacings = F.softplus(self.raw_thresholds) 
        thresholds = torch.cumsum(spacings, dim=-1) 

        logits = scores.unsqueeze(-1) - thresholds.unsqueeze(0) 
        return logits

    @staticmethod
    def predict_int(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)

    @staticmethod
    def predict_int_monotonic(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        P0 = 1.0 - p1
        P1 = p1 - p2
        P2 = p2 - p3
        P3 = p3
        class_probs = torch.stack([P0, P1, P2, P3], dim=-1)
        return class_probs.argmax(dim=-1)

    @staticmethod
    def predict_expectation(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        E = p1 + p2 + p3
        return E.round().long().clamp(0, 3)
