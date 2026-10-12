"""NN-T: tiny causal transformer over recent branch events."""

from __future__ import annotations

import torch
from torch import nn

from ml.transformer_data import VOCAB_SIZE


class CausalBranchTransformer(nn.Module):
    name = "nn_t_causal"
    arch_id = 3

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 48,
        nhead: int = 4,
        nlayers: int = 2,
        seq_len: int = 64,
        dim_ff: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.seq_len = seq_len
        dim_ff = dim_ff or d_model * 2
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, L] int64 -> logits [B]."""
        b, L = tokens.shape
        if L > self.seq_len:
            tokens = tokens[:, -self.seq_len :]
            L = self.seq_len
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(b, L)
        x = self.embed(tokens) + self.pos_embed(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=tokens.device)
        x = self.encoder(x, mask=mask)
        return self.head(x[:, -1, :]).squeeze(-1)
