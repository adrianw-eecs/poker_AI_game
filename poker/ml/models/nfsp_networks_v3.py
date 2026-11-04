"""Attention-based network for observation v3.

``PokerNetV2`` feeds the seat block (9 x 15) into a flat MLP, so in a heads-up
pot 7 of 9 seat slots are padding the network must learn to ignore, and the same
weights are applied to "seat 0" and "seat 7" even though relative position is
already encoded in the features.

Seats are a *set*, not a sequence. This module pools them with self-attention
over occupied seats only, which:

  * makes the encoding independent of how many players are seated,
  * lets a padded slot be masked out rather than learned around,
  * costs one small attention layer instead of 135 wasted input dimensions.

Everything else mirrors ``PokerNetV2``: card embeddings, a table MLP, a layernorm
merge, residual trunk blocks, and the same head options (dueling value head,
policy head, or both via ``HybridNetV3``).
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from poker.ml.models.nfsp_networks import _ResBlock, _mlp
from poker.ml.observation_v3 import OBS_V3_DIM
from poker.ml.tendencies import MAX_SEATS

_CARD_SLOTS = 7
_EMPTY_CARD = 52
_SEAT_DIM = 15
_SEAT_BLOCK_START = 43  # v2 seat block begins here (OBS_SPEC_V2.seats[0])
_SEAT_BLOCK_END = 178   # 43 + 9 * 15


class SeatAttention(nn.Module):
    """Multi-head self-attention over the seat block plus a masked mean.

    Input is ``(B, MAX_SEATS, SEAT_DIM)`` with an occupancy mask ``(B, MAX_SEATS)``.
    Padded slots are masked to -inf before softmax and excluded from the mean, so
    the output does not depend on how many slots happen to be empty.
    """

    def __init__(self, seat_dim: int = _SEAT_DIM, hidden: int = 64, heads: int = 4) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden {hidden} must be divisible by heads {heads}")
        self.embed = nn.Sequential(
            nn.Linear(seat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        # Two outputs are concatenated: the attended summary and the pooled
        # mean, so the trunk sees both "the most relevant opponent" and "the
        # table average".
        self.out_dim = hidden * 2

    def forward(self, seats: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        h = self.embed(seats)                      # (B, S, H)
        key_padding = ~occupancy.bool()            # True = ignore
        # A fully padded batch row cannot happen (there is always a hero), but
        # guard anyway so attention never receives an all-masked row.
        all_masked = key_padding.all(dim=1)
        if bool(all_masked.any()):
            key_padding = key_padding.clone()
            key_padding[all_masked, 0] = False

        attended, _ = self.attn(h, h, h, key_padding_mask=key_padding, need_weights=False)
        attended = self.norm(attended)

        mask = occupancy.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (h * mask).sum(dim=1) / denom

        # Take the attended vector for the hero slot (index 0 is hero-relative).
        summary = attended[:, 0, :]
        return torch.cat([summary, pooled], dim=-1)


class PokerNetV3(nn.Module):
    """Card embeddings + attention-pooled seats + residual trunk."""

    def __init__(
        self,
        input_dim: int = OBS_V3_DIM,
        num_actions: int = 7,
        rank_dim: int = 32,
        suit_dim: int = 16,
        card_dim: int = 32,
        card_hidden: int = 128,
        table_hidden: int = 256,
        trunk_dim: int = 256,
        num_res_blocks: int = 3,
        expansion: float = 1.0,
        dropout_rate: float = 0.1,
        head_type: str = "value",
        attn_hidden: int = 64,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        if head_type not in ("value", "policy", "hybrid"):
            raise ValueError(f"head_type must be value/policy/hybrid, got {head_type}")
        if input_dim != OBS_V3_DIM:
            raise ValueError(
                f"PokerNetV3 expects the v3 width {OBS_V3_DIM}, got {input_dim}"
            )
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.head_type = head_type
        self.card_slots = _CARD_SLOTS
        self.empty_card = _EMPTY_CARD
        self.emb_dim = rank_dim + suit_dim + card_dim

        self.rank_emb = nn.Embedding(14, rank_dim)
        self.suit_emb = nn.Embedding(5, suit_dim)
        self.card_emb = nn.Embedding(53, card_dim)
        self.card_mlp = _mlp(self.emb_dim * 3, card_hidden)

        self.seat_attn = SeatAttention(_SEAT_DIM, attn_hidden, attn_heads)

        # Table features are everything that is neither cards nor seats:
        #   derived + hero (before the seat block)
        #   street summaries + action history + scalars (after it)
        #   tendencies (appended by v3)
        before_seats = _SEAT_BLOCK_START - _CARD_SLOTS
        after_seats = input_dim - _SEAT_BLOCK_END
        table_in = before_seats + after_seats
        self.table_mlp = _mlp(table_in, table_hidden)

        merged = card_hidden + table_hidden + self.seat_attn.out_dim
        self.merge_norm = nn.LayerNorm(merged)
        self.merge = nn.Linear(merged, trunk_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.trunk = nn.ModuleList(
            [_ResBlock(trunk_dim, expansion=expansion) for _ in range(num_res_blocks)]
        )

        if head_type in ("value", "hybrid"):
            self.value_head = nn.Linear(trunk_dim, 1)
            self.adv_head = nn.Linear(trunk_dim, num_actions)
        if head_type in ("policy", "hybrid"):
            self.policy_head = nn.Linear(trunk_dim, num_actions)
            nn.init.zeros_(self.policy_head.weight)
            nn.init.zeros_(self.policy_head.bias)

    # -- shared trunk -------------------------------------------------
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        empty = ids == self.empty_card
        rank = torch.where(empty, torch.full_like(ids, 13), torch.remainder(ids, 13))
        suit = torch.where(
            empty,
            torch.full_like(ids, 4),
            torch.div(ids, 13, rounding_mode="floor"),
        )
        return torch.cat(
            [self.rank_emb(rank), self.suit_emb(suit), self.card_emb(ids)], dim=-1
        )

    def _occupancy(self, x: torch.Tensor) -> torch.Tensor:
        """Infer which seat slots are real from the first seat feature.

        v2 writes ``feats[0] = 1.0`` for a real seat and leaves padding at 0,
        so the occupancy flag is recoverable without an extra input.
        """
        flags = x[:, _SEAT_BLOCK_START:_SEAT_BLOCK_END]
        flags = flags.view(x.shape[0], MAX_SEATS, _SEAT_DIM)[:, :, 0]
        return (flags > 0.5).to(dtype=x.dtype)

    def trunk_forward(self, x: torch.Tensor) -> torch.Tensor:
        ids = x[:, : self.card_slots].round().long().clamp(0, self.empty_card)
        emb = self._embed(ids)
        hole = torch.cat([emb[:, 0], emb[:, 1]], dim=-1)
        board = emb[:, 2 : self.card_slots]
        present = (ids[:, 2 : self.card_slots] != self.empty_card).unsqueeze(-1).to(
            dtype=board.dtype
        )
        board_pool = (board * present).sum(dim=1)
        card_h = self.card_mlp(torch.cat([hole, board_pool], dim=-1))

        seats = x[:, _SEAT_BLOCK_START:_SEAT_BLOCK_END].view(
            x.shape[0], MAX_SEATS, _SEAT_DIM
        )
        occup = self._occupancy(x)
        seat_h = self.seat_attn(seats, occup)

        # Table features: everything outside cards and seats, in original order.
        table_parts = [
            x[:, self.card_slots : _SEAT_BLOCK_START],
            x[:, _SEAT_BLOCK_END :],
        ]
        table_h = self.table_mlp(torch.cat(table_parts, dim=-1))

        hidden = self.dropout(
            F.relu(
                self.merge(
                    self.merge_norm(torch.cat([card_h, table_h, seat_h], dim=-1))
                )
            )
        )
        for block in self.trunk:
            hidden = block(hidden)
        return hidden

    def heads(
        self, hidden: torch.Tensor, legal_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """``(q_values, logits)``; either may be None depending on head_type."""
        q_values = None
        logits = None
        if self.head_type in ("value", "hybrid"):
            value = self.value_head(hidden)
            advantages = self.adv_head(hidden)
            if legal_mask is None:
                mean_adv = advantages.mean(dim=-1, keepdim=True)
            else:
                mask = legal_mask.bool()
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                advantages = advantages.masked_fill(~mask, 0.0)
                denom = mask.sum(dim=-1, keepdim=True).clamp(min=1).to(dtype=advantages.dtype)
                mean_adv = advantages.sum(dim=-1, keepdim=True) / denom
            q_values = value + advantages - mean_adv
        if self.head_type in ("policy", "hybrid"):
            logits = self.policy_head(hidden)
        return q_values, logits

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.trunk_forward(x)
        q_values, logits = self.heads(hidden, legal_mask)
        if q_values is not None:
            return q_values
        assert logits is not None
        return logits

    def forward_both(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk_forward(x)
        q_values, logits = self.heads(hidden, legal_mask)
        assert q_values is not None and logits is not None, (
            "forward_both requires head_type='hybrid'"
        )
        return q_values, logits


def build_q_network_v3(**kwargs: Any) -> PokerNetV3:
    kwargs.pop("arch", None)
    return PokerNetV3(head_type="value", **kwargs)


def build_policy_network_v3(**kwargs: Any) -> PokerNetV3:
    kwargs.pop("arch", None)
    return PokerNetV3(head_type="policy", **kwargs)


def build_hybrid_network_v3(**kwargs: Any) -> PokerNetV3:
    kwargs.pop("arch", None)
    return PokerNetV3(head_type="hybrid", **kwargs)
