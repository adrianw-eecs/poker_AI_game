"""Shared network architectures for NFSP agents."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from poker.ml.observation_v2 import OBS_V2_DIM


def _build_output_proj(
    hidden_dim: int,
    output_dims: tuple[int, ...],
    dropout_rate: float,
) -> tuple[nn.Sequential, int]:
    """Build the MLP above the residual trunk.

    One stage, ``(128,)``, is Linear -> LayerNorm -> ReLU.
    Two stages, ``(256, 128)``, insert ReLU + Dropout between them so the
    parameter names match production checkpoints.
    """
    if not output_dims:
        raise ValueError("output_dims must contain at least one width")

    layers: list[nn.Module] = []
    in_dim = hidden_dim
    for stage, out_dim in enumerate(output_dims):
        if stage > 0:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.LayerNorm(out_dim))
        in_dim = out_dim
    layers.append(nn.ReLU())
    return nn.Sequential(*layers), in_dim


def _linear_layers(state_dict: dict, prefix: str) -> list[tuple[int, int]]:
    """``(out_features, in_features)`` for 2D weights under ``prefix``, in order."""
    found: list[tuple[int, int, int]] = []
    for key, value in state_dict.items():
        if not key.startswith(prefix) or not key.endswith(".weight") or value.ndim != 2:
            continue
        index = int(key[len(prefix):].split(".", 1)[0])
        found.append((index, int(value.shape[0]), int(value.shape[1])))
    found.sort()
    return [(out_dim, in_dim) for _index, out_dim, in_dim in found]


def _v1_kwargs_from_state_dict(state_dict: dict) -> dict:
    """Recover PokerNetwork constructor kwargs from a checkpoint state_dict."""
    input_weight = state_dict["input_proj.0.weight"]
    hidden_dim = int(input_weight.shape[0])
    input_dim = int(input_weight.shape[1])

    num_res_blocks = 0
    while f"res_blocks.{num_res_blocks}.fc1.weight" in state_dict:
        num_res_blocks += 1
    if num_res_blocks == 0:
        raise ValueError("Checkpoint has no residual blocks")

    fc1 = state_dict["res_blocks.0.fc1.weight"]
    expansion = float(fc1.shape[0]) / float(fc1.shape[1])

    output_dims: list[int] = []
    for key, value in state_dict.items():
        if key.startswith("output_proj.") and key.endswith(".weight") and value.ndim == 2:
            output_dims.append(int(value.shape[0]))
    if not output_dims:
        raise ValueError("Checkpoint output projection has no linear layers")

    head = state_dict["head.weight"]
    return {
        "arch": "v1",
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_actions": int(head.shape[0]),
        "num_res_blocks": num_res_blocks,
        "output_dims": tuple(output_dims),
        "expansion": expansion,
    }


def _v2_kwargs_from_state_dict(state_dict: dict) -> dict:
    """Recover PokerNetV2 constructor kwargs from a checkpoint state_dict."""
    card_layers = _linear_layers(state_dict, "card_mlp.")
    table_layers = _linear_layers(state_dict, "table_mlp.")
    if not card_layers or not table_layers:
        raise ValueError("PokerNetV2 checkpoint is missing the card or table MLP")
    num_res_blocks = 0
    while f"trunk.{num_res_blocks}.fc1.weight" in state_dict:
        num_res_blocks += 1
    if num_res_blocks == 0:
        raise ValueError("PokerNetV2 checkpoint has no residual blocks")
    fc1 = state_dict["trunk.0.fc1.weight"]
    trunk_dim = int(fc1.shape[1])
    if "adv_head.weight" in state_dict:
        num_actions = int(state_dict["adv_head.weight"].shape[0])
    elif "policy_head.weight" in state_dict:
        num_actions = int(state_dict["policy_head.weight"].shape[0])
    else:
        raise ValueError("PokerNetV2 checkpoint has no action head")
    card_slots = 7
    return {
        "arch": "v2",
        "input_dim": int(table_layers[0][1]) + card_slots,
        "num_actions": num_actions,
        "rank_dim": int(state_dict["rank_emb.weight"].shape[1]),
        "suit_dim": int(state_dict["suit_emb.weight"].shape[1]),
        "card_dim": int(state_dict["card_emb.weight"].shape[1]),
        "card_hidden": int(card_layers[-1][0]),
        "table_hidden": int(table_layers[-1][0]),
        "trunk_dim": trunk_dim,
        "num_res_blocks": num_res_blocks,
        "expansion": float(fc1.shape[0]) / float(trunk_dim),
    }


def network_kwargs_from_state_dict(state_dict: dict) -> dict:
    """Recover constructor kwargs from a v1 or v2 checkpoint state_dict.

    The returned dict includes ``arch`` (``v1`` or ``v2``). Drop that key
    before splatting into ``PokerNetwork`` or ``PokerNetV2``.
    """
    if "rank_emb.weight" in state_dict:
        return _v2_kwargs_from_state_dict(state_dict)
    if "input_proj.0.weight" not in state_dict:
        raise ValueError("Checkpoint is neither PokerNetwork nor PokerNetV2")
    return _v1_kwargs_from_state_dict(state_dict)


class _ResBlock(nn.Module):
    """2-layer pre-activation residual block with expanded bottleneck."""

    def __init__(self, dim: int, expansion: float = 1.5) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(F.relu(self.norm1(x)))
        x = self.fc2(F.relu(self.norm2(x)))
        return residual + x


class PokerNetwork(nn.Module):
    """Deep residual network for poker Q-values and policy logits.

    Architecture (with defaults hidden_dim=256, num_res_blocks=2):
      Input(98) -> FC(256) -> LN -> Dropout -> ReLU
      -> ResBlock(256) x2
      -> FC(128) -> LN -> ReLU
      -> Output(7)

    Total ~90K parameters — appropriate for a 98-dim input space.

    head_type controls final-layer initialization:
      "value"  (Q-network) : PyTorch default init — outputs raw Q-values.
      "policy" (policy net): near-zero init — starts near-uniform after softmax.
    """

    def __init__(
        self,
        input_dim: int = 98,
        hidden_dim: int = 256,
        num_actions: int = 7,
        num_res_blocks: int = 2,
        dropout_rate: float = 0.1,
        head_type: str = "value",
        output_dims: tuple[int, ...] = (128,),
        expansion: float = 1.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [_ResBlock(hidden_dim, expansion=expansion) for _ in range(num_res_blocks)]
        )

        self.output_proj, head_in = _build_output_proj(hidden_dim, output_dims, dropout_rate)
        self.head = nn.Linear(head_in, num_actions)

        # Policy networks start near-uniform (all logits ~0 -> uniform softmax).
        # Q-networks use PyTorch default init (no constraint on output scale).
        if head_type == "policy":
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_proj(x)
        return self.head(x)


def build_q_network(
    input_dim: int = 98,
    num_actions: int = 7,
    hidden_dim: int = 256,
    num_res_blocks: int = 2,
    dropout_rate: float = 0.1,
    output_dims: tuple[int, ...] = (128,),
    expansion: float = 1.5,
) -> PokerNetwork:
    """Create a Q-network with value head (default PyTorch init)."""
    return PokerNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
        num_res_blocks=num_res_blocks,
        dropout_rate=dropout_rate,
        head_type="value",
        output_dims=output_dims,
        expansion=expansion,
    )


def build_policy_network(
    input_dim: int = 98,
    num_actions: int = 7,
    hidden_dim: int = 256,
    num_res_blocks: int = 2,
    dropout_rate: float = 0.1,
    output_dims: tuple[int, ...] = (128,),
    expansion: float = 1.5,
) -> PokerNetwork:
    """Create a policy network with near-zero head init (starts near-uniform)."""
    return PokerNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
        num_res_blocks=num_res_blocks,
        dropout_rate=dropout_rate,
        head_type="policy",
        output_dims=output_dims,
        expansion=expansion,
    )


def make_target_network(source: nn.Module) -> nn.Module:
    """Create a frozen copy of source network to serve as the target network."""
    target = copy.deepcopy(source)
    for param in target.parameters():
        param.requires_grad = False
    target.eval()
    return target


def sync_target_network(source: nn.Module, target: nn.Module) -> None:
    """Copy weights from source to target (hard update)."""
    target.load_state_dict(source.state_dict())
    target.eval()


_CARD_SLOTS = 7
_EMPTY_CARD = 52


def _mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
    )


class PokerNetV2(nn.Module):
    """Card-embedding network with a dueling Q head or a policy head.

    Default widths follow the v2 diagram (card MLP 128, table MLP 256,
    three pre-norm residual blocks of width 256) and land near 700k
    parameters.

    The flat observation is split into 7 card ids and the remaining table
    features. Cards use rank, suit, and card embeddings. Hole cards are
    concatenated and the board is a masked sum. A table MLP, a layer-norm
    merge, and three pre-norm residual blocks feed the head.

    ``head_type="value"`` is the dueling Q head ``V + A - mean(A)``. When a
    legal mask is passed, that mean is over legal actions only.
    ``head_type="policy"`` is a zero-initialized logit head.
    """

    def __init__(
        self,
        input_dim: int = OBS_V2_DIM,
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
    ) -> None:
        super().__init__()
        if head_type not in ("value", "policy"):
            raise ValueError(f"head_type must be 'value' or 'policy', got {head_type}")
        if input_dim <= _CARD_SLOTS:
            raise ValueError(f"input_dim must exceed {_CARD_SLOTS} card slots, got {input_dim}")
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
        self.table_mlp = _mlp(input_dim - _CARD_SLOTS, table_hidden)
        merged = card_hidden + table_hidden
        self.merge_norm = nn.LayerNorm(merged)
        self.merge = nn.Linear(merged, trunk_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.trunk = nn.ModuleList(
            [_ResBlock(trunk_dim, expansion=expansion) for _ in range(num_res_blocks)]
        )
        if head_type == "value":
            self.value_head = nn.Linear(trunk_dim, 1)
            self.adv_head = nn.Linear(trunk_dim, num_actions)
        else:
            self.policy_head = nn.Linear(trunk_dim, num_actions)
            nn.init.zeros_(self.policy_head.weight)
            nn.init.zeros_(self.policy_head.bias)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        empty = ids == self.empty_card
        rank = torch.where(empty, torch.full_like(ids, 13), torch.remainder(ids, 13))
        suit = torch.where(
            empty,
            torch.full_like(ids, 4),
            torch.div(ids, 13, rounding_mode="floor"),
        )
        return torch.cat(
            [self.rank_emb(rank), self.suit_emb(suit), self.card_emb(ids)],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ids = x[:, : self.card_slots].round().long().clamp(0, self.empty_card)
        emb = self._embed(ids)
        hole = torch.cat([emb[:, 0], emb[:, 1]], dim=-1)
        board = emb[:, 2 : self.card_slots]
        present = (ids[:, 2 : self.card_slots] != self.empty_card).unsqueeze(-1).to(dtype=board.dtype)
        board_pool = (board * present).sum(dim=1)
        card_h = self.card_mlp(torch.cat([hole, board_pool], dim=-1))
        table_h = self.table_mlp(x[:, self.card_slots :])
        hidden = self.dropout(F.relu(self.merge(self.merge_norm(torch.cat([card_h, table_h], dim=-1)))))
        for block in self.trunk:
            hidden = block(hidden)
        if self.head_type == "policy":
            return self.policy_head(hidden)

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
        return value + advantages - mean_adv


def build_q_network_v2(**kwargs) -> PokerNetV2:
    """Q-network with a dueling head."""
    kwargs.pop("arch", None)
    return PokerNetV2(head_type="value", **kwargs)


def build_policy_network_v2(**kwargs) -> PokerNetV2:
    """Policy network with a zero-initialized logit head."""
    kwargs.pop("arch", None)
    return PokerNetV2(head_type="policy", **kwargs)
