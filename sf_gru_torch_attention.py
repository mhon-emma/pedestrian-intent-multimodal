"""
sf_gru_torch_attention.py
==========================
PyTorch ports of the attention/fusion SF-GRU variants from the
Pedestrian-intent repo (originally tensorflow.keras). Each class subclasses
SFGRUTorch (sf_gru_torch.py) and only overrides build_model() -- the entire
data pipeline (pose loading, VGG16 features, box/speed sequences, train/test
loop) is reused unchanged and already verified correct.

Architectures (faithfully mirroring the original Keras source in
Pedestrian-intent/sf_gru_pose_attention.py, sf_gru_modality_fusion_attention.py,
sf_gru_cross_modal_attention.py, sf_gru_other_modal_attention.py):

  PoseAttentionSFGRU
    Original stacked-GRU fusion (local_box -> local_context -> pose -> box ->
    speed), with scaled dot-product self-attention applied to the
    concatenated tensor right after a chosen modality is concatenated in
    (attention_on='box' or 'pose').

  ModalityFusionSFGRU
    Each modality encoded independently (GRU, return_sequences=False) into a
    single vector, stacked into [batch, modalities, features], self-attention
    across modalities, mean-pooled, classified.

  CrossModalSFGRU
    Each modality encoded independently (GRU, return_sequences=True),
    concatenated along the feature axis, self-attention applied over the
    *temporal* dimension of the concatenated tensor, split back into
    per-modality slices, re-concatenated, and passed through a final GRU.

  OtherModalSFGRU
    Each modality encoded independently (GRU, return_sequences=True). For
    each modality, query = itself, key/value = concatenation of all *other*
    modalities' encodings ("attend to everything else"), residual + FFN,
    concatenated, passed through a final GRU.

Note on scaled_dot_product_attention: the original Keras code applies
softmax + dropout(0.1) unconditionally, including at eval time (Keras
Lambda layers don't automatically become no-ops in inference mode the way
nn.Dropout does). We use nn.functional.dropout with model.training so
dropout is correctly disabled at eval time -- an intentional, correct
deviation from the original's behavior, not a divergence to "fix" back.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sf_gru_torch import SFGRUTorch


def scaled_dot_product_attention(q, k, v, training=True, dropout_p=0.1):
    """q, k, v: [batch, seq, dim]. Returns attended [batch, seq, dim]."""
    dim = k.shape[-1]
    scores = torch.bmm(q, k.transpose(1, 2)) / (dim ** 0.5)
    weights = F.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout_p, training=training)
    return torch.bmm(weights, v)


class _AttentionBlock(nn.Module):
    """Q/K/V projections + scaled-dot-product attention + residual + FFN,
    matching _apply_attention_on_concat / _apply_other_modal_attention's
    per-instance layer structure in the original Keras code.

    query_dim and context_dim may differ (e.g. OtherModalSFGRU: query comes
    from one modality, key/value come from the concatenation of all *other*
    modalities, which is wider) -- matches Keras Dense(feature_dim)(context),
    where Dense only constrains the output width, not the input width. The
    residual add still requires query_dim == context_dim; that always holds
    for the two calling patterns used here (self-attention where both are
    query_dim, since output_dim == query_dim by construction)."""

    def __init__(self, query_dim, context_dim=None, dropout_p=0.1):
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim
        self.query = nn.Linear(query_dim, query_dim)
        self.key = nn.Linear(context_dim, query_dim)
        self.value = nn.Linear(context_dim, query_dim)
        self.out = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_p = dropout_p

    def forward(self, x_q, x_kv=None):
        if x_kv is None:
            x_kv = x_q
        q = self.query(x_q)
        k = self.key(x_kv)
        v = self.value(x_kv)
        attended = scaled_dot_product_attention(q, k, v, training=self.training,
                                                dropout_p=self.dropout_p)
        # residual + normalize by sqrt(2), matching (inp[0]+inp[1])/sqrt(2) in Keras
        fused = (x_q + attended) / (2.0 ** 0.5)
        fused = F.relu(self.out(fused))
        fused = self.dropout(fused)
        return fused


# ---------------------------------------------------------------------------
# 1. Pose/box concat attention (attention_on='box' or 'pose')
# ---------------------------------------------------------------------------

class PoseAttentionStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units, attention_on='box'):
        super().__init__()
        self.data_types = data_types
        self.attention_on = attention_on
        self.grus = nn.ModuleList()
        self.attn_blocks = nn.ModuleDict()
        for i, size in enumerate(data_sizes):
            in_dim = size[-1] if i == 0 else hidden_units + size[-1]
            self.grus.append(nn.GRU(input_size=in_dim, hidden_size=hidden_units, batch_first=True))
            if i > 0 and data_types[i] == attention_on:
                self.attn_blocks[str(i)] = _AttentionBlock(in_dim)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, inputs):
        x = None
        for i, gru in enumerate(self.grus):
            is_last = (i == len(self.grus) - 1)
            if i == 0:
                seq_in = inputs[0]
            else:
                seq_in = torch.cat([x, inputs[i]], dim=2)
                if str(i) in self.attn_blocks:
                    seq_in = self.attn_blocks[str(i)](seq_in)
            out, h = gru(seq_in)
            x = out if not is_last else h.squeeze(0)
        return torch.sigmoid(self.output(x))


class PoseAttentionSFGRU(SFGRUTorch):
    """attention_on: 'box' (default, matches original) or 'pose'."""

    def __init__(self, *args, attention_on='box', **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_on = attention_on

    def build_model(self, data_types, data_sizes):
        return PoseAttentionStackedGRU(data_types, data_sizes, self._num_hidden_units,
                                       attention_on=self.attention_on).to(self.device)


# ---------------------------------------------------------------------------
# 2. Modality fusion attention
# ---------------------------------------------------------------------------

class ModalityFusionStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.GRU(input_size=size[-1], hidden_size=hidden_units, batch_first=True)
            for size in data_sizes
        ])
        self.fusion_attn = _AttentionBlock(hidden_units)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, inputs):
        encodings = []
        for gru, inp in zip(self.encoders, inputs):
            _, h = gru(inp)
            encodings.append(h.squeeze(0))  # [batch, hidden]
        x = torch.stack(encodings, dim=1)  # [batch, modalities, hidden]
        x = self.fusion_attn(x)
        x = x.mean(dim=1)  # [batch, hidden]
        return torch.sigmoid(self.output(x))


class ModalityFusionSFGRU(SFGRUTorch):
    def build_model(self, data_types, data_sizes):
        return ModalityFusionStackedGRU(data_types, data_sizes, self._num_hidden_units).to(self.device)


# ---------------------------------------------------------------------------
# 3. Cross-modal (temporal) attention
# ---------------------------------------------------------------------------

class CrossModalStackedGRU(nn.Module):
    """Each modality encoded with return_sequences=True, concatenated along
    the feature axis, self-attention applied over the temporal dimension of
    the concatenated tensor, split back per-modality, re-concatenated, final
    GRU. Mirrors _apply_cross_modal_attention_timestep exactly (attention
    computed on the full concatenated feature vector, not per-modality)."""

    def __init__(self, data_types, data_sizes, hidden_units):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.GRU(input_size=size[-1], hidden_size=hidden_units, batch_first=True)
            for size in data_sizes
        ])
        num_modalities = len(data_sizes)
        concat_dim = hidden_units * num_modalities
        self.temporal_attn = _AttentionBlock(concat_dim)
        self.final_gru = nn.GRU(input_size=concat_dim, hidden_size=hidden_units, batch_first=True)
        self.output = nn.Linear(hidden_units, 1)
        self.hidden_units = hidden_units
        self.num_modalities = num_modalities

    def forward(self, inputs):
        sequences = []
        for gru, inp in zip(self.encoders, inputs):
            out, _ = gru(inp)  # [batch, seq, hidden]
            sequences.append(out)
        concatenated = torch.cat(sequences, dim=2)  # [batch, seq, hidden*num_modalities]
        attended_concat = self.temporal_attn(concatenated)
        # split back into per-modality slices then re-concatenate (matches
        # the original's split-then-reconcatenate, which is a no-op on the
        # values but preserved here for architectural fidelity)
        slices = [attended_concat[:, :, i * self.hidden_units:(i + 1) * self.hidden_units]
                  for i in range(self.num_modalities)]
        x = torch.cat(slices, dim=2)
        _, h = self.final_gru(x)
        x = h.squeeze(0)
        return torch.sigmoid(self.output(x))


class CrossModalSFGRU(SFGRUTorch):
    def build_model(self, data_types, data_sizes):
        return CrossModalStackedGRU(data_types, data_sizes, self._num_hidden_units).to(self.device)


# ---------------------------------------------------------------------------
# 4. Other-modal attention (each modality attends to all others)
# ---------------------------------------------------------------------------

class OtherModalStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units):
        super().__init__()
        self.data_types = data_types
        self.encoders = nn.ModuleList([
            nn.GRU(input_size=size[-1], hidden_size=hidden_units, batch_first=True)
            for size in data_sizes
        ])
        num_modalities = len(data_sizes)
        # query is one modality (hidden_units wide); key/value come from the
        # concatenation of all *other* modalities ((num_modalities-1)*hidden_units
        # wide) -- matches Keras Dense(feature_dim)(context), where Dense's
        # output width is feature_dim regardless of context's input width.
        context_dim = hidden_units * (num_modalities - 1)
        self.attn_blocks = nn.ModuleList([
            _AttentionBlock(hidden_units, context_dim=context_dim) for _ in data_sizes
        ])
        self.final_gru = nn.GRU(input_size=hidden_units * num_modalities, hidden_size=hidden_units,
                                batch_first=True)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, inputs):
        sequences = []
        for gru, inp in zip(self.encoders, inputs):
            out, _ = gru(inp)  # [batch, seq, hidden]
            sequences.append(out)

        attended_sequences = []
        for i, modality in enumerate(sequences):
            others = [m for j, m in enumerate(sequences) if j != i]
            context = others[0] if len(others) == 1 else torch.cat(others, dim=2)
            attended = self.attn_blocks[i](modality, context)
            attended_sequences.append(attended)

        x = torch.cat(attended_sequences, dim=2)
        _, h = self.final_gru(x)
        x = h.squeeze(0)
        return torch.sigmoid(self.output(x))


class OtherModalSFGRU(SFGRUTorch):
    def build_model(self, data_types, data_sizes):
        return OtherModalStackedGRU(data_types, data_sizes, self._num_hidden_units).to(self.device)
