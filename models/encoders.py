# models/encoders.py
import math
import torch
import torch.nn as nn

def _build_sinusoidal_pe(L, d_model, device, dtype):
    pe = torch.zeros(L, d_model, device=device, dtype=dtype)
    position = torch.arange(0, L, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe  # [L, d_model]

class MHAEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, depth=3, ffn=1024, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn,
            dropout=dropout,
            batch_first=True,
            activation="gelu",  # harmless improvement
            norm_first=True     # often a bit stabler
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, D]
        return self.enc(x, src_key_padding_mask=key_padding_mask)  # [B, T, D]

# -------- Pointwise (time step = token) ----------
class EncPointwise(nn.Module):
    def __init__(self, in_dim, d_model=256, n_heads=8, depth=3, ffn=1024, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.backbone = MHAEncoder(d_model, n_heads, depth, ffn, dropout)
        self.register_buffer("time_pe", torch.empty(0), persistent=False)

    def _ensure_time_pe(self, L, d_model, device, dtype):
        if self.time_pe.numel() == 0 or self.time_pe.size(0) < L or self.time_pe.size(1) != d_model or self.time_pe.device != device:
            self.time_pe = _build_sinusoidal_pe(L, d_model, device, dtype)

    def forward(self, x):
        # x: [B, L, V]  (tokens = L; feature dim = V)
        B, L, V = x.shape
        h = self.proj(x)  # [B, L, d_model]
        self._ensure_time_pe(L, h.size(-1), h.device, h.dtype)
        h = h + self.time_pe[:L].unsqueeze(0)
        h = self.backbone(h)  # [B, L, d_model]
        return h  # keep full sequence; pool in the head if needed


# -------- PatchTST (per-variate patches = tokens) ----------
class EncPatchTST(nn.Module):
    def __init__(self, patch_len=16, d_model=256, n_heads=8, depth=3, ffn=1024, dropout=0.1, agg='mean'):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)  # project each patch vector
        self.backbone = MHAEncoder(d_model, n_heads, depth, ffn, dropout)
        self.agg = agg  # 'mean' or 'last'
        self.register_buffer("patch_pe", torch.empty(0), persistent=False)

    def _ensure_patch_pe(self, P, d_model, device, dtype):
        if self.patch_pe.numel() == 0 or self.patch_pe.size(0) < P or self.patch_pe.size(1) != d_model or self.patch_pe.device != device:
            self.patch_pe = _build_sinusoidal_pe(P, d_model, device, dtype)

    def forward(self, x):
        # x: [B, V, P, P_len]
        B, V, P, P_len = x.shape
        h = self.proj(x.contiguous().view(B * V, P, P_len))  # [B*V, P, d_model]
        self._ensure_patch_pe(P, h.size(-1), h.device, h.dtype)
        h = h + self.patch_pe[:P].unsqueeze(0)
        h = self.backbone(h)  # [B*V, P, d_model]
        if self.agg == 'mean':
            h = h.mean(dim=1)     # [B*V, d_model]
        else:
            h = h[:, -1]          # [B*V, d_model]
        h = h.view(B, V, -1)      # [B, V, d_model]
        return h  # per-variate embeddings


# -------- iTransformer (variates = tokens; time embedded) ----------
class EnciTransformer(nn.Module):
    def __init__(self, L, d_model=256, n_heads=8, depth=3, ffn=1024, dropout=0.1):
        super().__init__()
        # map each variate's length-L vector to d_model via 1D conv or linear
        self.time_embed = nn.Linear(L, d_model)
        self.backbone = MHAEncoder(d_model, n_heads, depth, ffn, dropout)
        self.register_buffer("var_pe", torch.empty(0), persistent=False)

    def _ensure_var_pe(self, V, d_model, device, dtype):
        if self.var_pe.numel() == 0 or self.var_pe.size(0) < V or self.var_pe.size(1) != d_model or self.var_pe.device != device:
            self.var_pe = _build_sinusoidal_pe(V, d_model, device, dtype)

    def forward(self, x):
        # x: [B, V, L]
        B, V, L = x.shape
        h = self.time_embed(x)  # [B, V, d_model]  (time compressed into channels)
        self._ensure_var_pe(V, h.size(-1), h.device, h.dtype)
        h = h + self.var_pe[:V].unsqueeze(0)
        h = self.backbone(h)  # [B, V, d_model]  (attention across variates)
        return h  # per-variate embeddings


# -------- Cross-Lite (time encoder + small cross-variate MHSA) ----------
class CrossLite(nn.Module):
    """
    Stage 1: PatchTST over time per variate → [B, V, d_model]
    Stage 2: Tiny cross-variate MHSA over those V tokens
    """
    def __init__(self, patch_len=8, d_model=256, n_heads_time=8, depth_time=2,
                 ffn=1024, dropout=0.1, n_heads_cross=2, depth_cross=1):
        super().__init__()
        self.time_enc = EncPatchTST(patch_len=patch_len, d_model=d_model,
                                    n_heads=n_heads_time, depth=depth_time,
                                    ffn=ffn, dropout=dropout, agg='mean')
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads_cross,
            dim_feedforward=ffn, dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True
        )
        self.cross = nn.TransformerEncoder(layer, num_layers=depth_cross)
        self.register_buffer("var_pe", torch.empty(0), persistent=False)

    def _ensure_var_pe(self, V, d_model, device, dtype):
        if self.var_pe.numel() == 0 or self.var_pe.size(0) < V or self.var_pe.size(1) != d_model or self.var_pe.device != device:
            self.var_pe = _build_sinusoidal_pe(V, d_model, device, dtype)

    def forward(self, x):
        # x: [B, V, P, P_len]  (PatchDataset output)
        h = self.time_enc(x)  # [B, V, d_model]
        B, V, D = h.shape
        self._ensure_var_pe(V, D, h.device, h.dtype)
        h = h + self.var_pe[:V].unsqueeze(0)
        h = self.cross(h)  # [B, V, d_model]
        return h
