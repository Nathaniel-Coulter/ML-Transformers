# models/extra_encoders.py
from __future__ import annotations
import warnings
import importlib
import torch
import torch.nn as nn

# ---- small helpers -----------------------------------------------------------

def _import_module_candidates(candidates: list[str]):
    """
    Try importing a list of dotted modules. Return the first imported module.
    Raise last error if all fail.
    """
    last_err = None
    for mod in candidates:
        try:
            return importlib.import_module(mod)
        except Exception as e:
            last_err = e
    raise last_err

def _get_class_any(mod, names: list[str]):
    """
    From an imported module `mod`, return the first present class among `names`.
    """
    for name in names:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"None of {names} present in module {mod.__name__}")

def _time_pool(seq_feats: torch.Tensor, mode: str = "mean"):
    # seq_feats: [B, L, V, d] -> [B, V, d]
    if mode == "last":
        return seq_feats[:, -1]
    return seq_feats.mean(dim=1)

# ---- lightweight local stand-ins (used only if all real imports fail) --------

class _LocalSeqEncoder(nn.Module):
    """Tiny transformer-like block that outputs [B,V,d]; for fallback only."""
    def __init__(self, L, V, d_model=256, n_heads=8, depth=3, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4*d_model,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.proj_in  = nn.Linear(V, d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x):  # x: [B,L,V]
        h = self.proj_in(x)             # [B,L,d]
        h = self.enc(h)                  # [B,L,d]
        h = self.proj_out(h)             # [B,L,d]
        return h.mean(dim=1)[:, None, :].expand(-1, x.shape[2], -1)  # [B,V,d]

# ---- Autoformer ---------------------------------------------------------------

class EncAutoformer(nn.Module):
    """
    Uses third_party/Autoformer (pref), else a local fallback.
    Expect output [B,V,d].
    """
    def __init__(self, L:int, V:int, d_model:int=256, depth:int=3, n_heads:int=8,
                 dropout:float=0.1, time_pool:str="mean", **kw):
        super().__init__()
        self.time_pool = time_pool
        try:
            # In thuml/Autoformer, the model class is models.Autoformer.Model
            mod = _import_module_candidates([
                "models.Autoformer",      # when PYTHONPATH points at third_party/Autoformer
                "autoformer.models.Autoformer",  # if installed as a package
            ])
            AutoCls = _get_class_any(mod, ["Model", "Autoformer"])
            # Minimal args that are accepted by most forks
            self.backbone = AutoCls(seq_len=L, label_len=0, pred_len=1,
                                    enc_in=V, dec_in=V, c_out=V,
                                    d_model=d_model, n_heads=n_heads, e_layers=depth, d_layers=1,
                                    dropout=dropout, **kw)
            self._mode = "real"
        except Exception as e:
            warnings.warn(f"[Autoformer] Falling back to local encoder: {e}")
            self.backbone = _LocalSeqEncoder(L, V, d_model, n_heads, depth, dropout)
            self._mode = "fallback"

    def forward(self, x):  # x: [B,L,V]
        out = self.backbone(x)
        # Common returns: [B, L, V] (pred) or [B, L, V, d] or [B, V, d]
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 3 and out.shape[1] == x.shape[1]:   # [B,L,V]
            out = out.unsqueeze(-1)                          # [B,L,V,1]
        if out.dim() == 4:                                   # [B,L,V,d]
            return _time_pool(out, "mean")                   # [B,V,d]
        if out.dim() == 3 and out.shape[1] == x.shape[2]:    # [B,V,d]
            return out
        # Fallback path returns [B,V,d] already
        return out

# ---- FEDformer ----------------------------------------------------------------

class EncFEDformer(nn.Module):
    def __init__(self, L:int, V:int, d_model:int=256, depth:int=3, n_heads:int=8,
                 dropout:float=0.1, time_pool:str="mean", **kw):
        super().__init__()
        self.time_pool = time_pool
        try:
            mod = _import_module_candidates([
                "models.FEDformer",
                "fedformer.models.FEDformer",
            ])
            FEDCls = _get_class_any(mod, ["Model", "FEDformer"])
            self.backbone = FEDCls(seq_len=L, label_len=0, pred_len=1,
                                   enc_in=V, dec_in=V, c_out=V,
                                   d_model=d_model, n_heads=n_heads, e_layers=depth, d_layers=1,
                                   dropout=dropout, **kw)
            self._mode = "real"
        except Exception as e:
            warnings.warn(f"[FEDformer] Falling back to local encoder: {e}")
            self.backbone = _LocalSeqEncoder(L, V, d_model, n_heads, depth, dropout)
            self._mode = "fallback"

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 3 and out.shape[1] == x.shape[1]:
            out = out.unsqueeze(-1)
        if out.dim() == 4:
            return _time_pool(out, "mean")
        if out.dim() == 3 and out.shape[1] == x.shape[2]:
            return out
        return out

# ---- Informer -----------------------------------------------------------------

class EncInformer(nn.Module):
    def __init__(self, L:int, V:int, d_model:int=256, depth:int=3, n_heads:int=8,
                 dropout:float=0.1, time_pool:str="last", **kw):
        super().__init__()
        self.time_pool = time_pool
        try:
            mod = _import_module_candidates([
                "models.model_informer",          # many forks
                "models.model",                   # some forks
                "informer.models.model_informer", # if packaged
            ])
            InfCls = _get_class_any(mod, ["Informer", "Model"])
            # Informer signature differs; these are common fields
            self.backbone = InfCls(enc_in=V, dec_in=V, c_out=V,
                                   seq_len=L, label_len=0, out_len=1,
                                   d_model=d_model, n_heads=n_heads,
                                   e_layers=depth, d_layers=1, dropout=dropout, **kw)
            self._mode = "real"
        except Exception as e:
            warnings.warn(f"[Informer] Falling back to local encoder: {e}")
            self.backbone = _LocalSeqEncoder(L, V, d_model, n_heads, depth, dropout)
            self._mode = "fallback"

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 3 and out.shape[1] == x.shape[1]:
            out = out.unsqueeze(-1)
        if out.dim() == 4:
            return _time_pool(out, self.time_pool)
        if out.dim() == 3 and out.shape[1] == x.shape[2]:
            return out
        return out

# ---- TimesNet -----------------------------------------------------------------

class EncTimesNet(nn.Module):
    def __init__(self, L:int, V:int, d_model:int=256, depth:int=3, n_heads:int=8,
                 dropout:float=0.1, time_pool:str="mean", **kw):
        super().__init__()
        self.time_pool = time_pool
        try:
            mod = _import_module_candidates([
                "models.TimesNet",
                "TimesNet.models.TimesNet",
            ])
            TNCls = _get_class_any(mod, ["Model", "TimesNet"])
            self.backbone = TNCls(seq_len=L, pred_len=1, enc_in=V, c_out=V,
                                  d_model=d_model, e_layers=depth, n_heads=n_heads,
                                  dropout=dropout, **kw)
            self._mode = "real"
        except Exception as e:
            warnings.warn(f"[TimesNet] Falling back to local encoder: {e}")
            self.backbone = _LocalSeqEncoder(L, V, d_model, n_heads, depth, dropout)
            self._mode = "fallback"

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 3 and out.shape[1] == x.shape[1]:
            out = out.unsqueeze(-1)
        if out.dim() == 4:
            return _time_pool(out, "mean")
        if out.dim() == 3 and out.shape[1] == x.shape[2]:
            return out
        return out

# ---- TimeXer ------------------------------------------------------------------

class EncTimeXer(nn.Module):
    def __init__(self, L:int, V:int, d_model:int=256, depth:int=3, n_heads:int=8,
                 dropout:float=0.1, patch_len:int=16, stride:int=8, time_pool:str="mean", **kw):
        super().__init__()
        self.time_pool = time_pool
        try:
            mod = _import_module_candidates([
                "models.timexer",
                "TimeXer.models.timexer",
            ])
            TXCls = _get_class_any(mod, ["Model", "TimeXer"])
            self.backbone = TXCls(seq_len=L, patch_len=patch_len, stride=stride,
                                  n_vars=V, d_model=d_model, n_heads=n_heads,
                                  depth=depth, dropout=dropout, **kw)
            self._mode = "real"
        except Exception as e:
            warnings.warn(f"[TimeXer] Falling back to local encoder: {e}")
            self.backbone = _LocalSeqEncoder(L, V, d_model, n_heads, depth, dropout)
            self._mode = "fallback"

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 3 and out.shape[1] == x.shape[1]:
            out = out.unsqueeze(-1)
        if out.dim() == 4:
            return _time_pool(out, "mean")
        if out.dim() == 3 and out.shape[1] == x.shape[2]:
            return out
        return out
