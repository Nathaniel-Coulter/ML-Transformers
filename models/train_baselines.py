#!/usr/bin/env python3
# src/train_baselines.py
import argparse, os, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# our modules
from src.loader import PointwiseDataset, PatchDataset, VarTokenDataset
from models.encoders import EncPointwise, EncPatchTST, EnciTransformer, CrossLite

# -------------------------
# Small heads / utilities
# -------------------------

class LinearHead(nn.Module):
    """
    Projects per-variate embeddings [B, V, d] -> per-variate predictions [B, V].
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
    def forward(self, h):
        # h: [B, V, d] -> [B, V]
        return self.proj(h).squeeze(-1)

class PointwiseAdapterHead(nn.Module):
    """
    Adapts EncPointwise output [B, L, d] to per-variate predictions [B, V].
    Strategy: temporal pool to [B, d], then shared linear to V.
    """
    def __init__(self, d_model: int, V: int, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(d_model, V)
    def forward(self, h):
        # h: [B, L, d]
        if self.pool == "last":
            z = h[:, -1, :]
        else:
            z = h.mean(dim=1)
        return self.proj(z)  # [B, V]

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(pred: torch.Tensor, y: torch.Tensor):
    mse = torch.mean((pred - y)**2).item()
    mae = torch.mean(torch.abs(pred - y)).item()
    return {"mse": mse, "mae": mae}

# -------------------------
# Main training routine
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/tensors_L252_H1.npz",
                    help="Path to tensors npz produced by src.loader")
    ap.add_argument("--encoder", choices=["pointwise", "patch", "ivar", "cross"],
                    default="pointwise")

    # model hyperparams
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)

    # patching
    ap.add_argument("--patch-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)

    # training hyperparams
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    # options
    ap.add_argument("--eval-test", action="store_true", help="Evaluate on test split after training")
    ap.add_argument("--save", action="store_true", help="Save encoder+head checkpoint to outputs/")
    ap.add_argument("--tag", type=str, default="", help="Optional run tag appended to checkpoint filename")

    args = ap.parse_args()
    set_seed(args.seed)

    # -------------------------
    # Load tensors
    # -------------------------
    if not os.path.exists(args.npz):
        raise FileNotFoundError(f"npz not found: {args.npz}")
    npz = np.load(args.npz, allow_pickle=True)
    Xtr, ytr = npz["X_train"], npz["y_train"]
    Xva, yva = npz["X_val"],   npz["y_val"]
    Xte, yte = npz["X_test"],  npz["y_test"]
    tickers = list(npz["tickers"])

    L, V = Xtr.shape[1], Xtr.shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    # -------------------------
    # Datasets & Encoders
    # -------------------------
    if args.encoder == "pointwise":
        dset_tr = PointwiseDataset(Xtr, ytr)
        dset_va = PointwiseDataset(Xva, yva)

        enc = EncPointwise(
            in_dim=V, d_model=args.d_model, n_heads=args.heads,
            depth=args.depth, ffn=args.ffn, dropout=args.dropout
        ).to(device)
        head = PointwiseAdapterHead(d_model=args.d_model, V=V, pool="mean").to(device)

        def forward_batch(xb):
            h = enc(xb.to(device))          # [B, L, d]
            return head(h)                  # [B, V]

    elif args.encoder == "patch":
        dset_tr = PatchDataset(Xtr, ytr, patch_len=args.patch_len, stride=args.stride)
        dset_va = PatchDataset(Xva, yva, patch_len=args.patch_len, stride=args.stride)

        enc = EncPatchTST(
            patch_len=args.patch_len, d_model=args.d_model, n_heads=args.heads,
            depth=args.depth, ffn=args.ffn, dropout=args.dropout, agg='mean'
        ).to(device)
        head = LinearHead(d_model=args.d_model).to(device)

        def forward_batch(xb):
            h = enc(xb.to(device))          # [B, V, d]
            return head(h)                  # [B, V]

    elif args.encoder == "ivar":
        dset_tr = VarTokenDataset(Xtr, ytr)
        dset_va = VarTokenDataset(Xva, yva)

        enc = EnciTransformer(
            L=L, d_model=args.d_model, n_heads=args.heads,
            depth=args.depth, ffn=args.ffn, dropout=args.dropout
        ).to(device)
        head = LinearHead(d_model=args.d_model).to(device)

        def forward_batch(xb):
            h = enc(xb.to(device))          # [B, V, d]
            return head(h)                  # [B, V]

    else:  # cross
        dset_tr = PatchDataset(Xtr, ytr, patch_len=args.patch_len, stride=args.stride)
        dset_va = PatchDataset(Xva, yva, patch_len=args.patch_len, stride=args.stride)

        enc = CrossLite(
            patch_len=args.patch_len, d_model=args.d_model,
            n_heads_time=args.heads, depth_time=max(1, args.depth-1),
            ffn=args.ffn, dropout=args.dropout,
            n_heads_cross=max(2, args.heads//4), depth_cross=1
        ).to(device)
        head = LinearHead(d_model=args.d_model).to(device)

        def forward_batch(xb):
            h = enc(xb.to(device))          # [B, V, d]
            return head(h)                  # [B, V]

    tr = DataLoader(dset_tr, batch_size=args.bs, shuffle=True, drop_last=True)
    va = DataLoader(dset_va, batch_size=args.bs, shuffle=False, drop_last=False)

    # -------------------------
    # Optimizer / loss
    # -------------------------
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # -------------------------
    # Train loop
    # -------------------------
    best_val = float("inf")
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        enc.train(); head.train()
        train_loss = 0.0
        for xb, yb in tr:
            opt.zero_grad(set_to_none=True)
            pred = forward_batch(xb)            # [B, V]
            loss = loss_fn(pred, yb.to(pred.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= max(1, len(tr))

        # validation
        enc.eval(); head.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_mae = 0.0
            n_batches = 0
            for xb, yb in va:
                pred = forward_batch(xb)
                yb = yb.to(pred.device)
                val_loss += loss_fn(pred, yb).item()
                val_mae  += torch.mean(torch.abs(pred - yb)).item()
                n_batches += 1
            val_loss /= max(1, n_batches)
            val_mae  /= max(1, n_batches)

        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss, "val_mae": val_mae})
        print(f"[epoch {epoch:02d}] train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  val_mae={val_mae:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "encoder": enc.state_dict(),
                "head": head.state_dict(),
                "config": vars(args),
                "tickers": tickers,
                "L": L, "V": V
            }

    dur = time.time() - t0
    print(f"[done] best_val_mse={best_val:.6f}  time={dur:.1f}s")

    # -------------------------
    # Optional: evaluate on test
    # -------------------------
    if args.eval_test:
        # Build test set matching the chosen view
        if args.encoder == "pointwise":
            dset_te = PointwiseDataset(Xte, yte)
        elif args.encoder == "ivar":
            dset_te = VarTokenDataset(Xte, yte)
        else:
            dset_te = PatchDataset(Xte, yte, patch_len=args.patch_len, stride=args.stride)
        dl_te = DataLoader(dset_te, batch_size=args.bs, shuffle=False)

        enc.load_state_dict(best_state["encoder"])
        head.load_state_dict(best_state["head"])
        enc.eval(); head.eval()
        mse = mae = n = 0
        with torch.no_grad():
            for xb, yb in dl_te:
                pred = forward_batch(xb)
                yb = yb.to(pred.device)
                mse += torch.mean((pred - yb)**2).item() * len(xb)
                mae += torch.mean(torch.abs(pred - yb)).item() * len(xb)
                n += len(xb)
        mse /= n; mae /= n
        print(f"[test] mse={mse:.6f}  mae={mae:.6f}")

    # -------------------------
    # Optional: save checkpoint + history
    # -------------------------
    if args.save:
        os.makedirs("outputs", exist_ok=True)
        tag = f"_{args.tag}" if args.tag else ""
        ckpt_path = f"outputs/{args.encoder}_L{L}_V{V}_d{args.d-model if hasattr(args,'d-model') else args.d_model}_best{tag}.pt"
        # Fix name due to argparse hyphen:
        ckpt_path = f"outputs/{args.encoder}_L{L}_V{V}_d{args.d_model}_best{tag}.pt"
        torch.save(best_state, ckpt_path)
        with open(f"outputs/{args.encoder}_history{tag}.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"[save] checkpoint -> {ckpt_path}")
        print(f"[save] history    -> outputs/{args.encoder}_history{tag}.json")

if __name__ == "__main__":
    main()
