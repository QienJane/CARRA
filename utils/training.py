import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from losses import SupConLoss

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

def train_backbone(model, dataset, cfg, device, stage="base"):
    if stage not in ("base", "incremental"):
        raise ValueError("stage must be 'base' or 'incremental'.")
    if len(dataset) == 0:
        raise ValueError("Backbone training requires current-task samples.")
    device = torch.device(device)
    epochs = int(cfg["base_epochs"] if stage == "base" else cfg["incremental_epochs"])
    lr = float(cfg["base_lr"] if stage == "base" else cfg["incremental_lr"])
    batch_size = int(cfg.get("batch_size", 32))
    views = int(cfg.get("views", 2))
    if epochs < 1 or lr <= 0 or batch_size < 1 or views < 2:
        raise ValueError("Training requires positive epochs, learning rate and batch size, and at least two views.")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=int(cfg.get("workers", 0)), pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
    )
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = None
    if stage == "base":
        final_lr = float(cfg.get("base_final_lr", 1e-6))
        if not 0 < final_lr <= lr:
            raise ValueError("base_final_lr must be positive and no greater than base_lr.")
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=final_lr / lr,
            total_iters=max(epochs - 1, 1),
        )
    criterion = SupConLoss(float(cfg.get("temperature", 0.07)))
    for epoch in range(epochs):
        model.train()
        loss_sum, sample_count = 0.0, 0
        for batch in loader:
            x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
            labels = batch["label"].to(device, dtype=torch.long, non_blocking=True)
            if x.ndim != 6 or x.shape[1] != views:
                raise ValueError(f"Training expects N,{views},C,T,V,M, got {tuple(x.shape)}.")
            n = x.shape[0]
            optimizer.zero_grad(set_to_none=True)
            features = model.forward_contrastive(x.flatten(0, 1)).reshape(n, views, -1)
            loss = criterion(features, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite contrastive loss in {stage} epoch {epoch + 1}.")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * n
            sample_count += n
        print(f"{stage} epoch {epoch + 1}/{epochs} loss={loss_sum / sample_count:.6f} "
              f"lr={optimizer.param_groups[0]['lr']:.3g} samples={sample_count}", flush=True)
        if scheduler is not None and epoch + 1 < epochs:
            scheduler.step()
    model.eval()
    return model

@torch.no_grad()
def extract_features(model, dataset, device, batch_size=32, workers=0):
    device = torch.device(device)
    model.to(device).eval()
    if len(dataset) == 0:
        return np.empty((0, model.feature_dim), dtype=np.float32), np.empty(0, dtype=np.int64)
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, drop_last=False,
        num_workers=int(workers), pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
    )
    feature_batches, label_batches = [], []
    for batch in loader:
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
        if x.ndim != 5:
            raise ValueError(f"Feature extraction expects N,C,T,V,M, got {tuple(x.shape)}.")
        features = model.forward_feature(x)
        if not torch.isfinite(features).all():
            raise RuntimeError("Nonfinite backbone features.")
        feature_batches.append(features.cpu().numpy().astype(np.float32, copy=False))
        label_batches.append(batch["label"].cpu().numpy().astype(np.int64, copy=False))
    return np.concatenate(feature_batches), np.concatenate(label_batches)
