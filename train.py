from typing import Dict, List, Tuple, Callable, Any
import os
import argparse
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklift.metrics.metrics import uplift_auc_score, qini_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torch.optim.swa_utils import AveragedModel

import wandb
from tqdm import tqdm

from args import add_training_args, add_model_args, add_dataset_args
from dataset.dataset import UpliftDataset, collate_fn
from models.siamese import SiameseNetwork
from models.dragonnet import Dragonnet
from models.encoder import TCNEncoder, RNNEncoder
from utils import set_seed


NUM_ACTIONS = 35
NUM_METHODS = 5


def create_encoder(config):
    if config.backbone_type == "tcn":
        encoder = TCNEncoder([NUM_ACTIONS, NUM_METHODS], [config.embedding_dim, 2], config.feature_dim, num_layers=config.num_layers, max_length=config.max_length, dropout_p=config.dropout, positional_embedding=not config.no_positional_embedding, flipped_embedding=config.flipped_embedding, no_embedding=config.no_embedding, pool=config.pool_type)
    elif config.backbone_type == "lstm":
        encoder = RNNEncoder([NUM_ACTIONS, NUM_METHODS], [config.embedding_dim, 2], config.feature_dim, num_layers=config.num_layers, max_length=config.max_length, dropout_p=config.dropout, positional_embedding=not config.no_positional_embedding, flipped_embedding=config.flipped_embedding, no_embedding=config.no_embedding, rnn_type="lstm")
    elif config.backbone_type == "gru":
        encoder = RNNEncoder([NUM_ACTIONS, NUM_METHODS], [config.embedding_dim, 2], config.feature_dim, num_layers=config.num_layers, max_length=config.max_length, dropout_p=config.dropout, positional_embedding=not config.no_positional_embedding, flipped_embedding=config.flipped_embedding, no_embedding=config.no_embedding, rnn_type="gru")
    else:
        raise ValueError(f"Unknown backbone type: {config.backbone_type}")
    return encoder


def create_model(config):
    encoder = create_encoder(config)
    if config.model_type == "siamese":
        model = SiameseNetwork(encoder, config.feature_dim, config.dropout)
    elif config.model_type == "dragonnet":
        model = Dragonnet(encoder, config.feature_dim, config.dropout)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    if config.pretrained_path is not None:
        if config.pretrained_path.endswith(".pth") or config.pretrained_path.endswith(".pt"):
            checkpoint = torch.load(config.pretrained_path)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            config.pretrained_path = os.path.join(config.pretrained_path, "best_model.pth")
        checkpoint = torch.load(config.pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded pretrained model from {config.pretrained_path}")
    return model


def create_optimizer(config, model: nn.Module):
    if config.optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=[config.momentum, 0.999], weight_decay=config.weight_decay)
    elif config.optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=[config.momentum, 0.999], weight_decay=config.weight_decay)
    elif config.optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_type}")
    return optimizer


def train(config, model: nn.Module, train_loader: DataLoader, device: torch.device, optimizer: optim.Optimizer, epoch: int) -> Dict[str, float]:
    model.train()
    train_loss = 0.0
    mse_losses = 0.0
    bce_losses = 0.0
    for batch_idx, batch in enumerate(tqdm(train_loader, ncols=80, desc=f'Epoch: {epoch} train', leave=False)):
        optimizer.zero_grad()
        # Forward
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch)
        # Loss
        if config.model_type == "siamese":
            loss, (mse_loss, bce_loss) = direct_uplift_loss(output, batch, alpha=config.alpha, e_x=0.5, return_all=True)
        elif config.model_type == "dragonnet":
            loss, (mse_loss, bce_loss) = dragonnet_loss(output, batch, alpha=config.alpha, return_all=True)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        mse_losses += mse_loss.item()
        bce_losses += bce_loss.item()
            
    train_loss /= len(train_loader)
    mse_losses /= len(train_loader)
    bce_losses /= len(train_loader)

    print(f"Train Epoch: {epoch} \tLoss: {train_loss:.6f}")
    metrics = {"train/loss": train_loss, "train/mse_loss": mse_losses, "train/bce_loss": bce_loss, "epoch": epoch}
    return metrics


def valid(config, model: nn.Module, valid_loader: DataLoader, device: torch.device, epoch: int, calc_metrics: Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, float]], prefix: str='valid') -> Dict[str, float]:
    model.eval()
    valid_loss = 0.0
    mse_losses = 0.0
    bce_losses = 0.0

    all_batches = {"y": [], "t": []}
    all_preds = {"y1": [], "y0": [], "t": []}
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(valid_loader, ncols=80, desc=f'Epoch: {epoch} {prefix}', leave=False)):
            
            # To calculate metrics
            all_batches["y"].append(batch["y"].detach().cpu())
            all_batches["t"].append(batch["t"].detach().cpu())

            # Forward
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch)

            all_preds["y1"].append(output["y1"].detach().cpu())
            all_preds["y0"].append(output["y0"].detach().cpu())
            if "t" in output:
                all_preds["t"].append(output["t"].detach().cpu())
            
            # Loss
            if config.model_type == "siamese":
                loss, (mse_loss, bce_loss) = direct_uplift_loss(output, batch, alpha=config.alpha, e_x=0.5, return_all=True)
            elif config.model_type == "dragonnet":
                loss, (mse_loss, bce_loss) = dragonnet_loss(output, batch, alpha=config.alpha, return_all=True)

            valid_loss += loss.item()
            mse_losses += mse_loss.item()
            bce_losses += bce_loss.item()

    all_batches = {k: torch.cat(v, dim=0) for k, v in all_batches.items()}
    all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items() if len(v) > 0}
    
    valid_loss /= len(valid_loader)
    mse_losses /= len(valid_loader)
    bce_losses /= len(valid_loader)

    metrics = {f"{prefix}/loss": valid_loss, f"{prefix}/mse_loss": mse_losses, f"{prefix}/bce_loss": bce_losses, "epoch": epoch}
    if calc_metrics is not None:
        valid_metrics = calc_metrics(all_batches, all_preds)
        metrics.update({f"{prefix}/{k}": v for k, v in valid_metrics.items()})
    
    return metrics


def calc_metrics(targets: Dict[str, Any], preds: Dict[str, Any]) -> Dict[str, float]:
    metrics = {}
    if "uplift" in preds:
        uplift = preds["uplift"]
    else:
        uplift = preds['y1'] - preds['y0']
    
    metrics["uplift_auc"] = uplift_auc_score(targets["y"], uplift, targets["t"])
    metrics["qini_auc"] = qini_auc_score(targets["y"], uplift, targets["t"])
    if "t" in preds:
        metrics["treatment_auc"] = roc_auc_score(targets["t"], preds["t"])

    

    return metrics


def save_model(config, model: nn.Module, optimizer: optim.Optimizer, epoch: int, metrics: Dict[str, float], save_dir: str, is_best: bool=False):
    if is_best:
        model_path = os.path.join(save_dir, f"best_model.pth")
    else:
        model_path = os.path.join(save_dir, f"model_{epoch}.pth")
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics, "config": dict(config)}, model_path)


def direct_uplift_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], alpha: float=0.5, e_x: float=0.5, return_all: bool=False) -> torch.Tensor:
    """Direct uplift loss. out - y1, y0 // batch - y, t"""
    y1 = out["y1"]
    y0 = out["y0"]
    y = batch["y"]
    t = batch["t"]

    z = t * y / e_x - (1-t) * y / (1-e_x)
    y_pred = torch.where(t == 1, y1, y0)

    loss_uplift = F.mse_loss((y1 - y0), z) # {(y1_hat - y0_hat) - z}^2
    loss_pred = F.binary_cross_entropy(y_pred, y) # {y|T=t, y_true}

    total_loss = (1-alpha) * loss_uplift + alpha * loss_pred
    if return_all:
        return total_loss, (loss_uplift, loss_pred)
    else:
        return total_loss


def dragonnet_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], alpha: float=0.5, return_all: bool=False) -> torch.Tensor:
    """Dragonnet loss. out - y1, y0, y // batch - y, t"""
    y1 = out["y1"]
    y0 = out["y0"]
    t_pred = out["t"]
    y = batch["y"]
    t = batch["t"]

    y_pred = torch.where(t == 1, y1, y0)

    loss_uplift = F.mse_loss(y_pred, y) # E[(y1 - y0)]
    loss_pred = F.binary_cross_entropy(t_pred, t) # E[T=1|X]   T <-//- e(X) <-- X --> Y
 
    total_loss = (1-alpha) * loss_uplift + alpha * loss_pred
    if return_all:
        return total_loss, (loss_uplift, loss_pred)
    else:
        return total_loss


def calc_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Set seed
    if config.seed is not None:
        set_seed(config.seed)
    
    # Load multiple datasets
    train_sets, valid_sets = [], []
    for dataset_path in config.dataset_path:
        raw_datasets = UpliftDataset(dataset_path)
        train_set, valid_set = raw_datasets.split(by='user', ratio=config.val_ratio, random_state=config.dataset_seed)
        train_sets.append(train_set)
        valid_sets.append(valid_set)
    train_set, valid_set = ConcatDataset(train_sets), ConcatDataset(valid_sets)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda data: collate_fn(data, config.max_length, pad_on_right=False), num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, drop_last=False,
        collate_fn=lambda data: collate_fn(data, config.max_length, pad_on_right=False), num_workers=4, pin_memory=True)
    
    if config.test_path is not None:
        test_loader = {}
        for test_path in config.test_path:
            PROPENSITY = 0.5
            testset_name = os.path.basename(test_path)
            raw_test_datasets = UpliftDataset(test_path)
            test_set, _ = raw_test_datasets.split(by='test', ratio=PROPENSITY, random_state=config.dataset_seed)
            test_loader[testset_name] = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, drop_last=False,
                collate_fn=lambda data: collate_fn(data, config.max_length, pad_on_right=False), num_workers=4, pin_memory=True)
        msg = f"Train size: {len(train_set)}, valid size: {len(valid_set)}, test size:" 
        for k, v in test_loader.items():
            msg += f" {k}: {len(v.dataset)}"
        print(msg)
    else:
        test_set = None
        test_loader = None
        print(f"Train size: {len(train_set)}, valid size: {len(valid_set)}, no test set")

    # Load model and optimizer
    model = create_model(config)
    model.to(device)
    if not config.disable_wandb:
        wandb.watch(model, log="all", log_freq=100)
    if config.use_swa:
        swa_model = AveragedModel(model)
    else:
        swa_model = None

    print(f"Model has {calc_num_params(model):,} trainable parameters")
    optimizer = create_optimizer(config, model)

    best_epoch, best_metric = 0, 0.0
    for epoch in range(1, config.epochs+1):

        all_metrics = {}

        train_metrics = train(config, model, train_loader, device, optimizer, epoch)
        if config.use_swa:
            swa_model.update_parameters(model)

        all_metrics.update(train_metrics)
        if not config.disable_wandb:
            wandb.log(train_metrics)
        
        if epoch % config.eval_every == 0:
            valid_metrics = valid(config, swa_model if config.use_swa else model, valid_loader, device, epoch, calc_metrics=calc_metrics, prefix='valid')
            if valid_metrics['valid/uplift_auc'] > best_metric:
                best_metric = valid_metrics['valid/uplift_auc']
                best_epoch = epoch
                if not config.disable_wandb:
                    wandb.run.summary["best_metric"] = best_metric
                    wandb.run.summary["best_epoch"] = best_epoch
                save_model(config, swa_model.module if config.use_swa else model, optimizer, epoch, all_metrics, config.save_dir, is_best=True)
                print(f"Best model saved at epoch {epoch}")

            all_metrics.update(valid_metrics)
            if not config.disable_wandb:
                wandb.log(valid_metrics)
        
            if test_loader is not None:
                for testset_name in test_loader:
                    test_metrics = valid(config, swa_model.module if config.use_swa else model, test_loader[testset_name], device, epoch, calc_metrics=calc_metrics, prefix=f'test/{testset_name}')
                    all_metrics.update(test_metrics)
                    if not config.disable_wandb:
                        wandb.log(test_metrics)
                all_metrics.update(test_metrics)

        if epoch % config.save_every == 0:
            save_model(config, swa_model.module if config.use_swa else model, optimizer, epoch, all_metrics, config.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Uplift Modeling")
    parser = add_dataset_args(parser)
    parser = add_training_args(parser)
    parser = add_model_args(parser)
    args = parser.parse_args()

    model_name = f"{args.model_type}_{args.backbone_type}_lr{args.lr:1.0e}_fdim{args.feature_dim}"
    if args.flag is not None: model_name += f"_{args.flag}"
    model_name += f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if args.save_dir is not None:
        args.save_dir = os.path.join(args.save_dir, model_name)
        os.makedirs(args.save_dir, exist_ok=True)

    if not args.disable_wandb:
        wandb.init(project="aaai23", config=args, name=model_name)
        config = wandb.config
    else:
        config = args
    
    main(config)





