from typing import Dict, List, Tuple
import os
import argparse
from dataclasses import dataclass

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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


import wandb
from tqdm import tqdm

from args import add_training_args, add_model_args
from dataset.dataset import UpliftDataset, collate_fn
from models.siamese import SiameseNetwork
from models.dragonnet import Dragonnet
from models.encoder import TCNEncoder, RNNEncoder
from utils import set_seed


NUM_ACTIONS = 35
NUM_METHODS = 5


def create_encoder(config):
    if config.backbone_type == "tcn":
        encoder = TCNEncoder([NUM_ACTIONS, NUM_METHODS], [config.embedding_dim, 2], config.feature_dim, num_layers=10, dropout_p=config.dropout, pool=config.pool_type)
    elif config.backbone_type == "lstm":
        encoder = RNNEncoder([NUM_ACTIONS, NUM_METHODS], [config.embedding_dim, 2], config.feature_dim, num_layers=2, dropout_p=config.dropout, rnn_type="lstm")
    elif config.backbone_type == "gru":
        encoder = RNNEncoder([NUM_ACTIONS, NUM_METHODS], [config.embedding_dim, 2], config.feature_dim, num_layers=2, dropout_p=config.dropout, rnn_type="lstm")
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
    return model


def create_optimizer(config, model: nn.Module):
    if config.optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=[config.momentum, 0.999], weight_decay=config.weight_decay)
    elif config.optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=[config.momentum, 0.999], weight_decay=config.weight_decay)
    elif config.optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    return optimizer


def train(config, model: nn.Module, train_loader: DataLoader, device: torch.device, optimizer: optim.Optimizer, epoch: int):
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward
        output = model(batch)

        # Loss
        if config.model_type == "siamese":
            loss = direct_uplift_loss(output, batch, alpha=config.alpha, e_x=0.5)
        elif config.model_type == "dragonnet":
            loss = dragonnet_loss(output, batch, alpha=config.alpha)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
            
    train_loss /= len(train_loader)
    print(f"Train Epoch: {epoch} \tLoss: {train_loss:.6f}")
    wandb.log({"train/loss": train_loss, "epoch": epoch})


def valid(config, model: nn.Module, valid_loader: DataLoader, device: torch.device, epoch: int):
    pass


def direct_uplift_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], alpha: float=0.5, e_x: float=0.5) -> torch.Tensor:
    """Direct uplift loss. out - y1, y0 // batch - y, t"""
    y1 = out["y1"]
    y0 = out["y0"]
    y = batch["y"]
    t = batch["t"]

    z = t * y / e_x - (1-t) * y / (1-e_x)
    y_pred = torch.where(t == 1, y1, y0)

    loss_uplift = F.mse_loss((y1 - y0), z)
    loss_pred = F.binary_cross_entropy_with_logits(y_pred, y)

    total_loss = (1-alpha) * loss_uplift + alpha * loss_pred
    return total_loss


def dragonnet_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], alpha: float=0.5) -> torch.Tensor:
    """Dragonnet loss. out - y1, y0, y // batch - y, t"""
    y1 = out["y1"]
    y0 = out["y0"]
    t_pred = out["t"]
    y = batch["y"]
    t = batch["t"]

    y_pred = torch.where(t == 1, y1, y0)

    loss_uplift = F.mse_loss(y_pred, y)
    loss_pred = F.binary_cross_entropy(t_pred, t)

    total_loss = (1-alpha) * loss_uplift + alpha * loss_pred
    return total_loss


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Set seed
    if config.seed is not None:
        set_seed(config.seed)
    
    # Load data
    raw_datasets = UpliftDataset(config.dataset_path)

    train_set, valid_set = raw_datasets.val_by_user(config.val_ratio, config.dataset_seed)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda data: collate_fn(data, config.max_length, pad_on_right=False), num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, drop_last=False,
        collate_fn=lambda data: collate_fn(data, config.max_length, pad_on_right=False), num_workers=4, pin_memory=True)

    # Load model and optimizer
    model = create_model(config)
    model.to(device)
    optimizer = create_optimizer(config, model)

    for epoch in range(1, config.epochs+1):
        train(config, model, device, train_loader, optimizer, epoch)
        valid(config, model, device, valid_loader, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Uplift Modeling")
    parser = add_training_args(parser)
    parser = add_model_args(parser)
    args = parser.parse_args()

    if not args.disable_wandb:
        wandb.init(project="aaai23", config=args)
        config = wandb.config
    else:
        config = args
    
    main(config)





