import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import wandb
from tqdm.auto import tqdm

def train(factor_model, dataloader, optimizer, scheduler, args):
    device = args.device
    factor_model.to(device)
    factor_model.train()
    total_loss = 0
    with tqdm(total=len(dataloader), desc="Training") as pbar:
        for char_with_label, _ in dataloader:
            char = char_with_label[:, :, :, :-1] 
            # 提取收益标签：最后一列
            returns = char_with_label[:, :, :, -1]
            
            inputs = char.float().to(device)
            # 获取时间序列最后一天 (seq_len 的末尾) 的截面收益，并增加最后 1 维
            # 从 (B, N, seq_len) -> 取末尾变为 (B, N) -> unsqueeze 变为 (B, N, 1)
            labels = returns[:, :, -1].unsqueeze(-1).float().to(device)
            
            optimizer.zero_grad()
            loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factor_model(inputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'batch_loss': loss.item()})
            pbar.update(1)

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(factor_model, dataloader, args):
    device = args.device
    factor_model.to(device)
    factor_model.eval()
    total_loss = 0
    with tqdm(total=len(dataloader), desc="Validation") as pbar:
        for char_with_label, _  in dataloader:
            char = char_with_label[:, :, :, :-1] 
            # 提取收益标签：最后一列
            returns = char_with_label[:, :, :, -1]
            inputs = char.float().to(device)
            # 获取时间序列最后一天 (seq_len 的末尾) 的截面收益，并增加最后 1 维
            # 从 (B, N, seq_len) -> 取末尾变为 (B, N) -> unsqueeze 变为 (B, N, 1)
            labels = returns[:, :, -1].unsqueeze(-1).float().to(device)
            
            loss, _, _, _, _, _ = factor_model(inputs, labels)
            total_loss += loss.item() 
            pbar.update(1)
    avg_loss = total_loss / len(dataloader)
    return avg_loss

@torch.no_grad()
def test(factor_model, dataloader, args):
    device = args.device
    factor_model.to(device)
    factor_model.eval()
    total_loss = 0
    with tqdm(total=len(dataloader), desc="Validation") as pbar:
        for char_with_label, _  in dataloader:
            char = char_with_label[:, :, :, :-1] 
            # 提取收益标签：最后一列
            returns = char_with_label[:, :, :, -1]
            
            inputs = char.float().to(device)
            # 获取时间序列最后一天 (seq_len 的末尾) 的截面收益，并增加最后 1 维
            # 从 (B, N, seq_len) -> 取末尾变为 (B, N) -> unsqueeze 变为 (B, N, 1)
            labels = returns[:, :, -1].unsqueeze(-1).float().to(device)
            
            loss, _, _, _, _, _ = factor_model(inputs, labels)
            total_loss += loss.item() * inputs.size(0)
            pbar.update(1)
    avg_loss = total_loss / len(dataloader)
    return avg_loss
