import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import argparse
from module import FactorVAE, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
from dataset import init_data_loader
from train_model import train, validate
from utils import set_seed, DataArgument
import wandb

def main(args:argparse.Namespace, data_args):
    set_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"*************** Using {device} ***************")
    args.device = device
    
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor).to(device)
    
    dataset = pd.read_pickle(args.dataset)
    
    # 【修改】：传入 args.batch_size 
    train_loader = init_data_loader(dataset, data_args.seq_len, shuffle=True, start=data_args.start_time, end=data_args.fit_end_time, select_feature=data_args.select_feature, batch_size=args.batch_size)
    val_loader = init_data_loader(dataset, data_args.seq_len, shuffle=False, start=data_args.val_start_time, end=data_args.val_end_time, select_feature=data_args.select_feature, batch_size=args.batch_size)
    
    optimizer = optim.Adam(factorVAE.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=1e-6  # 学习率最小不会低于这个值
    )

    best_val_loss = float('inf')

    patience = getattr(args, 'early_stop', 20)
    log_filename="./FactorVAE/VAElog.txt"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    if args.wandb:
        wandb.init(project="FactorVAE", config=args, name=args.run_name)
        wandb.watch(factorVAE)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write("========== VAE 训练配置 ==========\n")
        f.write(str(args)+"\n")
        f.write("===================================================\n\n")
        f.write("========== 训练过程 ==========\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train(factorVAE, train_loader, optimizer, scheduler, args)
        val_loss = validate(factorVAE, val_loader, args)
        scheduler.step()

        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch+1}: train_loss {train_loss:.4f}, valid_loss {val_loss:.4f}\n")

        if args.wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]['lr']})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            stop_steps = 0  # 发现更优模型，重置容忍度计数器
            
            save_root = os.path.join(args.save_dir, f"VAE-factor{args.num_factor}_hdn{args.hidden_size}_M{args.num_portfolio}.pt")
            torch.save(factorVAE.state_dict(), save_root)
            print(f"Model saved at Epoch {best_epoch}! (Loss: {best_val_loss:.4f})")
            
        else:
            stop_steps += 1
            print(f"No improvement for {stop_steps} epochs.")
            
            if stop_steps >= patience:
                print(f"Early stop triggered! Training stopped at epoch {epoch+1}.")
                with open(log_filename, "a", encoding="utf-8") as f:
                    f.write(f"\n========== 早停触发 (Early stop) 停于 Epoch {epoch+1} ==========\n")
                break  # 触发早停，跳出循环

    print(f"Training Finished! Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(f"\n========== 训练结束 ==========\n")
        f.write(f"最佳 Epoch: {best_epoch}\n")
        f.write(f"最佳 Val Loss: {best_val_loss:.6f}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    
    # 【修改】：新增 batch_size 参数（代表一批次并行处理的天数，默认为 16 天）
    parser.add_argument('--batch_size', type=int, default=512, help='batch size in days')
    
    parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
    parser.add_argument('--normalize', type=bool, default=True, help='normalize')
    
    parser.add_argument('--num_latent', type=int, default=158, help='latent size')
    parser.add_argument('--num_portfolio', type=int, default=128, help='portfolio size')
    parser.add_argument('--num_factor', type=int, default=60, help='factor size')
    parser.add_argument('--hidden_size', type=int, default=60, help='hidden size')

    parser.add_argument('--dataset', type=str, default='./FactorVAE/data/csi_data.pkl', help='dataset to use')
    parser.add_argument('--start_time', type=str, default='2009-01-01', help='start time')
    parser.add_argument('--fit_end_time', type=str, default='2016-12-31', help='fit end time')
    parser.add_argument('--val_start_time', type=str, default='2017-01-01', help='validation start time')
    parser.add_argument('--val_end_time', type=str, default='2019-12-31', help='validation end time')
    parser.add_argument('--end_time', type=str, default='2023-12-31', help='end time')

    parser.add_argument('--gpu', type=int, default=2, help='gpu device')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--run_name', type=str, default='VAE-Revision2', help='name of the run')
    parser.add_argument('--save_dir', type=str, default='./FactorVAE/best_models', help='directory to save model')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--wandb', action='store_true', help='use wandb')
    
    args = parser.parse_args()
    data_args = DataArgument(
        start_time=args.start_time,
        end_time=args.end_time,
        fit_end_time=args.fit_end_time,
        val_start_time=args.val_start_time,
        val_end_time=args.val_end_time,
        seq_len=args.seq_len,
    )
    main(args, data_args)