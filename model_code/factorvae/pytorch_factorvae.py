import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import warnings

from qlib.model.base import Model
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.utils import get_or_create_path

# 导入原生 FactorVAE
from .factorVAE.factor_VAE import FactorVAE

class FactorVAEModel(Model):
    def __init__(
        self,
        # 1. 替换为全新的规范参数名
        num_feature=158,
        seq_len=20,
        num_latent=158,
        num_factor=48,
        num_portfolio=48,
        hidden_size=48,
        
        gamma=1.0,            
        n_epochs=200,
        lr=0.001,
        early_stop=20,
        batch_size=256,        
        optimizer="adam",
        GPU=6,
        seed=None,
        **kwargs,
    ):
        self.logger = get_module_logger("FactorVAE")
        self.logger.info("Initializing Batched FactorVAE Wrapper...")
        
        # 2. 将类属性对齐
        self.num_feature = num_feature
        self.seq_len = seq_len
        self.num_latent = num_latent
        self.num_factor = num_factor
        self.num_portfolio = num_portfolio
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        self.early_stop = early_stop
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed=seed


        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # 3. 实例化底层 FactorVAE 时，传入规范后的参数
        self.model = FactorVAE(
            num_feature=self.num_feature,
            num_latent=self.num_latent,
            num_factor=self.num_factor,
            num_portfolio=self.num_portfolio,
            hidden_size=self.hidden_size,
            seq_len=self.seq_len
        ).to(self.device)

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.model_config = {
            "num_feature": num_feature,
            "seq_len": seq_len,
            "num_latent": num_latent,
            "num_factor": num_factor,
            "num_portfolio": num_portfolio,
            "hidden_size": hidden_size,
            "gamma": gamma,            
            "n_epochs": n_epochs,
            "lr": lr,
            "early_stop": early_stop,
            "batch_size": batch_size,        
            "optimizer": optimizer,
        }
        self.log_filename = f"{self.__class__.__name__}_train_log.txt"
        self.fitted = False

    def _extract_batch_data(self, sampler, batch_dates, idx):
        """
        动态批量数据提取器：按当前批次的最大股票数(max_N)进行动态补齐，摒弃 fixed stock_size。
        """
        batch_x_list, batch_y_list = [], []
        N_todays = [] 

        for date in batch_dates:
            pos_indices = np.where(idx.get_level_values("datetime") == date)[0]
            if len(pos_indices) == 0:
                continue
            
            day_data = np.stack([sampler[i] for i in pos_indices])
            x_val = day_data[:, :, :-1]
            y_val = day_data[:, -1, -1]
            
            batch_x_list.append(x_val)
            batch_y_list.append(y_val)
            N_todays.append(x_val.shape[0])

        if not batch_x_list:
            return None, None, []

        # 获取当前批次最大的股票数量，用作动态 Padding 的基准
        max_N = max(N_todays)
        
        batch_x, batch_y = [], []
        for x_val, y_val in zip(batch_x_list, batch_y_list):
            N_today = x_val.shape[0]
            
            # 如果当天的股票数少于 max_N，则补齐 0
            if N_today < max_N:
                pad_len = max_N - N_today
                pad_x = np.zeros((pad_len, self.seq_len, self.num_feature))
                x_val = np.concatenate([x_val, pad_x], axis=0)
                
                pad_y = np.zeros(pad_len)
                y_val = np.concatenate([y_val, pad_y], axis=0)
                
            batch_x.append(x_val)
            batch_y.append(y_val)

        batch_x = np.stack(batch_x) 
        batch_y = np.stack(batch_y) 

        # 轴调换：[B, seq_len, max_N, num_feature]
        batch_x = np.transpose(batch_x, (0, 2, 1, 3))
        
        batch_x = np.nan_to_num(batch_x, nan=0.0)
        batch_y = np.nan_to_num(batch_y, nan=0.0)

        x_tensor = torch.from_numpy(batch_x).float().to(self.device)
        y_tensor = torch.from_numpy(batch_y).float().to(self.device)

        return x_tensor, y_tensor, N_todays

    def train_epoch(self, sampler):
        self.model.train()
        idx = sampler.get_index()
        dates = idx.get_level_values("datetime").unique().sort_values()
        dates = np.random.permutation(dates) 

        total_loss = 0.0
        total_samples = 0
        
        for i in range(0, len(dates), self.batch_size):
            batch_dates = dates[i : i + self.batch_size]
            current_b_size = len(batch_dates)
            
            # 彻底去掉 Mask
            x_tensor, y_tensor, _ = self._extract_batch_data(sampler, batch_dates, idx)
            if x_tensor is None:
                continue

            # 去掉传给底层模型的 mask 参数
            loss = self.model.run_model(
                characteristics=x_tensor, 
                future_returns=y_tensor, 
                gamma=self.gamma
            )

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
            
            total_loss += loss.item() * current_b_size
            total_samples += current_b_size
            
        return total_loss / total_samples

    def test_epoch(self, sampler):
        self.model.eval()
        idx = sampler.get_index()
        dates = idx.get_level_values("datetime").unique().sort_values()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for i in range(0, len(dates), self.batch_size):
                batch_dates = dates[i : i + self.batch_size]
                current_b_size = len(batch_dates)
                
                x_tensor, y_tensor, _ = self._extract_batch_data(sampler, batch_dates, idx)
                if x_tensor is None:
                    continue
                    
                loss = self.model.run_model(
                    characteristics=x_tensor, 
                    future_returns=y_tensor, 
                    gamma=self.gamma
                )
                
                total_loss += loss.item() * current_b_size
                total_samples += current_b_size
                
        return total_loss / total_samples

    def fit(self, dataset: TSDatasetH, evals_result=dict(), save_path=None):
        sampler_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        sampler_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        
        if sampler_train.empty:
            raise ValueError("Training data is empty!")
            
        sampler_train.config(fillna_type="ffill+bfill")
        if not sampler_valid.empty:
            sampler_valid.config(fillna_type="ffill+bfill")

        save_path = get_or_create_path(save_path)
        best_loss = np.inf
        stop_steps = 0
        best_epoch = 0

        self.logger.info("Training Batched FactorVAE...")
        self.fitted = True
        best_param = copy.deepcopy(self.model.state_dict())
        with open(self.log_filename, "a", encoding="utf-8") as f:
            f.write(f"========== {self.__class__.__name__} 训练配置 ==========\n")
            for key, value in self.model_config.items():
                f.write(f"{key}: {value}\n")
            f.write("===================================================\n\n")
            f.write("========== 训练过程 ==========\n")
            
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(sampler_train)
            
            if not sampler_valid.empty:
                val_loss = self.test_epoch(sampler_valid)
                self.logger.info(f"Epoch {step}: train_loss {train_loss:.4f}, valid_loss {val_loss:.4f}")
                with open(self.log_filename, "a", encoding="utf-8") as f:
                    f.write(f"Epoch {step}: train_loss {train_loss:.4f}, valid_loss {val_loss:.4f}\n")
                if val_loss < best_loss:
                    best_loss = val_loss
                    stop_steps = 0
                    best_epoch = step
                    best_param = copy.deepcopy(self.model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        self.logger.info("Early stop triggered.")
                        break
            else:
                self.logger.info(f"Epoch {step}: train_loss {train_loss:.4f}")
                
        with open(self.log_filename, "a", encoding="utf-8") as f:
            f.write(f"Best Loss: {best_loss:.6f} @ Epoch {best_epoch}\n")
        self.logger.info(f"Best Loss: {best_loss:.6f} @ Epoch {best_epoch}")
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, dataset: TSDatasetH, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        sampler_test = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        sampler_test.config(fillna_type="ffill+bfill")
        
        self.model.eval()
        
        preds_sample = []
        preds_mu = []
        preds_sigma = []
        indices = []
        
        idx = sampler_test.get_index()
        dates = idx.get_level_values("datetime").unique().sort_values()

        with torch.no_grad():
            for i in range(0, len(dates), self.batch_size):
                batch_dates = dates[i : i + self.batch_size]
                
                x_tensor, _, N_todays = self._extract_batch_data(sampler_test, batch_dates, idx)
                if x_tensor is None:
                    continue
                
                pred_returns, mu_dec, sigma_dec = self.model.prediction(characteristics=x_tensor)
                
                for b_idx, date in enumerate(batch_dates):
                    p_ret = pred_returns[b_idx].squeeze().cpu().numpy()
                    p_mu = mu_dec[b_idx].squeeze().cpu().numpy()
                    p_sig = sigma_dec[b_idx].squeeze().cpu().numpy()
                    
                    # 【重要修正】：直接使用当天的真实股票数进行截断，彻底淘汰 stock_size 限制
                    actual_N = N_todays[b_idx] 
                    
                    preds_sample.append(p_ret[:actual_N])
                    preds_mu.append(p_mu[:actual_N])
                    preds_sigma.append(p_sig[:actual_N])
                    
                    pos_indices = np.where(idx.get_level_values("datetime") == date)[0]
                    for idx_pos in pos_indices[:actual_N]:
                        indices.append(idx[idx_pos])

        flat_sample = np.concatenate(preds_sample)
        flat_mu = np.concatenate(preds_mu)
        flat_sigma = np.concatenate(preds_sigma)
        
        multi_idx = pd.MultiIndex.from_tuples(indices, names=["datetime", "instrument"])
        
        return pd.DataFrame({
            "pred_sample": flat_sample,  
            "pred_mu": flat_mu,          
            "pred_sigma": flat_sigma     
        }, index=multi_idx)