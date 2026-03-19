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
        d_feat=20,
        time_span=20,
        stock_size=300,       
        latent_size=64,
        factor_size=16,
        gru_input_size=20,    
        hidden_size=64,
        gamma=1.0,            
        n_epochs=200,
        lr=0.001,
        early_stop=20,
        batch_size=16,        # 【新增】Batch Size：代表一次性并行处理的交易日天数
        optimizer="adam",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        self.logger = get_module_logger("FactorVAE")
        self.logger.info("Initializing Batched FactorVAE Wrapper...")

        self.d_feat = d_feat
        self.time_span = time_span
        self.stock_size = stock_size
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.lr = lr
        self.early_stop = early_stop
        self.batch_size = batch_size
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = FactorVAE(
            characteristic_size=d_feat,
            stock_size=stock_size,
            latent_size=latent_size,
            factor_size=factor_size,
            time_span=time_span,
            gru_input_size=gru_input_size,
            hidden_size=hidden_size,
        ).to(self.device)

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        self.fitted = False

    def _extract_batch_data(self, sampler, batch_dates, idx):
        """
        批量数据提取器：一次性打包多个交易日的截面数据，并生成 Mask
        返回: 
            x_tensor -> [batch_size, time_span, stock_size, d_feat]
            y_tensor -> [batch_size, stock_size]
            mask_tensor -> [batch_size, stock_size]
        """
        batch_x, batch_y, batch_mask = [], [], []
        N_todays = [] 

        for date in batch_dates:
            pos_indices = np.where(idx.get_level_values("datetime") == date)[0]
            if len(pos_indices) == 0:
                continue
            
            day_data = np.stack([sampler[i] for i in pos_indices])
            x_val = day_data[:, :, :-1]
            y_val = day_data[:, -1, -1]
            N_today = x_val.shape[0]
            N_todays.append(N_today)

            # 【新增】：初始化当天的 Mask，全设为 1.0 (真股票)
            day_mask = np.ones(self.stock_size)

            # 对齐当前交易日的股票数到固定大小 (stock_size)
            if N_today < self.stock_size:
                pad_len = self.stock_size - N_today
                pad_x = np.zeros((pad_len, self.time_span, self.d_feat))
                x_val = np.concatenate([x_val, pad_x], axis=0)
                
                pad_y = np.zeros(pad_len)
                y_val = np.concatenate([y_val, pad_y], axis=0)
                
                # 【新增】：将 Padding 进去的假股票位置的 Mask 设为 0.0
                day_mask[N_today:] = 0.0
                
            elif N_today > self.stock_size:
                x_val = x_val[:self.stock_size]
                y_val = y_val[:self.stock_size]
                # 截断时，所有留下的都是真股票，day_mask 保持全 1 即可

            batch_x.append(x_val)
            batch_y.append(y_val)
            batch_mask.append(day_mask) # 收集 Mask

        if not batch_x:
            return None, None, None, []

        batch_x = np.stack(batch_x) 
        batch_y = np.stack(batch_y) 
        batch_mask = np.stack(batch_mask) # 形状: [B, stock_size]

        # 轴调换，迎合原生代码 assert 需求：变为 [B, time_span, stock_size, d_feat]
        batch_x = np.transpose(batch_x, (0, 2, 1, 3))
        
        batch_x = np.nan_to_num(batch_x, nan=0.0)
        batch_y = np.nan_to_num(batch_y, nan=0.0)

        x_tensor = torch.from_numpy(batch_x).float().to(self.device)
        y_tensor = torch.from_numpy(batch_y).float().to(self.device)
        # 【新增】：将 Mask 转为 Tensor 放进 GPU
        mask_tensor = torch.from_numpy(batch_mask).float().to(self.device)

        return x_tensor, y_tensor, mask_tensor, N_todays

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
            # 【修改】：接收解包出来的 mask_tensor
            x_tensor, y_tensor, mask_tensor, _ = self._extract_batch_data(sampler, batch_dates, idx)
            if x_tensor is None:
                continue

            # 【修改】：将 mask 传递给原生的 run_model 函数
            loss = self.model.run_model(
                characteristics=x_tensor, 
                future_returns=y_tensor, 
                gamma=self.gamma,
                mask=mask_tensor  
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
                # 【修改】：接收解包出来的 mask_tensor
                x_tensor, y_tensor, mask_tensor, _ = self._extract_batch_data(sampler, batch_dates, idx)
                if x_tensor is None:
                    continue
                    
                # 【修改】：将 mask 传递给原生的 run_model 函数
                loss = self.model.run_model(
                    characteristics=x_tensor, 
                    future_returns=y_tensor, 
                    gamma=self.gamma,
                    mask=mask_tensor
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

        for step in range(self.n_epochs):
            train_loss = self.train_epoch(sampler_train)
            
            if not sampler_valid.empty:
                val_loss = self.test_epoch(sampler_valid)
                self.logger.info(f"Epoch {step}: train_loss {train_loss:.4f}, valid_loss {val_loss:.4f}")
                
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
        
        # 使用三个列表分别收集采样值、均值(预期收益)、标准差(风险)
        preds_sample = []
        preds_mu = []
        preds_sigma = []
        indices = []
        
        idx = sampler_test.get_index()
        dates = idx.get_level_values("datetime").unique().sort_values()

        with torch.no_grad():
            for i in range(0, len(dates), self.batch_size):
                batch_dates = dates[i : i + self.batch_size]
                
                x_tensor, _, _, N_todays = self._extract_batch_data(sampler_test, batch_dates, idx)
                if x_tensor is None:
                    continue
                
                # 获取全量三个参数
                pred_returns, mu_dec, sigma_dec = self.model.prediction(characteristics=x_tensor)
                
                for b_idx, date in enumerate(batch_dates):
                    # 剥离单天数据
                    p_ret = pred_returns[b_idx].squeeze().cpu().numpy()
                    p_mu = mu_dec[b_idx].squeeze().cpu().numpy()
                    p_sig = sigma_dec[b_idx].squeeze().cpu().numpy()
                    
                    actual_N = min(N_todays[b_idx], self.stock_size)
                    
                    # 切除 Padding，只保留真实的股票数据
                    preds_sample.append(p_ret[:actual_N])
                    preds_mu.append(p_mu[:actual_N])
                    preds_sigma.append(p_sig[:actual_N])
                    
                    pos_indices = np.where(idx.get_level_values("datetime") == date)[0]
                    for idx_pos in pos_indices[:actual_N]:
                        indices.append(idx[idx_pos])

        # 扁平化拼接
        flat_sample = np.concatenate(preds_sample)
        flat_mu = np.concatenate(preds_mu)
        flat_sigma = np.concatenate(preds_sigma)
        
        multi_idx = pd.MultiIndex.from_tuples(indices, names=["datetime", "instrument"])
        
        # 组装成多列 DataFrame 形式返回
        return pd.DataFrame({
            "pred_sample": flat_sample,  # 带有随机性的单次采样预测
            "pred_mu": flat_mu,          # 确定性的预期收益均值 (最核心指标)
            "pred_sigma": flat_sigma     # 预测的不确定性风险 (方差/标准差)
        }, index=multi_idx)