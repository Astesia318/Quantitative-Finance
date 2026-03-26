import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

# ================= 新增：Batch模式截面算子 =================
def zscore_batch(x, mask):
    """ 按天(Batch维度)进行截面标准化 """
    x_masked = x * mask
    valid_counts = mask.sum(dim=1, keepdim=True)
    valid_counts[valid_counts == 0] = 1 # 防除零
    
    means = x_masked.sum(dim=1, keepdim=True) / valid_counts
    diff = (x - means) * mask
    variances = (diff ** 2).sum(dim=1, keepdim=True) / valid_counts
    stds = torch.sqrt(variances)
    stds[stds == 0] = 1e-5
    
    return (x - means) / stds

def drop_extreme_batch(x, mask):
    """ 按天(Batch维度)进行极值过滤 """
    B, N = x.shape
    out_mask = mask.clone()
    for b in range(B):
        valid_indices = torch.where(mask[b])[0]
        day_x = x[b, valid_indices]
        if day_x.shape[0] > 0:
            sorted_tensor, indices = day_x.sort()
            N_day = day_x.shape[0]
            percent_2_5 = int(0.025 * N_day)
            if percent_2_5 > 0:
                drop_indices = torch.cat((indices[:percent_2_5], indices[-percent_2_5:]))
                original_drop_indices = valid_indices[drop_indices]
                out_mask[b, original_drop_indices] = False
    return out_mask
# ==========================================================

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


def pad_collate_fn(batch):
    """ 动态补齐函数，将 B 天内不同数量的股票补齐至 max_N，并生成 mask """
    processed_batch = []
    for item in batch:
        if isinstance(item, torch.Tensor) and item.dim() == 4 and item.shape[0] == 1:
            item = item.squeeze(0)
        elif isinstance(item, np.ndarray) and item.ndim == 4 and item.shape[0] == 1:
            item = np.squeeze(item, axis=0)
        processed_batch.append(item)
    
    max_N = max(item.shape[0] for item in processed_batch)
    B = len(processed_batch)
    _, T, F_dim = processed_batch[0].shape
    
    padded = torch.zeros((B, max_N, T, F_dim), dtype=torch.float32)
    mask = torch.zeros((B, max_N), dtype=torch.bool)
    
    for b, day_data in enumerate(processed_batch):
        N_i = day_data.shape[0]
        if isinstance(day_data, np.ndarray):
            padded[b, :N_i, :, :] = torch.from_numpy(day_data)
        else:
            padded[b, :N_i, :, :] = day_data.clone().detach()
        mask[b, :N_i] = True
        
    return padded, mask


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= '', batch_size=1):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.batch_size = batch_size

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1
        self.model = None
        self.train_optimizer = None
        self.save_path = save_path
        self.save_prefix = save_prefix

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")
        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        # 传入 batch_size 和自定义的 pad_collate_fn
        data_loader = DataLoader(data, sampler=sampler, batch_size=self.batch_size, drop_last=drop_last, collate_fn=pad_collate_fn)
        return data_loader

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for feature, mask in data_loader:
            feature = feature.to(self.device)
            mask = mask.to(self.device)

            features_input = feature[:, :, :, 0:-1]
            label = feature[:, :, -1, -1] # (B, max_N)

            mask = drop_extreme_batch(label, mask)
            label = zscore_batch(label, mask)

            pred = self.model(features_input.float(), mask=mask)
            
            valid_label_mask = ~torch.isnan(label) & mask
            loss = (pred[valid_label_mask] - label[valid_label_mask])**2
            
            if loss.numel() > 0:
                loss = torch.mean(loss)
                losses.append(loss.item())

                self.train_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for feature, mask in data_loader:
            feature = feature.to(self.device)
            mask = mask.to(self.device)

            features_input = feature[:, :, :, 0:-1]
            label = feature[:, :, -1, -1]

            valid_label_mask = ~torch.isnan(label) & mask
            label = zscore_batch(label, valid_label_mask)
                        
            pred = self.model(features_input.float(), mask=mask)
            
            loss = (pred[valid_label_mask] - label[valid_label_mask])**2
            if loss.numel() > 0:
                losses.append(torch.mean(loss).item())

        return float(np.mean(losses))

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (step, train_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR']))
            else: print("Epoch %d, train_loss %.6f" % (step, train_loss))
            
            if train_loss <= self.train_stop_loss_thred:
                best_param = copy.deepcopy(self.model.state_dict())
                torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
                break

    def predict(self, dl_test):
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)
        preds = []
        ic = []
        ric = []

        self.model.eval()
        for feature, mask in test_loader:
            feature = feature.to(self.device)
            mask = mask.to(self.device)
            features_input = feature[:, :, :, 0:-1]
            label = feature[:, :, -1, -1]
            
            with torch.no_grad():
                pred = self.model(features_input.float(), mask=mask).detach().cpu().numpy()
            
            label_np = label.detach().cpu().numpy()
            mask_np = mask.cpu().numpy()
            B = feature.shape[0]
            
            for b in range(B):
                day_mask = mask_np[b]
                day_pred = pred[b][day_mask]
                day_label = label_np[b][day_mask]
                
                preds.append(day_pred.ravel())
                daily_ic, daily_ric = calc_ic(day_pred, day_label)
                ic.append(daily_ic)
                ric.append(daily_ric)

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())
        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }
        return predictions, metrics