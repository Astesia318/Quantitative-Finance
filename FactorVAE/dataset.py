import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from tqdm.auto import tqdm
def np_ffill(arr):
    """
    修复后的二维 NumPy 向前填充 (Forward Fill) 函数
    """
    mask = np.isnan(arr)
    # 增加 [:, None] 将 (20,) 变成 (20, 1)，从而可以和 (20, 159) 广播
    idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
    
    # 沿着时间轴向下累加最大索引，把上一个有效值的行号传递下来
    np.maximum.accumulate(idx, axis=0, out=idx)
    
    # 通过行索引矩阵 idx 和 列索引数组 获取对应的值
    out = arr[idx, np.arange(arr.shape[1])]
    return out

class TSDataSampler:
    def __init__(self, data: pd.DataFrame, start, end, step_len: int, fillna_type: str = "none", dtype=None, flt_data=None):
        self.data = data
        self.start = start
        self.end = end
        self.step_len = step_len
        self.fillna_type = fillna_type
        
    # (省略原有的 get_index() 等方法实现，与原版保持一致即可)

class TSDatasetH(Dataset):
    def __init__(self, df, step_len, start, end, fillna_type='ffill+bfill'):
        super().__init__()
        self.step_len = step_len
        self.start = start
        self.end = end
        self.fillna_type = fillna_type
        
        # 1. 提取全局时间轴并确定有效范围
        self.dates = df.index.get_level_values('datetime').unique().sort_values()
        start_idx = self.dates.searchsorted(pd.to_datetime(start))
        end_idx = self.dates.searchsorted(pd.to_datetime(end), side='right')
        valid_dates = self.dates[start_idx:end_idx]
        
        # 记录所有的有效索引用于 DataLoader 采样
        self.filtered_df = df.loc[valid_dates]
        self.indices = self.filtered_df.index.tolist()
        
        # 为了 O(1) 查找时间索引，建立一个字典映射
        self.date_to_idx = {date: idx for idx, date in enumerate(self.dates)}
        
        # ==========================================================
        # 核心性能优化：在内存中预处理所有股票数据为 Numpy 数组
        # ==========================================================
        print("[Dataset] 正在预处理数据至内存 (这只需执行一次，请稍候)...")
        self.stock_data_dict = {}
        
        # 按股票分组处理
        for stock, group in tqdm(df.groupby('instrument'), desc="Pre-processing"):
            # 剥离 instrument 索引，只保留 datetime
            group = group.droplevel('instrument')
            
            # 【关键】：将这只股票的时间轴强制对齐到全局 dates
            # 这样所有的股票数组长度完全一致，天数索引(date_idx)可以直接通用！
            group = group.reindex(self.dates)
            
            # 提前在全局层面完成 fillna，消灭热循环中的计算
            if self.fillna_type == 'ffill+bfill':
                group = group.ffill().bfill()
                
            # 存为纯粹的高速 NumPy 数组 (将缺失的日期填为 0)
            self.stock_data_dict[stock] = group.fillna(0.0).values
            
        print("[Dataset] 数据预处理完成！")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 1. 获取目标日期和股票代码
        date, instrument = self.indices[idx]
        
        # 2. O(1) 获取当前日期在全局数组中的绝对索引
        date_idx = self.date_to_idx[date]
        
        # 3. O(1) 获取该股票的高速 NumPy 数组
        stock_array = self.stock_data_dict[instrument]
        
        # 4. 纯 NumPy 内存切片提取 20 天窗口 (耗时接近 0)
        if date_idx < self.step_len - 1:
            pad_len = self.step_len - 1 - date_idx
            sequence = stock_array[0 : date_idx + 1]
            sequence = np.pad(sequence, ((pad_len, 0), (0, 0)), mode='constant', constant_values=0.0)
        else:
            sequence = stock_array[date_idx - self.step_len + 1 : date_idx + 1]

        # 直接返回，没有任何 Pandas 实例化和运算！
        return sequence, (date, instrument)

    def get_index(self):
        return self.indices

# 【核心修改1】：更新批次采样器，让它一次性吐出 B 个交易日的全部股票索引
class DateGroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=16, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        indices = self.dataset.get_index()
        self.date_indices = {}
        for i, (date, _) in enumerate(indices):
            if date not in self.date_indices:
                self.date_indices[date] = []
            self.date_indices[date].append(i)

    def __iter__(self):
        dates = list(self.date_indices.keys())
        if self.shuffle:
            random.shuffle(dates)
            
        for i in range(0, len(dates), self.batch_size):
            batch_dates = dates[i:i+self.batch_size]
            batch_indices = []
            for date in batch_dates:
                batch_indices.extend(self.date_indices[date])
            yield batch_indices

    def __len__(self):
        return (len(self.date_indices) + self.batch_size - 1) // self.batch_size

# 【核心修改2】：动态补齐 (Dynamic Padding)，将不规则股票数补齐对齐为 Tensor
def custom_collate_fn(batch):
    grouped = defaultdict(list)
    for item in batch:
        dt = item[1][0]
        grouped[dt].append(item)
    
    sorted_dts = sorted(grouped.keys())
    # 找到这几天的最大股票数
    max_N = max(len(grouped[dt]) for dt in sorted_dts)
    B = len(sorted_dts)
    
    seq_len, feature_dim = batch[0][0].shape
    
    # 初始化补齐矩阵，全部默认填充 0.0
    padded_data = torch.zeros((B, max_N, seq_len, feature_dim), dtype=torch.float32)
    batch_indices = []
    
    for b, dt in enumerate(sorted_dts):
        day_items = grouped[dt]
        actual_N = len(day_items)
        
        # 将真实股票数据填入矩阵前部，后部保持补齐的 0
        day_data = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in day_items], dim=0)
        padded_data[b, :actual_N, :, :] = day_data
        
        # 收集每一天真实的索引标签，回测时要用
        batch_indices.append([item[1] for item in day_items])
        
    return padded_data, batch_indices

def init_data_loader(df, step_len, shuffle, start, end, select_feature=None, batch_size=16):
    if select_feature is not None:
        df = df[select_feature]

    dataset = TSDatasetH(df, step_len=step_len, start=start, end=end, fillna_type='ffill+bfill')
    # 传入 batch_size (天数)
    sampler = DateGroupedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    data_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=custom_collate_fn,  
        pin_memory=True,
    )
    return data_loader