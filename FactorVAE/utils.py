import torch
import numpy as np
import pandas as pd
import random
import os
from dataclasses import dataclass, field
from module import *
from tqdm import tqdm
from scipy.stats import spearmanr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
@dataclass
class DataArgument:
    save_dir: str = field(
        default='./data',
        metadata={"help": 'directory to save model'}
    )
    start_time: str = field(
        default="2010-12-01",
        metadata={"help": "start_time"}
    )
    end_time: str =field(
        default='2020-12-31', 
        metadata={"help": "end_time"}
    )

    fit_end_time: str= field(
        default="2017-12-31", 
        metadata={"help": "fit_end_time"}
    )

    val_start_time : str = field(
        default='2018-01-01', 
        metadata={"help": "val_start_time"}
    )

    val_end_time: str =field(default='2018-12-31')

    seq_len : int = field(default=20)

    normalize: bool = field(
        default=True,
    )
    select_feature: str = field(
        default=None,
    )



def load_model(args):
    feature_extractor = FeatureExtractor(num_latent = args.num_latent, hidden_size =args.hidden_size)

    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)

    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)
    return factorVAE

@torch.no_grad()
def generate_prediction_scores(model, test_dataloader, args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_indices = []

    from tqdm.auto import tqdm
    with tqdm(total=len(test_dataloader)) as pbar: 
        # 注意：这里解包出来的第二个参数是我们刚刚在 Dataset 里定制的 real_indices（真实的股票标识列表）
        for i, (char_with_label, real_indices) in enumerate(test_dataloader):
            char = char_with_label[:, :, :, :-1].to(device)
            
            # predictions 形状变为: (B, max_N, 1)
            predictions = model.prediction(char.float()) 
            
            # 【关键】：只切片保存真实的股票，过滤掉 Padding 产生的废数据
            for b in range(char.shape[0]):
                day_real_indices = real_indices[b]
                actual_N = len(day_real_indices)
                
                # 切除假股票：截取 [:actual_N]
                valid_preds = predictions[b, :actual_N, 0].detach().cpu()
                all_predictions.append(valid_preds)
                all_indices.extend(day_real_indices)
                
            pbar.update(1)
            
    ls = torch.cat(all_predictions, dim=0)
    # 利用收集回来的真实索引组装 DataFrame，保证 100% 对齐
    multi_index = pd.MultiIndex.from_tuples(all_indices, names=["datetime","instrument"])
    ls = pd.DataFrame(ls.numpy(), index=multi_index, columns=['score'])
    return ls

@dataclass
class test_args:
    run_name: str
    num_factor: int
    normalize: bool = True
    select_feature: bool = True
    
    batch_size: int = 300
    seq_length: int = 20

    hidden_size: int = 20
    num_latent: int = 20
    num_portfolio: int = 128

    save_dir='./best_model'
    use_qlib: bool = False
    gpu:int = 6
    

def RankIC(df, column1='LABEL0', column2='Pred'):
    ric_values_multiindex = []

    for date in df.index.get_level_values(0).unique():
        daily_data = df.loc[date].copy()
        daily_data['LABEL0_rank'] = daily_data[column1].rank()
        daily_data['pred_rank'] = daily_data[column2].rank()
        ric, _ = spearmanr(daily_data['LABEL0_rank'], daily_data['pred_rank'])
        ric_values_multiindex.append(ric)

    if not ric_values_multiindex:
        return np.nan, np.nan

    ric = np.mean(ric_values_multiindex)
    std = np.std(ric_values_multiindex)
    ir = ric / std if std != 0 else np.nan
    return pd.DataFrame({'RankIC': [ric], 'RankIC_IR': [ir]})
    
    