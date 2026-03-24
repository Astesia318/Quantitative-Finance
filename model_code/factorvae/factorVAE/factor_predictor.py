import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        
        # Query 向量 (H,)
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, stock_latent):
        # stock_latent shape: (bs, N, H) 
        # bs 是 batch_size，N 是股票数量，H 是 hidden_size
        bs, N, H = stock_latent.shape

        self.key = self.key_layer(stock_latent)    # (bs, N, H)
        self.value = self.value_layer(stock_latent) # (bs, N, H)
        
        # calculate attention weights (兼容 batch 维度的点积)
        # key: (bs, N, H) 与 query: (H,) 做点积 -> 结果 (bs, N)
        attention_weights = torch.matmul(self.key, self.query) 
        
        # scaling (缩放因子)
        attention_weights = attention_weights / torch.sqrt(torch.tensor(H, dtype=torch.float32) + 1e-6)
        
        attention_weights = self.dropout(attention_weights)
        attention_weights = F.relu(attention_weights) # max(0, x)
        
        # softmax 必须在股票维度 N (dim=1) 上进行
        attention_weights = F.softmax(attention_weights, dim=1) # (bs, N)
        
        # calculate context vector (加权求和)
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            return torch.zeros(bs, H, device=stock_latent.device)
        else:
            # attention_weights: (bs, N) -> unsqueeze -> (bs, 1, N)
            # value: (bs, N, H)
            # bmm 矩阵相乘 -> (bs, 1, H) -> squeeze -> (bs, H)
            context_vector = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1) 
            return context_vector 


class FactorPredictor(nn.Module):
    def __init__(self, hidden_size, num_factor):
        super(FactorPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_factor = num_factor
        
        # 使用 ModuleList 生成多个独立的 AttentionLayer
        self.attention_layers = nn.ModuleList([
            AttentionLayer(self.hidden_size) for _ in range(num_factor)
        ])
        
        # 输出层映射
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        
        # 计算 mu 和 sigma
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent):
        # stock_latent: (bs, N, H)
        
        h_multi_list = []
        for i in range(self.num_factor):
            # 将 latent 输入到每个独立的头中，返回 context_vector: (bs, H)
            attention_layer_out = self.attention_layers[i](stock_latent)
            h_multi_list.append(attention_layer_out)
            
        # 沿着 factor 维度 (dim=1) 堆叠，形状变为: (bs, num_factor, H)
        h_multi = torch.stack(h_multi_list, dim=1)

        # 映射层网络操作
        # (bs, num_factor, H) -> Linear -> (bs, num_factor, H)
        h_multi = self.linear(h_multi)
        h_multi = self.leakyrelu(h_multi)
        
        # 获取预测值
        # (bs, num_factor, H) -> Linear -> (bs, num_factor, 1)
        pred_mu = self.mu_layer(h_multi)
        pred_sigma = self.sigma_layer(h_multi)
        pred_sigma = self.softplus(pred_sigma)
        
        # 原版模型期望的形状通常是不带最后一个维度 1 的
        # 所以最终形状需要转成 (bs, num_factor, 1) 或者保持不变以兼容主框架
        # 因为在 VAE 预测器输出通常要求 shape 为 (bs, num_factor, 1)
        return pred_mu, pred_sigma