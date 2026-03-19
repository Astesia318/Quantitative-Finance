import torch
import torch.nn as nn
from .basic_net import MLP

class PortfolioLayer(nn.Module):
    # 引入 M_portfolios 参数（原文通常设置为 20 或类似较小的值）
    def __init__(self, latent_size, M_portfolios=20, hidden_size=32):
        super(PortfolioLayer, self).__init__()
        
        self.net = nn.Linear(
            latent_size,M_portfolios
        )

    def forward(self, latent_features):
        # latent_features: (batch_size, stock_size, latent_size)
        out = self.net(latent_features) # out shape: (batch_size, stock_size, M_portfolios)
        
        # 在 stock 维度(dim=1)上做 softmax，确保每个 portfolio 内部的股票权重和为 1
        weights = torch.softmax(out, dim=1) 
        return weights

class FactorEncoder(nn.Module):
    def __init__(self, latent_size, stock_size, factor_size, M_portfolios=20, hidden_size=32):
        super(FactorEncoder, self).__init__()

        self.portfolio_layer = PortfolioLayer(latent_size, M_portfolios, hidden_size)
        
        # 【修复3】：MappingLayer 的输入不再是 stock_size，而是固定维度 M_portfolios
        self.mapping_layer = MappingLayer(M_portfolios, factor_size, hidden_size)

    def forward(self, latent_features, future_returns):
        # weights: (batch_size, stock_size, M_portfolios)
        portfolio_weights = self.portfolio_layer(latent_features)
        
        # 确保 future_returns 维度是 (batch_size, stock_size, 1)
        if len(future_returns.shape) == 2:
            future_returns = future_returns.unsqueeze(-1)

        # 【修复2】：完美实现原文公式(7)的加权求和
        # (bs, M_portfolios, stock_size) 矩阵乘 (bs, stock_size, 1) 
        # 结果为 (bs, M_portfolios, 1)，天然完成了沿着 stock_size 的加总求和
        portfolio_returns = torch.bmm(portfolio_weights.transpose(1, 2), future_returns)
        
        # 去掉最后的维度，变成 (batch_size, M_portfolios)
        portfolio_returns = portfolio_returns.squeeze(-1)

        # 此时送给 MappingLayer 的维度永远是 M_portfolios，彻底脱离了股票数量的束缚
        mu_post, sigma_post = self.mapping_layer(portfolio_returns)
        mu_post = mu_post.unsqueeze(-1)
        sigma_post = sigma_post.unsqueeze(-1)

        return mu_post, sigma_post

class MappingLayer(nn.Module):
    # 此处不需要大改，只需要在外面实例化时传入 input_size = M_portfolios 即可
    def __init__(self, input_size, factor_size, hidden_size=16):
        super(MappingLayer, self).__init__()

        self.mu_net = nn.Linear(input_size,factor_size)

        self.sigma_net = nn.Sequential(
            nn.Linear(input_size,factor_size),
            nn.Softplus()
        )

    def forward(self, portfolio_returns):
        mu_post = self.mu_net(portfolio_returns)
        sigma_post = self.sigma_net(portfolio_returns)
        return mu_post, sigma_post