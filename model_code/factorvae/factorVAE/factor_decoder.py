import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_net import MLP
from torch.distributions import Normal


class FactorDecoder(nn.Module):
    def __init__(
        self, latent_size, factor_size, stock_size, hidden_size=64
    ):
        """Generate Stock return y hat from factors ang latent features.

        Args:
            latent_size (int)
            factor_size (int)
            stock_size (int)
            alpha_h_size (int): The size of the hidden layer in alpha layer. Defaults to 64.
            hidden_size (int or list): Defaults to 64.
        """
        super(FactorDecoder, self).__init__()
        self.alpha_layer = AlphaLayer(
            latent_size=latent_size,
            hidden_size=hidden_size,
            stock_size=stock_size,
        )
        self.beta_layer = BetaLayer(
            latent_size=latent_size,
            stock_size=stock_size,
            factor_size=factor_size,
            hidden_size=hidden_size
        )

    def forward(self, factors, latent_features):
        mu_alpha, sigma_alpha = self.alpha_layer(latent_features)
        m_alpha = Normal(mu_alpha, sigma_alpha)
        alpha = m_alpha.sample()

        beta = self.beta_layer(latent_features)

        exposed_factors = torch.bmm(beta, factors)

        stock_returns = exposed_factors + alpha

        return stock_returns, mu_alpha, sigma_alpha, beta




class AlphaLayer(nn.Module):
    def __init__(self, latent_size, hidden_size, stock_size):
        super(AlphaLayer, self).__init__()
        # 注: stock_size 在这里其实用不到，保留是为了接口兼容
        
        # 1. 严格对应公式: h_alpha^(i) = LeakyReLU(w_alpha * e^(i) + b_alpha)
        # 这里只有一层 nn.Linear，对应 w_alpha 和 b_alpha
        self.hidden_layer = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.LeakyReLU()
        )
        
        # 2. 严格对应公式: mu_alpha^(i) = w_alpha_mu * h_alpha^(i) + b_alpha_mu
        # 纯线性映射，没有任何隐藏层，也没有激活函数
        self.mu_alpha_layer = nn.Linear(hidden_size, 1)

        # 3. 严格对应公式: sigma_alpha^(i) = Softplus(w_alpha_sigma * h_alpha^(i) + b_alpha_sigma)
        # 一层线性映射后直接接 Softplus
        self.sigma_alpha_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )

    def forward(self, latent_features):
        # latent_features: (batch_size, stock_size, latent_size)
        h_alpha = self.hidden_layer(latent_features)
        
        mu_alpha = self.mu_alpha_layer(h_alpha)
        sigma_alpha = self.sigma_alpha_layer(h_alpha)
        
        return mu_alpha, sigma_alpha


class BetaLayer(nn.Module):
    def __init__(self, latent_size, stock_size, factor_size, hidden_size=64):
        super(BetaLayer, self).__init__()

        self.factor_size = factor_size
        self.stock_size = stock_size

        self.beta_layer = nn.Linear(latent_size,factor_size)

    def forward(self, latent_features):
        # (bs, stock_size, latent_size) -> (bs, stock_size, factor_size)
        beta = self.beta_layer(latent_features)

        return beta
