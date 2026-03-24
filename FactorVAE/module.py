import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, num_latent, hidden_size, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.num_latent = num_latent
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.normalize = nn.LayerNorm(num_latent)
        self.linear = nn.Linear(num_latent, num_latent)
        self.leakyrelu = nn.LeakyReLU()
        self.gru = nn.GRU(num_latent, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        #! x: (B, N, seq_length, num_latent)
        B, N, T, F_dim = x.shape
        
        # 折叠 B 和 N 维度，让 GRU 并行处理所有天数的所有股票
        x = x.view(B * N, T, F_dim)
        
        x = self.normalize(x)
        out = self.linear(x)
        out = self.leakyrelu(out)
        
        stock_latent, _ = self.gru(out)
        stock_latent = stock_latent[:, -1, :] # 取最后一步 (B*N, hidden_size)
        
        # 重新展开为 (B, N, hidden_size)
        return stock_latent.view(B, N, self.hidden_size)

class FactorEncoder(nn.Module):
    def __init__(self, num_factors, num_portfolio, hidden_size):
        super(FactorEncoder, self).__init__()
        self.num_factors = num_factors
        
        self.portfolio_layer = nn.Linear(hidden_size, num_portfolio)
        self.linear_mu = nn.Linear(num_portfolio, num_factors)
        self.linear_sigma = nn.Linear(num_portfolio, num_factors)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent, returns):
        #! stock_latent: (B, N, hidden_size)
        #! returns: (B, N, 1)
        
        # (B, N, num_portfolio)
        weights = self.portfolio_layer(stock_latent)
        # 沿股票维度 N (dim=1) 进行 Softmax
        weights = F.softmax(weights, dim=1)
        
        # 使用批量矩阵乘法 bmm: (B, num_portfolio, N) @ (B, N, 1) -> (B, num_portfolio, 1)
        portfolio_returns = torch.bmm(weights.transpose(1, 2), returns)
        portfolio_returns = portfolio_returns.squeeze(-1) # (B, num_portfolio)

        mu = self.linear_mu(portfolio_returns)       # (B, num_factors)
        sigma = self.linear_sigma(portfolio_returns) # (B, num_factors)
        sigma = self.softplus(sigma)
        
        return mu.unsqueeze(-1), sigma.unsqueeze(-1) # (B, num_factors, 1)


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, stock_latent):
        # stock_latent: (B, N, hidden_size)
        B, N, H = stock_latent.shape

        self.key = self.key_layer(stock_latent)    # (B, N, H)
        self.value = self.value_layer(stock_latent) # (B, N, H)
        
        # (B, N, H) @ (H,) -> (B, N)
        attention_weights = torch.matmul(self.key, self.query)
        attention_weights = attention_weights / torch.sqrt(torch.tensor(H, dtype=torch.float32) + 1e-6)
        attention_weights = self.dropout(attention_weights)
        attention_weights = F.relu(attention_weights)
        attention_weights = F.softmax(attention_weights, dim=1) # 在股票维度归一化
        
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            return torch.zeros(B, H, device=stock_latent.device)
        else:
            # (B, 1, N) @ (B, N, H) -> (B, 1, H) -> (B, H)
            context_vector = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1)
            return context_vector 

class FactorPredictor(nn.Module):
    def __init__(self, hidden_size, num_factor):
        super(FactorPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_factor = num_factor
        self.attention_layers = nn.ModuleList([AttentionLayer(self.hidden_size) for _ in range(num_factor)])
        
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent):
        #! stock_latent: (B, N, H)
        h_multi_list = []
        for i in range(self.num_factor):
            attention_layer_out = self.attention_layers[i](stock_latent) # (B, H)
            h_multi_list.append(attention_layer_out)
            
        h_multi = torch.stack(h_multi_list, dim=1) # (B, num_factor, H)

        h_multi = self.linear(h_multi)
        h_multi = self.leakyrelu(h_multi)
        pred_mu = self.mu_layer(h_multi)       # (B, num_factor, 1)
        pred_sigma = self.sigma_layer(h_multi) # (B, num_factor, 1)
        pred_sigma = self.softplus(pred_sigma)
        
        return pred_mu, pred_sigma

class AlphaLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AlphaLayer, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, stock_latent):
        # (B, N, hidden_size) -> (B, N, 1)
        return self.linear(stock_latent)

class BetaLayer(nn.Module):
    def __init__(self, hidden_size, num_factor):
        super(BetaLayer, self).__init__()
        self.linear = nn.Linear(hidden_size, num_factor)

    def forward(self, stock_latent):
        # (B, N, hidden_size) -> (B, N, num_factor)
        return self.linear(stock_latent)

class FactorDecoder(nn.Module):
    def __init__(self, alpha_layer, beta_layer):
        super(FactorDecoder, self).__init__()
        self.alpha_layer = alpha_layer
        self.beta_layer = beta_layer

    def forward(self, stock_latent, factor_mu, factor_sigma):
        #! stock_latent: (B, N, hidden_size)
        #! factor_mu: (B, num_factor, 1)
        alpha = self.alpha_layer(stock_latent) # (B, N, 1)
        beta = self.beta_layer(stock_latent)   # (B, N, num_factor)

        # (B, N, num_factor) @ (B, num_factor, 1) -> (B, N, 1)
        y_pred = alpha + torch.bmm(beta, factor_mu)
        return y_pred

class FactorVAE(nn.Module):
    def __init__(self, feature_extractor, factor_encoder, factor_decoder, factor_predictor):
        super(FactorVAE, self).__init__()
        self.feature_extractor = feature_extractor
        self.factor_encoder = factor_encoder
        self.factor_decoder = factor_decoder
        self.factor_predictor = factor_predictor

    def KL_Divergence(self, mu1, sigma1, mu2, sigma2):
        # mu1, mu2: (B, num_factor, 1)
        kl = torch.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
        # 沿因子维度求和，然后沿 Batch 维度求均值
        return kl.sum(dim=1).mean() 

    def forward(self, x, returns):
        #! x: (B, N, seq_length, num_latent)
        #! returns: (B, N, 1)
        stock_latent = self.feature_extractor(x)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns)
        reconstruction = self.factor_decoder(stock_latent, factor_mu, factor_sigma)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)

        reconstruction_loss = F.mse_loss(reconstruction, returns)
        
        if torch.any(pred_sigma == 0):
            pred_sigma[pred_sigma == 0] = 1e-6
        kl_divergence = self.KL_Divergence(factor_mu, factor_sigma, pred_mu, pred_sigma)

        vae_loss = reconstruction_loss + kl_divergence
        return vae_loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma

    def prediction(self, x):
        stock_latent = self.feature_extractor(x)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        y_pred = self.factor_decoder(stock_latent, pred_mu, pred_sigma)
        return y_pred