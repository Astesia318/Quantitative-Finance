import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_net import MLP


class FactorPredictor(nn.Module):
    def __init__(self, latent_size, factor_size,hidden_size):
        """
        :param latent_size: 对应公式中的特征维度 H
        :param factor_size: 对应公式中的独立头数 K (即因子数量)
        """
        super(FactorPredictor, self).__init__()

        self.latent_size = latent_size
        self.factor_size = factor_size  # 即论文中的 K
        self.hidden_size = hidden_size
        # 1. 对应公式中的全局查询向量 q ∈ R^H
        # 因为有 K 个独立头，所以我们定义 K 个独立的 q
        self.q = nn.Parameter(torch.randn(factor_size, latent_size))

        # 2. 对应公式中的 w_key 和 w_value
        # 为了高效计算，用一个 Linear 一次性计算 K 个头的 Key 和 Value
        self.W_key = nn.Linear(latent_size, factor_size * latent_size, bias=False)
        self.W_val = nn.Linear(latent_size, factor_size * latent_size, bias=False)

        # 3. 对应公式中的 distribution_network π_prior
        # 注意：输入维度不再是 stock_size * latent_size，而是 factor_size * latent_size
        self.distribution_network_mu = MLP(
            input_size=factor_size * latent_size,
            output_size=factor_size,
            hidden_size=self.hidden_size,
            out_activation=None
        )

        self.distribution_network_sigma = MLP(
            input_size=factor_size * latent_size,
            output_size=factor_size,
            hidden_size=64,
            out_activation=nn.Softplus()
        )

    def forward(self, latent_features):
        # latent_features: (batch_size, N, latent_size)  这里 N 是动态的当天股票数量
        bs, N, H = latent_features.shape
        K = self.factor_size

        # --- 公式: k^(i) = w_key * e^(i), v^(i) = w_value * e^(i) ---
        # 形状转换: (bs, N, K * H) -> (bs, N, K, H)
        K_mat = self.W_key(latent_features).view(bs, N, K, H)
        V_mat = self.W_val(latent_features).view(bs, N, K, H)

        # --- 公式: a_att 的分子部分 (计算 Cosine 相似度) ---
        # 1. 对 q 和 k 进行 L2 范数归一化 (对应公式中的 ||q||_2 和 ||k^(i)||_2)
        q_norm = F.normalize(self.q, p=2, dim=-1)         # (K, H)
        k_norm = F.normalize(K_mat, p=2, dim=-1)          # (bs, N, K, H)
        
        # 2. 计算点积 (等价于计算归一化后的 Cosine 相似度: q * k^T / (||q||*||k||))
        # 沿 H 维度相乘并求和
        cos_sim = torch.sum(k_norm * q_norm.view(1, 1, K, H), dim=-1)  # (bs, N, K)

        # --- 公式: a_att = max(0, cos_sim) / sum(max(0, cos_sim)) ---
        # 1. 应用 max(0, x)，即 ReLU
        scores = F.relu(cos_sim)  # (bs, N, K)
        
        # 2. 沿股票维度 N (dim=1) 归一化
        # 加 1e-8 是工程 Trick，防止某一天所有股票的相似度均为 0 导致除以 0 报错
        attn_weights = scores / (torch.sum(scores, dim=1, keepdim=True) + 1e-8)  # (bs, N, K)

        # --- 公式: h_att = \sum_{i=1}^N a_att^(i) * v^(i) ---
        # attn_weights 扩展为 (bs, N, K, 1)，以便与 V_mat (bs, N, K, H) 广播相乘
        # 然后在股票维度 N (dim=1) 上求和，彻底抹平股票数量维度的影响！
        h_att = torch.sum(attn_weights.unsqueeze(-1) * V_mat, dim=1)  # 结果形状: (bs, K, H)

        # --- 公式: h_multi = Concat([φ_att1, ..., φ_attK]) ---
        # 将 K 个头的特征展平拼接在一起
        h_multi = h_att.view(bs, -1)  # 结果形状: (bs, K * H)

        # --- 公式: [μ_prior, σ_prior] = π_prior(h_multi) ---
        mu_prior = self.distribution_network_mu(h_multi).unsqueeze(-1)       # (bs, factor_size, 1)
        sigma_prior = self.distribution_network_sigma(h_multi).unsqueeze(-1) # (bs, factor_size, 1)

        return mu_prior, sigma_prior