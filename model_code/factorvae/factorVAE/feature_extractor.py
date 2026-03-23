import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        num_input,
        num_latent,
        seq_len
    ):
        """
        Generate latent features from historical sequential features.

        Args:
            num_input (int): Size of characteristic (特征数量，对应 158).
            num_latent (int): Size of latent features (隐变量维度).
            seq_len (int): T of data (时间窗口大小，对应 20).
        """
        super(FeatureExtractor, self).__init__()

        self.seq_len = seq_len
        self.num_input = num_input
        self.num_latent = num_latent

        # 线性映射层
        self.proj = nn.Sequential(
            nn.Linear(num_input, num_input),
            nn.LeakyReLU()
        )
        
        # 核心 GRU 层
        self.gru = nn.GRU(input_size=num_input, hidden_size=num_latent)

    def forward(self, x):
        """
        Generate latent features from historical sequential characteristics.

        Args:
            x (tensor): An array with the shape of (batch_size, seq_len, N, num_input)
                        注意这里的 N 是动态的股票数量

        Returns:
            torch.tensor: The latent features of stocks with the shape of (batch_size, N, num_latent)
        """
        # 1. 动态获取当前的 batch_size (B) 和 当前批次的股票数量 (N)
        B, T, N, D = x.shape

        # 断言只检查时间步长和特征维度，彻底放过对 N 的检查
        assert T == self.seq_len and D == self.num_input, f"input shape incorrect, expected (_, {self.seq_len}, _, {self.num_input}), got {x.shape}"

        # 2. 形状转换以适配 GRU 
        # (B, T, N, D) -> permute -> (T, B, N, D)
        # -> reshape -> (T, B * N, D)
        # 相当于把 Batch 和 股票维度合并，让 GRU 并行处理所有股票的时间序列
        x_reshaped = x.permute(1, 0, 2, 3).reshape(T, B * N, D)

        h_proj = self.proj(x_reshaped)

        # 3. 输入 GRU
        # out shape: (T, B * N, num_latent)
        # hidden shape: (1, B * N, num_latent)  # 1 是因为单向单层 GRU
        out, hidden = self.gru(h_proj)

        # 4. 动态切分还原形状
        # e is the latent features of stock
        # 将合并的 (B * N) 还原为 (B, N)
        e = hidden.view(B, N, self.num_latent)

        return e