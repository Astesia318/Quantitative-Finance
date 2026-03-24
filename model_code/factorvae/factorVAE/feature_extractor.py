import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        num_input,
        hidden_size,
        seq_len
    ):
        """
        Generate latent features from historical sequential features.

        Args:
            num_input (int): Size of characteristic (特征数量，对应 158).
            hidden_size (int): Size of latent features (隐变量维度).
            seq_len (int): T of data (时间窗口大小，对应 20).
        """
        super(FeatureExtractor, self).__init__()

        self.seq_len = seq_len
        self.num_input = num_input
        self.hidden_size = hidden_size

        # 线性映射层
        self.proj = nn.Sequential(
            nn.Linear(num_input, num_input),
            nn.LeakyReLU()
        )
        self.normalize = nn.LayerNorm(num_input)
        # 核心 GRU 层
        self.gru = nn.GRU(input_size=num_input, hidden_size=hidden_size,batch_first=True)

    def forward(self, x:torch.Tensor):
        """
        Generate latent features from historical sequential characteristics.

        Args:
            x (tensor): An array with the shape of (batch_size, seq_len, N, num_input)
                        注意这里的 N 是动态的股票数量

        Returns:
            torch.tensor: The latent features of stocks with the shape of (batch_size, N, hidden_size)
        """
        # 1. 动态获取当前的 batch_size (B) 和 当前批次的股票数量 (N)
        B, T, N, D = x.shape

        # 断言只检查时间步长和特征维度，彻底放过对 N 的检查
        assert T == self.seq_len and D == self.num_input, f"input shape incorrect, expected (_, {self.seq_len}, _, {self.num_input}), got {x.shape}"

        # 2. 形状转换以适配 GRU 
        # (B, T, N, D) -> permute -> (T, B, N, D)
        # -> reshape -> (T, B * N, D) -> (B*N,T,D)
        # 相当于把 Batch 和 股票维度合并，让 GRU 并行处理所有股票的时间序列
        x_reshaped = x.permute(1, 0, 2, 3).reshape(T, B * N, D).permute(1,0,2)
        x_normal=self.normalize(x_reshaped)
        h_proj = self.proj(x_normal)

        # 3. 输入 GRU
        # out shape: (T, B * N, hidden_size)
        # hidden shape: (1, B * N, hidden_size)  # 1 是因为单向单层 GRU
        out, hidden = self.gru(h_proj)

        # 4. 动态切分还原形状
        # e is the latent features of stock
        # 将合并的 (B * N) 还原为 (B, N)
        e = hidden.view(B, N, self.hidden_size)

        return e