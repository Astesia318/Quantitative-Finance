import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
from base_model import SequenceModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[-2], :]

class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.ModuleList([Dropout(p=dropout) for _ in range(nhead)])
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x, mask=None):
        B, N, T, D = x.shape
        x_norm = self.norm1(x)
        
        # 维度转换: (B, N, T, D) -> (B, T, N, D) -> (B*T, N, D)
        q = self.qtrans(x_norm).permute(0, 2, 1, 3).reshape(B*T, N, D)
        k = self.ktrans(x_norm).permute(0, 2, 1, 3).reshape(B*T, N, D)
        v = self.vtrans(x_norm).permute(0, 2, 1, 3).reshape(B*T, N, D)

        dim = int(self.d_model/self.nhead)
        att_output = []
        
        # 处理 Mask 掩码: 屏蔽无效股票，防止其被关联
        if mask is not None:
            mask_bt = mask.unsqueeze(1).repeat(1, T, 1).reshape(B*T, N)
            mask_score = (~mask_bt).unsqueeze(1) # shape: (B*T, 1, N)

        for i in range(self.nhead):
            qh = q[:, :, i*dim:(i+1)*dim] if i < self.nhead-1 else q[:, :, i*dim:]
            kh = k[:, :, i*dim:(i+1)*dim] if i < self.nhead-1 else k[:, :, i*dim:]
            vh = v[:, :, i*dim:(i+1)*dim] if i < self.nhead-1 else v[:, :, i*dim:]

            scores = torch.matmul(qh, kh.transpose(1, 2)) / self.temperature # (B*T, N, N)
            
            # 填入负无穷，让 Softmax 后分配的权重为 0
            if mask is not None:
                scores = scores.masked_fill(mask_score, float('-inf'))
            
            atten_ave_matrixh = torch.softmax(scores, dim=-1)
            
            # 防御性代码: 避免整行都是 -inf 导致的 NaN
            if mask is not None:
                atten_ave_matrixh = atten_ave_matrixh.masked_fill(torch.isnan(atten_ave_matrixh), 0.0)

            atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            out_h = torch.matmul(atten_ave_matrixh, vh)
            att_output.append(out_h)
            
        att_output = torch.cat(att_output, dim=-1) # (B*T, N, D)
        att_output = att_output.view(B, T, N, D).permute(0, 2, 1, 3) # 回复 (B, N, T, D)

        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)
        return att_output

class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.ModuleList([Dropout(p=dropout) for _ in range(nhead)]) if dropout > 0 else None
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        B, N, T, D = x.shape
        x_flat = x.view(B*N, T, D) # 展平进行自身时序 Attention 
        x_norm = self.norm1(x_flat)
        
        q = self.qtrans(x_norm)
        k = self.ktrans(x_norm)
        v = self.vtrans(x_norm)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            qh = q[:, :, i*dim:(i+1)*dim] if i < self.nhead-1 else q[:, :, i*dim:]
            kh = k[:, :, i*dim:(i+1)*dim] if i < self.nhead-1 else k[:, :, i*dim:]
            vh = v[:, :, i*dim:(i+1)*dim] if i < self.nhead-1 else v[:, :, i*dim:]
            
            scores = torch.matmul(qh, kh.transpose(1, 2)) / math.sqrt(dim)
            atten_ave_matrixh = torch.softmax(scores, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
            
        att_output = torch.cat(att_output, dim=-1)

        xt = x_flat + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output.view(B, N, T, D)

class Gate(nn.Module):
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output * output

class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)
        query = h[:, :, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1) 
        lam = torch.softmax(lam, dim=-1).unsqueeze(2) 
        output = torch.matmul(lam, z).squeeze(2)
        return output

class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(MASTER, self).__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.linear = nn.Linear(d_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.t_attn = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        self.s_attn = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        self.temp_attn = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        src = x[:, :, :, :self.gate_input_start_index]
        gate_input = x[:, :, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=2)
        
        out = self.linear(src)
        out = self.pos_enc(out)
        out = self.t_attn(out)
        out = self.s_attn(out, mask=mask)  # 将掩码传入以防止非法跨股票通信
        out = self.temp_attn(out)
        out = self.decoder(out).squeeze(-1)
        return out

class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
            T_dropout_rate, S_dropout_rate, beta, batch_size=4, **kwargs, # 新增接收 batch_size
    ):
        super(MASTERModel, self).__init__(batch_size=batch_size, **kwargs)
        self.d_model = d_model
        self.d_feat = d_feat
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta
        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                            T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                            gate_input_start_index=self.gate_input_start_index,
                            gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        super(MASTERModel, self).init_model()