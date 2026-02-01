#交叉注意力用于跨序列交互（例如在Transformer中的编码器-解码器注意力），Q来自一个序列，K和V来自另一个序列
#公式：cross-attention(Q_dec, K_enc, V_enc) = softmax(Q_dec K_enc^T / sqrt(d_k)) V_enc
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.w_q = nn.Linear(d_model, d_k)
        self.w_k = nn.Linear(d_model, d_k)
        self.w_v = nn.Linear(d_model, d_k)

    def forward(self, x_dec, x_enc):
        Q = self.w_q(x_dec)  # 线性变换生成Q,[batch_size, seq_len_dec, d_k]
        K = self.w_k(x_enc)  # 线性变换生成K,[batch_size, seq_len_enc, d_k]
        V = self.w_v(x_enc)  # 线性变换生成V,[batch_size, seq_len_enc, d_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d_k))
        atten_weights = torch.softmax(scores, dim=-1)  # 计算注意力权重
        output = torch.matmul(atten_weights, V)  # 加权求和得到输出
        return output