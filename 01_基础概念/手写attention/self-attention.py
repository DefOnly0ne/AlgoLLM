#流程1.输入矩阵：输入序列X为R^(n*d)，其中n为序列长度，d为特征维度
#线性变换生存Q、K、V：Q=XW_Q，K=XW_K，V=XW_V，其中W_Q、W_K、W_V为可学习的权重矩阵，
# WQ、WK、WV的维度分别为R^(d*d_k)、R^(d*d_k)、R^(d*d_v)，dk和dv为Q、K、V的特征维度，通常取d_k=d_v=d/h，h为注意力头数
#计算注意力得分：注意力得分矩阵S通过点积计算得到，S=QK^T/sqrt(d_k)，其中sqrt(d_k)为缩放因子，防止点积值过大
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model,d_k):
        super().__init__()
        self.d_k = d_k
        self.w_q = nn.Linear(d_model, d_k)
        self.w_k = nn.Linear(d_model, d_k)
        self.w_v = nn.Linear(d_model, d_k)

    def forward(self, x):
        Q = self.w_q(x)  # 线性变换生成Q,[batch_size, seq_len, d_k]
        K = self.w_k(x)  # 线性变换生成K
        V = self.w_v(x)  # 线性变换生成V
        scores = torch.matmul(Q,K.transpose(-1,-2)) / torch.sqrt(torch.tensor(self.d_k))
        atten_weights = torch.softmax(scores, dim=-1)  # 计算注意力权重
        output = torch.matmul(atten_weights, V)  # 加权求和得到输出
        return output