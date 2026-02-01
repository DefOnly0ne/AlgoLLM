#多头注意力通过并行多个注意力头，捕捉不同子空间的特征
#Multihead_attn(Q,K,V)=Concat(head_1,head_2,...,head_h)W_O
#其中head_i=Attention(QW_Qi,KW_Ki,VW_Vi)，h为注意力头数，W_Qi、W_Ki、W_Vi为第i个头的线性变换矩阵，W_O为输出线性变换矩
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size,seq_len,_ = x.size()
        #拆分多头
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)  # [batch_size, num_heads, seq
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        #计算注意力得分
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d_k))
        atten_weights = torch.softmax(scores, dim=-1)  # 计算注意力权重
        output = torch.matmul(atten_weights, V)
        #合并多头
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return self.w_o(output)  # 输出线性变换