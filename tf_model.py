import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 位置编码,将关于位置的信息注入到输入嵌入。多频率周期信号唯一标识位置，同时保留了数学可组合性
# d_model：输入嵌入的维度
# max_len：输入序列的最大长度
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 计算位置编码
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # torch.Size([5000, 1]) 将形状从 [max_len] 变为 [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # 不同维度上正弦波的频率。1/10000^(2i/d_model)。torch.arange(0, d_model, 2) 创建偶数序列。
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数位置使用余弦函数
        pe = pe.unsqueeze(0) # 添加一个维度，变成 [1, max_len, d_model]
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + self.pe[:, :x.size(1)]
        return x

d_model = 512
max_len = 100
num_heads = 8

positional_encoding = PositionalEncoding(d_model,max_len)

print(positional_encoding.pe.shape) # torch.Size([1, 100, 512])

input_sequence = torch.randn(5,max_len,d_model) # torch.Size([5, 100, 512])

output = positional_encoding(input_sequence)
print(output.shape) # torch.Size([5, 100, 512])