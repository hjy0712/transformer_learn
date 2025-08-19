import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

## 1. 位置编码,将关于位置的信息注入到输入嵌入(相加)。多频率周期信号唯一标识位置，同时保留了数学可组合性
# d_model：输入嵌入的维度
# max_len：输入序列的最大长度
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 计算位置编码
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # torch.Size([5000, 1]) 将形状从 [max_len] 变为 [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # 不同维度上正弦波的频率。e^(-2i * ln(10000) / d_model) = 1/10000^(2i/d_model)。torch.arange(0, d_model, 2) 创建偶数序列。
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数位置使用余弦函数
        pe = pe.unsqueeze(0) # 添加一个维度，变成 [1, max_len, d_model]
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + self.pe[:, :x.size(1)]
        return x

# ------------------测试位置编码-------------------
# d_model = 512
# max_len = 100
# num_heads = 8

# positional_encoding = PositionalEncoding(d_model,max_len)

# print(positional_encoding.pe.shape) # torch.Size([1, 100, 512])

# input_sequence = torch.randn(5,max_len,d_model) # torch.Size([5, 100, 512])

# output = positional_encoding(input_sequence)
# print(output.shape) # torch.Size([5, 100, 512])

# ------------------------------------------------

## 2. 多头注意力机制
# d_model: 输入嵌入的维度
# num_heads: 注意力头的数量
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        # 查询、键和值的线性投影
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        # 输出线性投影  
        self.out_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        # 重塑维度
        # 输入 x: [batch_size, seq_len, d_model]
        # 输出: [batch_size, num_heads, seq_len, head_dim]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1,2) # 交换维度1和2，从 [batch, seq_len, num_heads, head_dim] 变为 [batch, num_heads, seq_len, head_dim]
    
    def forward(self, query, key, value, mask=None):
        # 线性投影
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value) # [batch_size, seq_len, d_model] torch.Size([5, 100, 512])

        # 拆分头
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value) # [batch_size, num_heads, seq_len, head_dim] torch.Size([5, 8, 100, 64])  相当于把特征维度512拆成8*64

        # 计算注意力分数。矩阵乘法规则：保留前面的维度，最后两个维度进行矩阵乘法
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(self.head_dim) # [batch_size, num_heads, seq_len, seq_len] 如果不是多头注意力，维度是[batch_size, seq_len, seq_len]

        # 应用掩码。填充掩码和因果掩码
        if mask is not None:
            scores += scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1) # [batch_size, num_heads, seq_len, seq_len]

        # 加权求和
        attention_output = torch.matmul(attention_weights, value) # [batch_size, num_heads, seq_len, head_dim]

        # 合并头
        batch_size, _, seq_length, _ = attention_output.size()
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model) # contiguous确保内存连续 

        # 输出线性投影
        attention_output = self.out_linear(attention_output) # [batch_size, seq_length, d_model]  nn.Linear只对最后一个维度进行线性变换
        return attention_output

# ------------------测试多注意力头-------------------
# d_model = 512
# max_len = 100
# num_heads = 8

# multi_head_attention = MultiHeadAttention(d_model,num_heads)

# input_sequence = torch.randn(5,max_len,d_model) # torch.Size([5, 100, 512])
# attention_output = multi_head_attention(input_sequence,input_sequence,input_sequence)
# print(attention_output.shape) # torch.Size([5, 100, 512])
# ------------------------------------------------

## 3. 前馈网络。处理信息和从输入序列中提取特征
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# ------------------测试前馈网络 + 多头注意力网络-------------------
# d_model = 512
# max_len = 100
# num_heads = 8
# d_ff = 2048

# multihead_attn = MultiHeadAttention(d_model, num_heads)

# ff_network = FeedForward(d_model, d_ff)

# input_sequence = torch.randn(5, max_len, d_model)

# attn_output = multihead_attn(input_sequence, input_sequence, input_sequence)
# ff_output = ff_network(attn_output)

# print(ff_output.shape) # torch.Size([5, 100, 512])
# ------------------------------------------------

## 4. 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)  # LayerNorm 的计算公式：y = γ * (x - μ) / σ + β。γ和β是可学习的参数，μ和σ是均值和标准差。
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) # 训练时：随机丢弃一些神经元。测试时：使用所有神经元，但缩放输出。使得模型不能过度依赖某些特定的神经元，从而学习更鲁棒的特征表示

    def forward(self, x, mask):

        # 自注意力层
        attention_output = self.self_attention(x, x, x, mask)
        attention_output = self.dropout(attention_output)
        x = x + attention_output
        x = self.norm1(x)

        # 前馈网络层
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = x + ff_output
        x = self.norm2(x)
        return x

# ------------------测试编码器层-------------------
# d_model = 512
# max_len = 100
# num_heads = 8
# d_ff = 2048

# encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

# input_sequence = torch.randn(5, max_len, d_model)

# encoder_output = encoder_layer(input_sequence, None)
# print(encoder_output.shape) # torch.Size([5, 100, 512])
# ------------------------------------------------

## 5. 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        # 掩码的自注意力层
        self_attention_output = self.masked_self_attention(x, x, x, tgt_mask)
        self_attention_output = self.dropout(self_attention_output)
        x = x + self_attention_output
        x = self.norm1(x)

        # 编码器-解码器注意力层
        enc_dec_attention_output = self.enc_dec_attention(x, encoder_output, encoder_output, src_mask)
        enc_dec_attention_output = self.dropout(enc_dec_attention_output)
        x = x + enc_dec_attention_output
        x = self.norm2(x)

        # 前馈网络层
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout(feed_forward_output)
        x = x + feed_forward_output
        x = self.norm3(x)

        return x

# ------------------测试解码器层-------------------
# d_model = 512
# max_len = 100
# num_heads = 8
# d_ff = 2048
# dropout = 0.1

# decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

# src_mask = torch.randn(1, max_len, max_len) > 0.5
# tgt_mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0) == 0 # 因果掩码，下三角矩阵之后在第一维添加维度

# input_sequence = torch.randn(1, max_len, d_model)
# encoder_output = torch.randn(1, max_len, d_model)

# decoder_output = decoder_layer(input_sequence, encoder_output, src_mask, tgt_mask)
# print(decoder_output.shape) # torch.Size([5, 100, 512])
# ------------------------------------------------

## 6. 完整transformer模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1):
        super(Transformer, self).__init__()

        # 定义编码器和解码器的词嵌入层。相当于查找表，将词索引数字换为词嵌入
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model) # 词汇表大小 词嵌入维度
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 定义位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # 定义编码器和解码器
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # 定义输出层
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt): # 不关注序列中的pad部分，使不想要的地方softmax值为0
        # 源端掩码（padding mask） [batch_size, 1, 1, src_len] 方便在 注意力矩阵 [batch_size, heads, query_len, key_len] 上进行广播
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # 目标端掩码（padding mask） [batch_size, 1, tgt_len, 1] 方便在 decoder的key/value [batch_size, heads, tgt_len, tgt_len]上进行广播
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        # 目标端 no-peak 掩码（未来信息屏蔽）
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool() # 右下三角矩阵，不能看到未来信息

        # 综合目标端掩码
        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask


    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # 解码器输入预处理：输入的词转换为词嵌入，并进行位置编码
        encoder_embedding = self.encoder_embedding(src)
        encoder_embedding = self.positional_encoding(encoder_embedding)
        src_embedding = self.dropout(encoder_embedding)

        # 编码器输入预处理：输入的词转换为词嵌入，并进行位置编码
        decoder_embedding = self.decoder_embedding(tgt)
        decoder_embedding = self.positional_encoding(decoder_embedding)
        tgt_embedding = self.dropout(decoder_embedding)

        # 输入到编码器
        encoder_output = src_embedding
        for enc_layer in self.encoder_layers:
            encoder_output = enc_layer(encoder_output, src_mask)

        # 输入到解码器
        decoder_output = tgt_embedding
        for dec_layer in self.decoder_layers:
            decoder_output = dec_layer(decoder_output, encoder_output, src_mask, tgt_mask)

        # 输出层
        output = self.output_linear(decoder_output)
        return output

# ------------------测试 整体transformer-------------------
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
max_len = 100
num_heads = 8
num_layers = 6
d_ff = 2048
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)

src_data = torch.randint(1, src_vocab_size, (5, max_len)) # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (5, max_len))

tf_out = transformer(src_data, tgt_data[:, :])
print(tf_out.shape) # torch.Size([5, 99, 5000])
# ------------------------------------------------
