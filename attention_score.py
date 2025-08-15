import torch
import torch.nn.functional as F

# 示例输入序列
input_sequence = torch.tensor([[0.1, 0.2, 0.3, 0.1], [0.7, 0.8, 0.9, 0.1]]) # torch.Size([2, 4])

# 生成 Key、Query 和 Value 矩阵的随机权重
random_weights_key = torch.randn(input_sequence.size(-1), input_sequence.size(-1)-1) # torch.Size([4, 3])
random_weights_query = torch.randn(input_sequence.size(-1), input_sequence.size(-1)-1) # torch.Size([4, 3])
random_weights_value = torch.randn(input_sequence.size(-1), input_sequence.size(-1)-1) # torch.Size([4, 3])

# 计算 Key、Query 和 Value 矩阵
key = torch.matmul(input_sequence, random_weights_key) # torch.Size([2, 3])
query = torch.matmul(input_sequence, random_weights_query) # torch.Size([2, 3])
value = torch.matmul(input_sequence, random_weights_value) # torch.Size([2, 3])

attention_scores = torch.matmul(query, key.T) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32)) # torch.Size([2, 2])
print(attention_scores)

# 计算softmax。分母是指数加权和。对最后一个维度（每一行）进行归一化，每行之和为1
attention_weights = F.softmax(attention_scores, dim=-1)
print(attention_weights)

output = torch.matmul(attention_weights, value) # torch.Size([2, 3])
print(output)

