import torch
import torch.nn as nn
import torch.nn.functional as F

# B表示batch size D表示embedding size L表示序列的长度 H表示cnn out channels的维度

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, filter_sizes, dropout):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)  # 词嵌入
        self.convs = nn.ModuleList([nn.Conv1d(embedding_size, hidden_size, k) for k in filter_sizes])  # 卷积层
        # 指该层的神经元在每次迭代训练时会随机有%dropout的可能性被丢弃（失活），不参与训练,避免拟合过度
        self.dropout = nn.Dropout(dropout)
        # 全连接层，把最终的特征向量的大小转换成类别的大小，以此作为前向传播的输出
        self.fc = nn.Linear(hidden_size * len(filter_sizes), 2)

    def conv_and_pool(self, x, conv):
        # 激活函数层
        x = F.relu(conv(x))  # (B, D, L) -> (B, H, L)
        # 池化层
        x = F.max_pool1d(x, x.size(2)).squeeze()  # (B, H, L) -> (B, H)
        return x

    def forward(self, x):
        emb = self.embedding(x)  # (B, L) -> (B, L, D)
        emb = self.dropout(emb)
        emb = emb.transpose(1, 2)  # (B, L, D) -> (B, D, L)
        out = torch.cat([self.conv_and_pool(emb, conv) for conv in self.convs], -1)  # 3 * (B, D, L) -> (B, 3 * H)
        out = self.fc(out)  # (B, 3 * H) -> (B, 2)
        out = F.log_softmax(out, dim=-1)
        return out
