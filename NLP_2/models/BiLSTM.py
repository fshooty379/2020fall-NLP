import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)  # 词嵌入
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        # 指该层的神经元在每次迭代训练时会随机有%dropout的可能性被丢弃（失活），不参与训练,避免拟合过度
        self.dropout = nn.Dropout(dropout)
        # 全连接层，把最终的特征向量的大小转换成类别的大小，以此作为前向传播的输出
        self.fc = nn.Linear(hidden_size * 2, 2)


    def forward(self, x):
        emb = self.embedding(x)  # (B, L) -> (B, L, D)
        emb = self.dropout(emb)
        lstm_out, _ = self.lstm(emb)  # (B, L, H*2)
        out = self.fc(lstm_out[:, -1, :])  # (B, 2 * H) -> (B, 2)
        out = F.log_softmax(out, dim=-1)
        return out

