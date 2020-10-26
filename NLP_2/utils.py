import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import jieba


class Vocabulary(object):
    PAD = '<PAD>'
    UNK = '<UNK>'
    def __init__(self):
        self.token2id = {self.PAD: 0, self.UNK: 1}
        self.id2token = {0: self.PAD, 1: self.UNK}

    def add_token(self, token):
        if token not in self.token2id:
            self.token2id[token] = len(self.token2id)
            self.id2token[self.token2id[token]] = token

    def __len__(self):
        return len(self.token2id)

    def encode(self, text):
        return [self.token2id.get(x, self.token2id[self.UNK]) for x in text]

    def decode(self, ids):
        return [self.id2token.get(x) for x in ids]


class MyDataset(Dataset):
    def __init__(self, labels, inputs, train=True):
        self.labels = labels
        self.inputs = inputs
        self.train = train

    def __getitem__(self, item):
        return torch.LongTensor(self.labels[item]), \
               torch.LongTensor(self.inputs[item])

    def __len__(self):
        return len(self.inputs)


def collate_fn(data):
    labels, inputs = map(list, zip(*data))
    labels = torch.cat(labels, dim=0)
    inputs = pad_sequence(inputs, batch_first=True)

    if torch.cuda.is_available():
        labels = labels.cuda()
        inputs = inputs.cuda()

    return labels, inputs


def build_dataset(train_path, dev_path, test_path):
    vocab = Vocabulary()  # 构造词典
    def load_data(path, train=True):  # 读取标签和内容
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        labels = []
        inputs = []
        for text in data[1:]:
            text = text.strip()
            label, tokens = [int(text[0])], jieba.lcut(text[2:])
            if train:  # 如果是训练集
                for token in tokens:
                    vocab.add_token(token)  # 把读取得到单词用于构造词典
            tokens = vocab.encode(tokens)
            labels.append(label)
            inputs.append(tokens)
        return labels, inputs
    train_dataset = MyDataset(*load_data(train_path))  # 加载训练集
    dev_dataset = MyDataset(*load_data(dev_path), train=False)  # 加载验证集
    test_dataset = MyDataset(*load_data(test_path), train=False)  # 加载测试集

    return (train_dataset, dev_dataset, test_dataset), vocab
