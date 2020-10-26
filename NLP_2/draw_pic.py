import argparse
import utils
import random
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from models import CNN, BiLSTM
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(model.parameters(), args.learning_rate)

    def train(self, data_loader):
        self.model.train()

        loss_list = []
        pred_list = []
        label_list = []
        for labels, inputs in data_loader:
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
            label_list.append(labels.cpu().numpy())

        y_pred = np.concatenate(pred_list)
        y_true = np.concatenate(label_list)

        loss = np.mean(loss_list)
        acc = accuracy_score(y_true, y_pred)

        return loss, acc

    def evaluate(self, data_loader):
        self.model.eval()

        loss_list = []
        pred_list = []
        label_list = []
        with torch.no_grad():
            for labels, inputs in data_loader:
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                loss_list.append(loss.item())
                pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
                label_list.append(labels.cpu().numpy())

        y_pred = np.concatenate(pred_list)
        y_true = np.concatenate(label_list)

        loss = np.mean(loss_list)
        acc = accuracy_score(y_true, y_pred)

        return loss, acc

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', type=str, default='./data/train.csv')
    parser.add_argument('-dev_file', type=str, default='./data/dev.csv')
    parser.add_argument('-test_file', type=str, default='./data/test.csv')
    parser.add_argument('-save_path', type=str, default='./model.pkl')
    parser.add_argument('-model', type=str, default="cnn", help="[cnn, bilstm]")

    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-embedding_size', type=int, default=128)
    parser.add_argument('-hidden_size', type=int, default=128)
    parser.add_argument('-learning_rate', type=float, default=1e-3)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-epochs', type=int, default=20)

    parser.add_argument('-seed', type=int, default=1)
    args = parser.parse_args()

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("Loading Data...")
    dataests, vocab = utils.build_dataset(args.train_file,
                                          args.dev_file,
                                          args.test_file)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=args.batch_size,
                   collate_fn=utils.collate_fn,
                   shuffle=dataset.train)
        for i, dataset in enumerate(dataests))

    print("Building Model...")
    # if args.model == "cnn":
    #     model = CNN.Model(vocab_size=len(vocab),
    #                       embedding_size=args.embedding_size,
    #                       hidden_size=args.hidden_size,
    #                       filter_sizes=[3, 4, 5],
    #                       dropout=args.dropout)
    # elif args.model == "bilstm":
    #     model = BiLSTM.Model(vocab_size=len(vocab),
    #                          embedding_size=args.embedding_size,
    #                          hidden_size=args.hidden_size,
    #                          dropout=args.dropout)
    # if torch.cuda.is_available():
    #     model = model.cuda()

    train_loss_list = []
    dev_loss_list = []
    for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print("dropout:{}".format(dropout))
        print("??")
        model = CNN.Model(vocab_size=len(vocab),
                          embedding_size=args.embedding_size,
                          hidden_size=args.hidden_size,
                          filter_sizes=[3, 4, 5],
                          dropout=dropout)
        if torch.cuda.is_available():
            model = model.cuda()
        trainer = Trainer(model)
        best_acc = 0
        train_loss_list.append([])
        num = 0
        for i in range(args.epochs):
            print("Epoch: {} ################################".format(i))
            train_loss, train_acc = trainer.train(train_loader)
            train_loss_list.append(train_loss)
            dev_loss, dev_acc = trainer.evaluate(dev_loader)
            dev_loss_list.append(dev_loss)
            print("Train Loss: {:.4f} Acc: {:.4f}".format(train_loss, train_acc))
            print("Dev   Loss: {:.4f} Acc: {:.4f}".format(dev_loss, dev_acc))
            if dev_acc > best_acc:
                best_acc = dev_acc
                trainer.save(args.save_path)
            #print("###########################################")
        trainer.load(args.save_path)
        test_loss, test_acc = trainer.evaluate(test_loader)
        print("Test   Loss: {:.4f} Acc: {:.4f}".format(test_loss, test_acc))
        #print("!!!!!!!!!!")
    print("train~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(train_loss_list)
    print("dev~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(dev_loss_list)
    print("1")
