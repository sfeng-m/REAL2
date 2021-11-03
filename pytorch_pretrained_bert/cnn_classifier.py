import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Model(nn.Module):
    
    def __init__(self, config):
        super(CNN_Model, self).__init__()
        self.static = False
        self.dropout_prob = 0.5
        self.class_num = 256  # 得到256维的向量编码表征
        self.cnn_kernel_num = 100
        self.cnn_kernel_sizes = [3, 4, 5]
        
        V = config.vocab_size
        D = config.hidden_size
        C = self.class_num
        Ci = 1
        Co = self.cnn_kernel_num
        Ks = self.cnn_kernel_sizes
        # Ks = [int(k) for k in args.cnn_kernel_sizes.split(',')]

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        if self.static:
            self.embed.weight.requires_grad = False

    def forward(self, x_id):
        x_emb = self.embed(x_id)  # (N, W, D)
    
        x = x_emb.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)

        return logit


'''Recurrent Convolutional Neural Networks for Text Classification'''

class RCNN_Model(nn.Module):
    def __init__(self, config):
        super(RCNN_Model, self).__init__()

        vocab_size = config.vocab_size
        embed_size = config.hidden_size
        hidden_size = config.hidden_size
        num_layers = 2
        dropout_prob = 0.5
        pad_size = 192
        class_num = 256  # 得到256维的向量编码表征

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=vocab_size-1)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout_prob)
        self.maxpool = nn.MaxPool1d(pad_size)
        self.fc = nn.Linear(hidden_size * 2 + embed_size, class_num)

    def forward(self, x):
        # x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out




