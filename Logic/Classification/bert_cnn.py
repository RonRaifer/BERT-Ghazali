import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class Bert_KCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static):
        super(Bert_KCNN, self).__init__()

        en = embed_num
        ed = embed_dim
        cn = class_num
        kn = kernel_num
        ks = kernel_sizes

        self.static = static
        self.embed = nn.Embedding(en, ed)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kn, (kernel, ed)) for kernel in ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(ks) * kn, cn)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.static:
            x = autograd.Variable(x)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(ks)*cn)
        logit = self.fc1(x)  # (N, C)

        return self.sigmoid(logit)
