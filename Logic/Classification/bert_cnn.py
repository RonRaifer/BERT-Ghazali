import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class Bert_KCNN(nn.Module):
    """
    Our CNN Network class.

    Params:
        - embed_num(`int`):
          The length of the input (number of tokens in sample).

        - embed_dim(`int`):
          The dimension of the input. For out task, its 768.

        - class_num(`int`):
          The number of classes at the output (to be classified to), in our task, its 1.

        - kernel_num(`int`):
          Number of kernels/filters for Convs.

        - kernel_sizes(`list.int`):
          A list containing integer. Represents the kernel sizes.

        - dropout(`double`):
          Dropout regularization value.

        - static(`bool`):
          If true, we use ```autograd.Variable``` on the input. Else, just use the input as is.

    Returns:
    sigmoid(logit), a number between (0,1) indicates how much the sample related to the class.
    """
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
