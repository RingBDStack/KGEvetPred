"""
@author: Li Xi
@file: decoder.py
@time: 2019-05-13 09:46
@desc:
"""
import torch
from torch import nn
from torch.nn import Sigmoid


class Score(nn.Module):
    def __init__(self,
                 size1,
                 size2,
                 threshold):

        super(Score, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.threshold = threshold
        self.sigmoid = Sigmoid()
        self.W_si = nn.Linear(self.size1, 1)
        self.W_sc = nn.Linear(self.size2, 1)

    def forward(self, input1, input2):
        batch_size = input1.size()[0]
        # [batch_size, seq_len-1, 1]

        # print(input1)
        # print(input2)

        s = self.sigmoid(self.W_si(input1) + self.W_sc(input2))
        # print(s)
        # [batch_size, 1]
        s = torch.mean(s, dim=1)

        logits = []
        for item in s:
            if item > self.threshold:
                logits.append(1)
            else:
                logits.append(0)

        # scores = []
        # for item in s:
        #     if item > 1.0:
        #         scores.append(1.0)
        #     elif item < 0.0:
        #         scores.append(0.0)
        #     elif 0.0 <=item <= 1.0:
        #         scores.append(item)
        #     else:
        #         scores.append(1.0)

        logits_f1 = []
        for item in logits:
            if item == 1:
                logits_f1.append([0, 1])
            else:
                logits_f1.append([1, 0])

        # scores = (torch.FloatTensor(scores)).view(batch_size, -1).cuda()
        logits = (torch.LongTensor(logits)).view(batch_size)
        logits_f1 = (torch.LongTensor(logits_f1)).view(batch_size, -1)

        return s, logits, logits_f1
