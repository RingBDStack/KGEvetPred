"""
@author: Li Xi
@file: encoder.py
@time: 2019-05-11 22:36
@desc:
"""
import torch
from torch import nn
from torch.nn import Tanh


class EventEmbedding(nn.Module):
    def __init__(self,
                 args,
                 word_embeddings):

        super(EventEmbedding, self).__init__()

        self.word_embeddings = word_embeddings
        self.embedding_size = args.embedding_size

        self.W_e0 = nn.Linear(self.embedding_size, self.embedding_size)
        self.W_e1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.W_e2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.tanh = Tanh()

    def forward(self, trigger, trigger_agent, trigger_object):
        """
        get event embedding
        :param trigger: trigger sequence
        :param trigger_agent: agent sequence
        :param trigger_object: object sequence
        :return:
        """
        tri = self.word_embeddings(trigger)
        agnt = self.word_embeddings(trigger_agent)
        obj = self.word_embeddings(trigger_object)

        event = self.W_e0((torch.mean(tri, dim=1)).unsqueeze(1))
        if agnt.size()[1] != 0:
            event += self.W_e1((torch.mean(agnt, dim=1)).unsqueeze(1))
        if agnt.size()[1] != 0:
            event += self.W_e1((torch.mean(agnt, dim=1)).unsqueeze(1))
        if obj.size()[1] != 0:
            event += self.W_e2((torch.mean(obj, dim=1)).unsqueeze(1))
        event = self.tanh(event)
        tri = (torch.mean(tri, dim=1)).unsqueeze(1)

        return tri, event


class EventTypeEmbedding(nn.Module):
    def __init__(self,
                 args,
                 word_embeddings):

        super(EventTypeEmbedding, self).__init__()

        self.word_embeddings = word_embeddings
        self.embedding_size = args.embedding_size

    def forward(self, event_type):
        event_type = self.word_embeddings(event_type)
        event_type = (torch.mean(event_type, dim=1)).unsqueeze(1)
        return event_type
