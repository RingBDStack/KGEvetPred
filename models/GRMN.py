"""
@author: Li Xi
@file: GRMN.py
@time: 2019-04-28 14:17
@desc: recurrent memory network with gate
"""
import argparse
import os
import sys
from typing import Dict

from allennlp.training import Trainer

from models.layers.attention import Attention
from models.layers.decoder import Score
from models.layers.dynamic_rnn import DynamicLSTM
from models.layers.encoder import EventEmbedding

sys.path.append('..')

import torch
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import BooleanAccuracy, F1Measure
from overrides import overrides
from torch import nn
from torch import optim
from torch.nn import Embedding, Tanh, Sigmoid, BCELoss

from models.event_reader import EventDataReader


@Model.register("GRMN")
class GRMN(Model):

    def __init__(self,
                 args,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)

        # parameters
        self.args = args
        self.word_embeddings = word_embeddings

        # gate
        self.x_linear = nn.Linear(self.args.embedding_size, self.args.hidden_size, bias=False)
        self.W_z = nn.Linear(self.args.hidden_size, 1, bias=False)
        self.U_z = nn.Linear(self.args.hidden_size, 1, bias=False)
        self.W_r = nn.Linear(self.args.hidden_size, 1, bias=False)
        self.U_r = nn.Linear(self.args.hidden_size, 1, bias=False)
        self.W = nn.Linear(self.args.hidden_size, 1, bias=False)
        self.U = nn.Linear(self.args.hidden_size, 1, bias=False)

        # layers
        self.event_embedding = EventEmbedding(args, self.word_embeddings)
        self.lstm = DynamicLSTM(self.args.embedding_size, self.args.hidden_size, num_layers=1, batch_first=True)
        self.attention = Attention(self.args.hidden_size, score_function='mlp')
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.score = Score(self.args.hidden_size, self.args.embedding_size, threshold=self.args.threshold)

        # metrics
        self.accuracy = BooleanAccuracy()
        self.f1_score = F1Measure(positive_label=1)
        self.loss_function = BCELoss()

    def gated_atten(self, vt_1, atten_input):
        """
        gated attention block
        :param vt_1: v_t-1
        :param atten_input: [h1, h2, ... ,h_n-1]
        :return: v_t
        """
        # [batch_size, 1, hidden_size]
        out_at, _ = self.attention(atten_input, vt_1)
        # [batch_size, hidden_size]
        h_e = torch.sum(out_at * atten_input, dim=1)
        # [batch_size, 1]
        z = (self.sigmoid(self.W_z(h_e.unsqueeze(1)) + self.U_z(vt_1))).squeeze(1)
        # [batch_size, 1]
        r = (self.sigmoid(self.W_r(h_e.unsqueeze(1)) + self.U_r(vt_1))).squeeze(1)
        # [batch_size, 1]
        h = self.tanh(self.W(h_e.unsqueeze(1)) + self.U((torch.mul(r, vt_1.squeeze(1))).unsqueeze(1))).squeeze(1)
        # [baych_size, 1, hidden_size]
        vt = (torch.mul((1 - z), vt_1.squeeze(1)) + torch.mul(z, h)).unsqueeze(1)

        return vt

    @overrides
    def forward(self,
                trigger_0: Dict[str, torch.LongTensor],
                trigger_agent_0: Dict[str, torch.LongTensor],
                agent_attri_0: Dict[str, torch.LongTensor],
                trigger_object_0: Dict[str, torch.LongTensor],
                object_attri_0: Dict[str, torch.LongTensor],
                trigger_1: Dict[str, torch.LongTensor],
                trigger_agent_1: Dict[str, torch.LongTensor],
                agent_attri_1: Dict[str, torch.LongTensor],
                trigger_object_1: Dict[str, torch.LongTensor],
                object_attri_1: Dict[str, torch.LongTensor],
                trigger_2: Dict[str, torch.LongTensor],
                trigger_agent_2: Dict[str, torch.LongTensor],
                agent_attri_2: Dict[str, torch.LongTensor],
                trigger_object_2: Dict[str, torch.LongTensor],
                object_attri_2: Dict[str, torch.LongTensor],
                trigger_3: Dict[str, torch.LongTensor],
                trigger_agent_3: Dict[str, torch.LongTensor],
                agent_attri_3: Dict[str, torch.LongTensor],
                trigger_object_3: Dict[str, torch.LongTensor],
                object_attri_3: Dict[str, torch.LongTensor],
                trigger_4: Dict[str, torch.LongTensor],
                trigger_agent_4: Dict[str, torch.LongTensor],
                agent_attri_4: Dict[str, torch.LongTensor],
                trigger_object_4: Dict[str, torch.LongTensor],
                object_attri_4: Dict[str, torch.LongTensor],
                event_type: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        # tri, e: [batch_size, 1, embedding_size]
        tri0, e0 = self.event_embedding(trigger_0, trigger_agent_0, trigger_object_0)
        tri1, e1 = self.event_embedding(trigger_1, trigger_agent_1, trigger_object_1)
        tri2, e2 = self.event_embedding(trigger_2, trigger_agent_2, trigger_object_2)
        tri3, e3 = self.event_embedding(trigger_3, trigger_agent_3, trigger_object_3)
        tri4, e4 = self.event_embedding(trigger_4, trigger_agent_4, trigger_object_4)
        # [batch_size, seq_Len, embedding_size]
        e = (torch.stack([e0, e1, e2, e3, e4], dim=1)).squeeze(2)

        batch_size = e.size()[0]
        # [seq_Len, batch_size, hidden_size]
        lstm_output, (_, _) = self.lstm(e, torch.LongTensor([5] * batch_size))
        # [batch_size, seq_len-1, atten_size]

        atten = lstm_output[:, :4, :]
        # [batch_size, 1, hidden_size]
        vt = self.x_linear(tri4)
        for i in range(self.args.hop_num):
            # [batch_size, 1, hidden_size]
            vt = self.gated_atten(vt, atten)

        # [batch_size, 1] , [batch_size], [batch_size, label_size]
        score, logits, logits_f1 = self.score(atten, tri4)

        output = {"logits": logits,
                  "score": score}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_score(logits_f1, label)
            output["loss"] = self.loss_function(score.squeeze(1), label.float())

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self.accuracy.get_metric(reset)
        precision, recall, f1_measure = self.f1_score.get_metric(reset)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_measure": f1_measure
        }


if __name__ == '__main__':
    # init parameters
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.embedding_size = 300
    args.learning_rate = 1e-4
    args.batch_size = 1
    args.epochs = 10
    args.patience = 1
    args.cuda_device = -1
    args.hidden_size = 8
    args.hop_num = 6
    args.label_num = 2
    args.threshold = 0.5

    # load data
    reader = EventDataReader()
    train_dataset = ensure_list(reader.read(os.path.join('..', 'dataset', 'example', 'train.data')))
    eval_dataset = ensure_list(reader.read(os.path.join('..', 'dataset', 'example', 'train.data')))
    test_dataset = ensure_list(reader.read(os.path.join('..', 'dataset', 'example', 'train.data')))

    # get vocabulary and embedding
    vocab = Vocabulary.from_instances(train_dataset + eval_dataset,
                                      min_count={"trigger_0": 0,
                                                 "trigger_agent_0": 0,
                                                 "agent_attri_0": 0,
                                                 "trigger_object_0": 0,
                                                 "object_attri_0": 0,
                                                 "trigger_1": 0,
                                                 "trigger_agent_1": 0,
                                                 "agent_attri_1": 0,
                                                 "trigger_object_1": 0,
                                                 "object_attri_1": 0,
                                                 "trigger_2": 0,
                                                 "trigger_agent_2": 0,
                                                 "agent_attri_2": 0,
                                                 "trigger_object_2": 0,
                                                 "object_attri_2": 0,
                                                 "trigger_3": 0,
                                                 "trigger_agent_3": 0,
                                                 "agent_attri_3": 0,
                                                 "trigger_object_3": 0,
                                                 "object_attri_3": 0,
                                                 "trigger_4": 0,
                                                 "trigger_agent_4": 0,
                                                 "agent_attri_4": 0,
                                                 "trigger_object_4": 0,
                                                 "object_attri_4": 0})
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=args.embedding_size)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    # init model
    model = GRMN(args, word_embeddings, vocab)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    iterator = BucketIterator(batch_size=args.batch_size, sorting_keys=[("trigger_0", "num_tokens")])
    iterator.index_with(vocab)

    # init trainer
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=eval_dataset,
                      shuffle=False,  # to ensure label 0 has a label id 0
                      num_epochs=args.epochs,
                      patience=args.patience,  # stop training before loss raise
                      cuda_device=args.cuda_device,  # cuda device id
                      )

    # start train
    metrics = trainer.train()
